"""ALMT — Adaptive Language-guided Multimodal Transformer (Zhang et al., EMNLP 2023).

Every model before this one lets audio and vision speak for themselves, either
directly (MulT's cross-modal blocks) or through text as a mediator (TETFN). ALMT
takes a stronger position: the non-linguistic modalities never become part of the
representation at all. What they do is *shift* a representation that language
owns.

The mechanism is a **hyper-modality** — a learned tensor, the same for every
sample, that is refined layer by layer. At each of the three AHL layers, text
supplies the queries and audio and vision supply the keys and values; whatever
they contribute is added to the hyper-modality as a displacement. Text is then
fused with the result and the first token is read out.

The shape of it:

    each modality -> Linear -> Transformer with 8 learned tokens -> keep those 8
    text  -> l_encoder (depth AHL_depth-1, keeping every hidden state)
    hyper -> AHL x3, guided by text level i, absorbing audio and vision
    fuse(hyper, text_last)[:, 0] -> Linear(dim, 1)

Ported from MMSA (`models/singleTask/ALMT.py`, MIT, THUIAR) and checked against
the authors' repository (`Haoyu-ha/ALMT`): all six layer classes are line-for-line
identical between them, and the main model differs only in how the config is
unpacked. This is the cleanest of MMSA's three unpublished ports.

Written with plain tensor ops rather than einops, which the reference uses and
this project does not depend on. `rearrange(t, 'b n (h d) -> b h n d')` is a
reshape and a transpose; spelling it out costs a line and removes a dependency.

Four things about this model differ from every other one here, and all four are
the authors' own choices, not MMSA's:

* the loss is **MSE**, not L1 — the only such model in this storyline;
* the optimiser is **AdamW**, so the weight decay is decoupled;
* the schedule is a linear warmup into cosine annealing, and its **first epoch
  runs at a learning rate of exactly zero** (see docs/investigations.md#almt-warmup);
* `max_grad_norm: 2` appears in MMSA's config and **neither implementation ever
  clips**. Unlike MulT's phantom weight decay, this one is inert on both sides,
  so it is a dead config key rather than a deviation.

Uses **unaligned** data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DatasetSpec
from ..registry import register_model
from .base import MSAModel
from .bert import BertTextEncoder


def _split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    """(b, n, heads*d) -> (b, heads, n, d)."""
    b, n, _ = x.shape
    return x.view(b, n, heads, -1).transpose(1, 2)


def _join_heads(x: torch.Tensor) -> torch.Tensor:
    """(b, heads, n, d) -> (b, n, heads*d)."""
    b, h, n, d = x.shape
    return x.transpose(1, 2).reshape(b, n, h * d)


class _Attention(nn.Module):
    """Ordinary scaled dot-product attention over separate q/k/v inputs.

    `dim_head` defaults to 64 while `dim` is 128 and `heads` is 8, so the inner
    width is 512 — four times the model width. That is the authors' default and
    neither they nor MMSA override it for the embedding blocks, while the AHL
    layers pass `dim_head = dim / heads` and so run at width 128. The asymmetry
    is theirs; it is reproduced, not tidied.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.0) -> None:
        super().__init__()
        inner = dim_head * heads
        self.heads, self.scale = heads, dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = (nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
                       if not (heads == 1 and dim_head == dim) else nn.Identity())

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q, k, v = (_split_heads(f(t), self.heads)
                   for f, t in ((self.to_q, q), (self.to_k, k), (self.to_v, v)))
        attention = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        return self.to_out(_join_heads(attention @ v))


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EncoderLayer(nn.Module):
    """Pre-norm attention and feed-forward, each with a residual added outside.

    Attention normalises its three inputs independently, which matters when q
    comes from one stream and k/v from another.
    """

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: float,
                 dropout: float) -> None:
        super().__init__()
        self.norm_q, self.norm_k, self.norm_v = (nn.LayerNorm(dim) for _ in range(3))
        self.attention = _Attention(dim, heads, dim_head, dropout)
        self.norm_ff = nn.LayerNorm(dim)
        self.feed_forward = _FeedForward(dim, int(mlp_dim), dropout)

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.attention(self.norm_q(q), self.norm_k(k), self.norm_v(v))

    def forward_ff(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.norm_ff(x))


class _Encoder(nn.Module):
    """Self-attention stack; optionally returns the input and every layer output."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int,
                 mlp_dim: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _EncoderLayer(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor, save_hidden: bool = False):
        hidden = [x]
        for layer in self.layers:
            x = layer.attend(x, x, x) + x
            x = layer.forward_ff(x) + x
            hidden.append(x)
        return hidden if save_hidden else x


class _CrossEncoder(nn.Module):
    """The target attends to the source; only the target carries residuals."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int,
                 mlp_dim: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _EncoderLayer(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            target = layer.attend(target, source, source) + target
            target = layer.forward_ff(target) + target
        return target


class _TokenTransformer(nn.Module):
    """Prepend `token_len` learned tokens, encode, and keep only those tokens.

    This is how a 375-step audio stream and a 50-token sentence come out the same
    length: the learned tokens attend over the whole sequence and become its
    summary, and everything else is discarded by the caller.
    """

    def __init__(self, num_frames: int, token_len: int, dim: int, depth: int,
                 heads: int, mlp_dim: float, dim_head: int = 64,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.token_len = token_len
        self.position = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
        self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        self.encoder = _Encoder(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x: torch.Tensor, keep_all: bool = False) -> torch.Tensor:
        b, n, _ = x.shape
        tokens = self.extra_token.expand(b, -1, -1)
        x = torch.cat((tokens, x), dim=1) + self.position[:, : n + self.token_len]
        encoded = self.encoder(x)
        # ALMT always wants just the summary tokens. DPDF-LQ's local path wants
        # the whole sequence -- the reference returns everything and only its
        # *global* path slices, which is where its hard-coded 58 comes from.
        return encoded if keep_all else encoded[:, : self.token_len]


class _PositionalEncoder(nn.Module):
    """Learned positions over an already-short sequence, then self-attention.

    The reference builds this from the same `Transformer` class as
    `_TokenTransformer`, with `token_len=None` — which still creates a positional
    embedding. Easy to miss when reading, and it costs the text stream its only
    notion of order before the AHL layers query it.
    """

    def __init__(self, num_frames: int, dim: int, depth: int, heads: int,
                 dim_head: int, mlp_dim: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, num_frames, dim))
        self.encoder = _Encoder(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x: torch.Tensor, save_hidden: bool = False):
        return self.encoder(x + self.position[:, : x.size(1)], save_hidden)


class _CrossTransformer(nn.Module):
    """Positions plus a shared CLS token, then cross-attention.

    The same CLS parameter is prepended to *both* streams, and it is the token
    the model reads out at the end — `[:, 0]` is this, not the first content
    token. Dropping it silently changes what the regression head sees.
    """

    def __init__(self, source_len: int, target_len: int, dim: int, depth: int,
                 heads: int, dim_head: int, mlp_dim: float,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.position_source = nn.Parameter(torch.randn(1, source_len + 1, dim))
        self.position_target = nn.Parameter(torch.randn(1, target_len + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.encoder = _CrossEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, source: torch.Tensor, target: torch.Tensor,
                additional: torch.Tensor | None = None) -> torch.Tensor:
        cls = self.cls_token.expand(source.size(0), -1, -1)
        source = torch.cat((cls, source), dim=1)
        source = source + self.position_source[:, : source.size(1)]
        target = torch.cat((cls, target), dim=1)
        target = target + self.position_target[:, : target.size(1)]
        # `additional` is appended AFTER the positional embedding is applied, so
        # it carries no position information at all. DPDF-LQ's local path relies
        # on this: it passes vision as the third positional argument, which the
        # reference's signature silently routes here.
        if additional is not None:
            target = torch.cat((target, additional), dim=1)
        return self.encoder(source, target)


class _HyperLayer(nn.Module):
    """One refinement of the hyper-modality: text asks, audio and vision answer.

    The two contributions are summed *before* the output projection, so audio and
    vision are never weighted against each other — they are simply added.
    """

    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        inner = dim_head * heads
        self.heads, self.scale = heads, dim_head ** -0.5
        # All four inputs are normalised, the hyper-modality included — so the
        # residual below accumulates onto the normalised copy, not the raw one.
        self.norm_text, self.norm_audio, self.norm_vision, self.norm_hyper = (
            nn.LayerNorm(dim) for _ in range(4)
        )
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k_a = nn.Linear(dim, inner, bias=False)
        self.to_k_v = nn.Linear(dim, inner, bias=False)
        self.to_v_a = nn.Linear(dim, inner, bias=False)
        self.to_v_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, text: torch.Tensor, audio: torch.Tensor, vision: torch.Tensor,
                hyper: torch.Tensor) -> torch.Tensor:
        text, hyper = self.norm_text(text), self.norm_hyper(hyper)
        audio, vision = self.norm_audio(audio), self.norm_vision(vision)
        q = _split_heads(self.to_q(text), self.heads)
        contributions = []
        for to_k, to_v, source in ((self.to_k_a, self.to_v_a, audio),
                                   (self.to_k_v, self.to_v_v, vision)):
            k = _split_heads(to_k(source), self.heads)
            v = _split_heads(to_v(source), self.heads)
            attention = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
            contributions.append(_join_heads(attention @ v))
        return hyper + self.to_out(contributions[0] + contributions[1])


class _HyperEncoder(nn.Module):
    """One `_HyperLayer` per text level; level i reads text hidden state i."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_HyperLayer(dim, heads, dim_head, dropout)
                                    for _ in range(depth))

    def forward(self, text_levels: list[torch.Tensor], audio: torch.Tensor,
                vision: torch.Tensor, hyper: torch.Tensor) -> torch.Tensor:
        for level, layer in enumerate(self.layers):
            hyper = layer(text_levels[level], audio, vision, hyper)
        return hyper


@register_model("almt")
class ALMT(MSAModel):
    """Uses **unaligned** data; the learned tokens do the resampling."""

    @classmethod
    def build(cls, spec: DatasetSpec, **kwargs) -> MSAModel:
        return cls(text_dim=spec.text_dim, audio_dim=spec.audio_dim,
                   vision_dim=spec.vision_dim, **kwargs)

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        # MOSI's unaligned stream lengths. The reference hardcodes the same
        # three numbers in its config (`feature_length: [50, 375, 500]`); they
        # size the positional embedding, which is then sliced to the sequence
        # actually seen, so a shorter stream is fine and a longer one errors.
        text_length: int = 50,
        audio_length: int = 375,
        vision_length: int = 500,
        dim: int = 128,
        token_len: int = 8,
        embedding_depth: int = 1,
        embedding_heads: int = 8,
        embedding_mlp_dim: int = 128,
        text_encoder_heads: int = 8,
        ahl_depth: int = 3,
        ahl_heads: int = 8,
        fusion_depth: int = 2,
        fusion_heads: int = 8,
        fusion_mlp_dim: int = 128,
        pretrained: str = "bert-base-uncased",
        finetune: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = BertTextEncoder(pretrained, finetune)
        self.token_len = token_len
        # Same for every sample and learned; the AHL layers displace a copy of it.
        self.hyper = nn.Parameter(torch.ones(1, token_len, dim))

        def embed(in_dim: int, length: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, dim),
                _TokenTransformer(length, token_len, dim, embedding_depth,
                                  embedding_heads, embedding_mlp_dim),
            )

        self.embed_text = embed(self.encoder.hidden_size, text_length)
        self.embed_audio = embed(audio_dim, audio_length)
        self.embed_vision = embed(vision_dim, vision_length)

        # depth - 1 layers, because save_hidden also returns the input: the three
        # AHL layers read levels 0, 1 and 2 of a two-layer stack.
        self.text_encoder = _PositionalEncoder(token_len, dim, ahl_depth - 1,
                                               text_encoder_heads, dim_head=64,
                                               mlp_dim=dim)
        self.hyper_encoder = _HyperEncoder(dim, ahl_depth, ahl_heads,
                                           dim_head=dim // ahl_heads)
        self.fusion = _CrossTransformer(token_len, token_len, dim, fusion_depth,
                                        fusion_heads, dim_head=64,
                                        mlp_dim=fusion_mlp_dim)
        self.head = nn.Linear(dim, 1)

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """MSE, not L1 — the reference and the authors both use `nn.MSELoss`."""
        return F.mse_loss(outputs["M"], batch["label"])

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text = self.embed_text(self.encoder(batch["text_bert"]))
        audio = self.embed_audio(batch["audio"])
        vision = self.embed_vision(batch["vision"])

        levels = self.text_encoder(text, save_hidden=True)
        hyper = self.hyper.expand(text.size(0), -1, -1)
        hyper = self.hyper_encoder(levels, audio, vision, hyper)
        # Index 0 is the CLS token the fusion layer prepends, not a content token.
        fused = self.fusion(hyper, levels[-1])[:, 0]
        return {"M": self.head(fused).view(-1)}
