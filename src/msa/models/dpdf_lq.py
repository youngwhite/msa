"""DPDF-LQ — Dual-Path Dynamic Fusion with Learnable Query (Zhou et al., EMNLP 2025).

The first method here from the 2024-2026 wave. Two paths run in parallel and a
gate decides how much of each survives: a **global path** where a set of
learnable queries attends across one summary token per modality, and a **local
path** where the full audio and vision sequences cross-attend against text.

Structure follows the authors' implementation (`ZhouMiaoGX/DPDF-LQ`, MIT);
hyper-parameters come from the paper's Table 3, which states them in full --
query length 8 and width 128, DGLQA depth 3, cross-attention depth 2 on both
paths, lr 1e-4, weight decay 1e-4, batch 64. The loss is a single MSE term
(eq. 36): no auxiliary objectives, no staged training.

**Reading the paper alone would have produced a different model.** Four things
are only visible in the code, and the first two change what is being measured:

* **BERT is fine-tuned**, from `text_bert` token ids -- and each path builds its
  **own** encoder rather than sharing one. A port reading "BERT for text" as
  frozen features would sit a generation behind: in this project's own table,
  the step to fine-tuned BERT is worth eight Acc-7 points.
* The global path reads `fusion_layer(...)[:, 0]`, the shared CLS token of the
  cross-transformer, not a mean over the sequence.
* Gating happens **twice**: once inside the global path over
  `[queries; text; fused]`, and once between the two paths with a softmax over
  two weights.
* The learnable queries are initialised to **ones**, not to a truncated normal.

The transformer blocks are ALMT's, reused rather than rewritten: the authors
build both models from the same `Transformer` / `CrossTransformer` classes, and
this repository's copies already passed a weight-copy equivalence test against
that reference at 0.000e+00 (`scripts/check_almt_equivalence.py`).

**A defect inherited from the reference, recorded not reproduced.** The global
path constructs `conv_att_v = ConvModulOperationSpatialAttention(...)` and never
calls it -- vision goes through `proj_v` directly. The same module *is* used in
the local path. It is the "declared but never reached" form from convention 5,
and it inflates the reference's parameter count without touching its output. We
do not construct it in the global path; see docs/spec_confede_dpdflq.md.

Also noted: the reference hard-codes `source_num_frames=58` in the local path
(8 + 50), which breaks on any other sequence length. Here it is derived.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from .almt import _CrossTransformer, _PositionalEncoder, _TokenTransformer
from .base import MSAModel
from .bert import BertTextEncoder


class _ChannelAttention(nn.Module):
    """Squeeze-and-excite over the feature axis (`ChannelAttentionBlock`)."""

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(dim, dim // reduction), nn.GELU(),
            nn.Linear(dim // reduction, dim), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.se(x.transpose(1, 2)).unsqueeze(-1).transpose(1, 2)
        return x * weight


class _DGLQAttention(nn.Module):
    """Dynamic Global Learnable Query Attention.

    One query set, three key/value projections -- one per modality -- and a gate
    over the modality means decides how the three attention outputs are mixed.
    Contrast with ALMT's hyper-modality layer, which displaces a single stream
    by each modality in turn; here the three run in parallel and are weighted.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.0) -> None:
        super().__init__()
        inner = dim_head * heads
        self.heads, self.dim_head = heads, dim_head
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(3 * dim, inner * 3, bias=False)
        self.to_v = nn.Linear(3 * dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(dim * 3, 3), nn.Softmax(dim=-1))

    def forward(self, text: torch.Tensor, audio: torch.Tensor,
                vision: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        batch = text.shape[0]
        joint = torch.cat([text, audio, vision], dim=-1)

        def split(x: torch.Tensor, chunks: int) -> torch.Tensor:
            return (x.view(batch, -1, chunks, self.heads, self.dim_head)
                     .permute(2, 0, 3, 1, 4))

        k = split(self.to_k(joint), 3)                       # (3, b, h, n, d)
        v = split(self.to_v(joint), 3)
        q = (self.to_q(queries).view(batch, -1, self.heads, self.dim_head)
             .permute(0, 2, 1, 3))                           # (b, h, m, d)

        attention = torch.softmax(
            torch.einsum("bhid,cbhjd->cbhij", q, k) * self.scale, dim=-1)
        out = torch.einsum("cbhij,cbhjd->cbhid", attention, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(3, batch, -1, self.heads * self.dim_head)

        weights = self.gate(joint.mean(dim=1)).unsqueeze(1)   # (b, 1, 3)
        fused = sum(weights[:, :, i].unsqueeze(-1) * out[i] for i in range(3))
        return queries + self.to_out(fused)


class _PreNormDGLQA(nn.Module):
    """LayerNorm on all four inputs, then the attention (`PreNormAHL`)."""

    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, text: torch.Tensor, audio: torch.Tensor,
                vision: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm1(text), self.norm2(audio),
                       self.norm3(vision), self.norm4(queries))


class _DGLQAEncoder(nn.Module):
    """Depth-N DGLQA. Layer i reads the i-th hidden state of the text encoder.

    Grouped exactly as the reference groups it -- one (PreNorm+attention,
    channel attention) pair per layer -- so parameter order matches and the
    equivalence test can copy positionally.
    """

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleList([
                _PreNormDGLQA(dim, _DGLQAttention(dim, heads, dim_head, dropout)),
                _ChannelAttention(dim),
            ])
            for _ in range(depth)
        )

    def forward(self, text_levels: list[torch.Tensor], audio: torch.Tensor,
                vision: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        for index, (attend, channel) in enumerate(self.layers):
            queries = queries + attend(text_levels[index], audio, vision, queries)
            queries = queries + channel(queries)
        return queries


class _ConvSpatialAttention(nn.Module):
    """Convolutional modulation over the vision stream (local path only).

    The reference treats the sequence as a height-1 image and uses 2-D
    convolutions -- which is what the `unsqueeze(-1)` / `permute` / `squeeze(-1)`
    dance around the call site is for. Kept as 2-D so the weights are the same
    tensors, not merely the same arithmetic.
    """

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        # Channels-first LayerNorm: normalise over the channel axis at each
        # position. nn.GroupNorm(1, dim) is the obvious substitute and is NOT
        # equivalent -- it normalises over channels and space together, which
        # cost 4.8e-01 in the local path until the equivalence test found it.
        self.norm_weight = nn.Parameter(torch.ones(dim))
        self.norm_bias = nn.Parameter(torch.zeros(dim))
        self.norm_eps = 1e-6
        self.att = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim),
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, steps, dim) -> (batch, dim, steps, 1) and back.
        x = x.transpose(1, 2).unsqueeze(-1)
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.norm_eps)
        x = self.norm_weight[:, None, None] * x + self.norm_bias[:, None, None]
        x = self.proj(self.att(x) * self.v(x))
        return x.squeeze(-1).transpose(1, 2)


class _Streams(nn.Module):
    """The per-path front end: its own BERT, then one projection per modality."""

    def __init__(self, dims: dict[str, int], lengths: dict[str, int], width: int,
                 token_len: int, depth: int, heads: int, mlp_dim: float,
                 dropout: float, finetune: bool, pretrained: str) -> None:
        super().__init__()
        self.encoder = BertTextEncoder(pretrained, finetune)
        # One Sequential per modality, in the reference's order (proj_l, proj_a,
        # proj_v), so parameter registration order matches and the equivalence
        # test can copy weights positionally.
        self.stream = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(dims[m], width),
                _TokenTransformer(lengths[m], token_len, width, depth, heads,
                                  mlp_dim, dropout=dropout),
            )
            for m in ("text", "audio", "vision")
        })

    def forward(self, batch: dict[str, torch.Tensor],
                vision_gate: nn.Module | None = None,
                keep_all: bool = False) -> dict[str, torch.Tensor]:
        text = self.encoder(batch["text_bert"])
        raw = {"text": text, "audio": batch["audio"], "vision": batch["vision"]}
        if vision_gate is not None:
            raw["vision"] = vision_gate(raw["vision"])
        out = {}
        for modality, value in raw.items():
            projected = self.stream[modality][0](value)
            out[modality] = self.stream[modality][1](projected, keep_all=keep_all)
        return out


@register_model("dpdf_lq")
class DualPathDynamicFusion(MSAModel):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        text_length: int = 50,
        audio_length: int = 50,
        vision_length: int = 50,
        width: int = 128,
        token_len: int = 8,
        proj_depth: int = 1,
        proj_heads: int = 8,
        dglqa_depth: int = 3,
        dglqa_heads: int = 8,
        dim_head: int = 64,
        # The reference sizes DGLQA's heads at 16, not the 64 its other blocks use.
        dglqa_dim_head: int = 16,
        fusion_depth: int = 2,
        fusion_heads: int = 8,
        local_depth: int = 2,
        # The reference sizes this pair 58/58 (8 tokens + 50 frames), hard-coded.
        local_source_len: int | None = None,
        local_target_len: int | None = None,
        dropout: float = 0.2,
        pretrained: str = "bert-base-uncased",
        finetune: bool = True,
    ) -> None:
        super().__init__()
        # The text stream enters as BERT's hidden width, not the dataset's
        # feature width, because BERT is fine-tuned here rather than frozen.
        self.token_len = token_len
        dims = {"text": 768, "audio": audio_dim, "vision": vision_dim}
        lengths = {"text": text_length, "audio": audio_length, "vision": vision_length}
        streams = dict(dims=dims, lengths=lengths, width=width, token_len=token_len,
                       depth=proj_depth, heads=proj_heads, mlp_dim=width,
                       dropout=dropout, finetune=finetune, pretrained=pretrained)

        # --- global path, declared in the reference's order -------------------
        self.queries = nn.Parameter(torch.ones(1, token_len, width))
        self.global_streams = _Streams(**streams)
        self.text_levels = _PositionalEncoder(token_len, width, dglqa_depth - 1,
                                              proj_heads, dim_head, width, dropout)
        self.dglqa = _DGLQAEncoder(width, dglqa_depth, dglqa_heads, dglqa_dim_head, dropout)
        self.fusion = _CrossTransformer(token_len, token_len, width, fusion_depth,
                                        fusion_heads, dim_head, width, dropout)
        self.inner_gate = nn.Sequential(nn.Linear(width * 3, width), nn.Sigmoid())
        self.global_dropout = nn.Dropout(dropout)
        # kernel 3 without padding, as the reference: 8 query tokens become 6.
        self.depthwise = nn.Sequential(
            nn.Conv1d(width, width, 3, groups=width), nn.GELU(), nn.BatchNorm1d(width))

        # --- local path ------------------------------------------------------
        self.local_streams = _Streams(**streams)
        self.vision_gate = _ConvSpatialAttention(vision_dim)
        # The local target is audio and vision concatenated, so it is twice the
        # token length -- the reference hard-codes 58 here and would break on any
        # other sequence length.
        self.local_fusion = _CrossTransformer(
            local_source_len or token_len, local_target_len or token_len * 2,
            width, local_depth, fusion_heads, dim_head, width, dropout)

        # --- between paths ---------------------------------------------------
        self.path_gate = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 2), nn.Softmax(dim=-1))
        self.head = nn.Sequential(
            nn.Linear(width, width // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width // 2, 1))

    def _global(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        streams = self.global_streams(batch)
        levels = self.text_levels(streams["text"], save_hidden=True)
        queries = self.queries.expand(streams["text"].shape[0], -1, -1)
        queries = self.dglqa(levels, streams["audio"], streams["vision"], queries)

        queries = self.depthwise(queries.transpose(1, 2)).transpose(1, 2)
        # `[:, 0]` is the cross-transformer's shared CLS token, not a content one.
        # The first call reads the FULL text level; only the second one is
        # truncated to the post-convolution query length. Getting this wrong is
        # worth 6.6e-03 at the output, which the equivalence test caught.
        fused = self.fusion(queries, levels[-1])[:, 0]

        text_tail = levels[-1][:, : queries.shape[1]]
        gate = self.inner_gate(torch.cat(
            [queries, text_tail, fused.unsqueeze(1).expand(-1, queries.shape[1], -1)],
            dim=-1))
        queries = gate * queries + (1 - gate) * text_tail
        return self.global_dropout(self.fusion(queries, text_tail)[:, 0])

    def _local(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # The local path keeps the full sequence; only the global path slices.
        streams = self.local_streams(batch, vision_gate=self.vision_gate, keep_all=True)
        # Text is the source, audio the target, and vision is appended to the
        # target *after* positional embedding -- the reference calls
        # `fusion_transformer(text, audio, video)` and its signature routes the
        # third argument to `additional_x`, which is concatenated post-position.
        # So vision enters this path carrying no positional information.
        return self.local_fusion(streams["text"], streams["audio"],
                                 streams["vision"])[:, 0]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        global_feature = self._global(batch)
        local_feature = self._local(batch)
        gate = self.path_gate(torch.cat([global_feature, local_feature], dim=1))
        fused = gate[:, 0:1] * global_feature + gate[:, 1:2] * local_feature
        return {"M": self.head(fused).squeeze(-1)}

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        # Both paths fine-tune their own BERT; the usual smaller rate for a
        # pretrained encoder, as every other fine-tuning model here does.
        bert, rest = [], []
        for name, parameter in self.named_parameters():
            (bert if ".encoder.bert." in name else rest).append(parameter)
        return [
            {"params": bert, "lr": lr * 0.1, "weight_decay": weight_decay},
            {"params": rest, "lr": lr, "weight_decay": weight_decay},
        ]

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Eq. 36: mean squared error and nothing else. The paper argues that
        # simplicity is a feature, so an auxiliary term here would be a
        # different method.
        return F.mse_loss(outputs["M"], batch["label"])
