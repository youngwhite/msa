"""TETFN — Text Enhanced Transformer Fusion Network (Wang et al., PR 2023).

The last model in the reproduction list, and the one that combines the two
threads that came before it: MulT's cross-modal attention and Self-MM's
self-supervised unimodal labels, both on a fine-tuned BERT.

Its own contribution is what it does to the audio-vision pair. MulT gives every
ordered pair its own cross-modal block, including A->V and V->A directly. TETFN
routes those two through text instead: the source modality first attends to the
text stream, and the target then attends to *that* text-enhanced representation.
Audio and vision never look at each other unmediated.

Ported from MMSA (`models/multiTask/TETFN.py`, MIT, THUIAR). Four things in that
file are declared and inert, and are reproduced as they behave rather than as
they read — see docs/investigations.md#tetfn-inert:

* `conv1d_kernel_size_v` (3 in the config) never reaches the vision branch:
  `video_model` is constructed with `conv1d_kernel_size_a`. A copy-paste, but it
  is what produced the reference number, so vision uses kernel 1 here too.
* `AuViSubNet.forward` takes `lengths` and ignores it — no packing, the LSTM runs
  over the full padded sequence. (Self-MM's identically-named class does pack.)
* `get_network`'s `layers` argument is ignored; every block is two layers.
* Positional encoding is on for the two text-enhanced blocks and off for the
  other seven. Inconsistent within one model, and notable because it shows MMSA
  knew the switch existed — it simply never set it for MulT.

The pseudo-label machinery is not re-implemented: MMSA's `update_labels`,
`update_centers` and `weighted_loss` are byte-identical between TETFN and
Self-MM, so both use `msa.models.pseudo_labels`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DatasetSpec
from ..registry import register_model
from .base import MSAModel
from .bert import BertTextEncoder
from .pseudo_labels import PseudoLabelMixin
from .transformers import TransformerEncoder


class _AudioVisualSubNet(nn.Module):
    """LSTM over the raw stream, then a temporal conv onto the shared width.

    `lengths` is accepted and unused, matching the reference: no packing, and the
    padded steps go through the LSTM like any other.
    """

    def __init__(self, in_size: int, hidden: int, kernel_size: int, out_size: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden, num_layers=1, batch_first=True)
        self.conv = nn.Conv1d(hidden, out_size, kernel_size=kernel_size, bias=False)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        hidden, _ = self.rnn(x)
        return self.conv(hidden.transpose(1, 2))     # (batch, out_size, steps)


class _TextEnhanced(nn.Module):
    """Audio and vision meet only through text.

    `lower` lets the source modality read the text stream; `upper` lets the
    target read that. Both carry positional encoding, unlike the plain
    cross-modal blocks in the same model.
    """

    def __init__(self, dim: int, heads: int, layers: int, attn_dropout: float,
                 relu_dropout: float, res_dropout: float, embed_dropout: float) -> None:
        super().__init__()

        def block(n: int) -> TransformerEncoder:
            return TransformerEncoder(
                dim, heads, n, attn_dropout=attn_dropout, relu_dropout=relu_dropout,
                res_dropout=res_dropout, embed_dropout=embed_dropout,
                attn_mask=True, position_embedding=True,
            )

        self.lower = block(1)
        self.upper = block(layers)

    def forward(self, source: torch.Tensor, target: torch.Tensor,
                text: torch.Tensor) -> torch.Tensor:
        enhanced = self.lower(source, text, text)
        return self.upper(target, enhanced, enhanced)


@register_model("tetfn")
class TETFN(PseudoLabelMixin, MSAModel):
    @classmethod
    def build(cls, spec: DatasetSpec, **kwargs) -> MSAModel:
        kwargs.setdefault("train_size", spec.split_sizes["train"])
        return cls(text_dim=spec.text_dim, audio_dim=spec.audio_dim,
                   vision_dim=spec.vision_dim, **kwargs)

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        train_size: int,
        dim: int = 50,
        heads: int = 5,
        layers: int = 2,
        audio_hidden: int = 16,
        vision_hidden: int = 64,
        kernel_text: int = 1,
        kernel_audio: int = 1,
        attn_dropout: float = 0.1,
        attn_dropout_audio: float = 0.0,
        attn_dropout_vision: float = 0.1,
        relu_dropout: float = 0.0,
        res_dropout: float = 0.1,
        embed_dropout: float = 0.0,
        fusion_dim: int = 64,
        post_text_dim: int = 32,
        post_audio_dim: int = 32,
        post_vision_dim: int = 16,
        fusion_dropout: float = 0.0,
        post_text_dropout: float = 0.0,
        post_audio_dropout: float = 0.0,
        post_vision_dropout: float = 0.0,
        label_bound: float = 3.0,
        exclude_zero: bool = True,
        pretrained: str = "bert-base-uncased",
        finetune: bool = True,
    ) -> None:
        super().__init__()
        self.label_bound = label_bound
        self.exclude_zero = exclude_zero

        self.encoder = BertTextEncoder(pretrained, finetune)
        # kernel_audio for vision too: the reference passes conv1d_kernel_size_a
        # to both branches and never reads conv1d_kernel_size_v.
        self.audio_model = _AudioVisualSubNet(audio_dim, audio_hidden, kernel_audio, dim)
        self.vision_model = _AudioVisualSubNet(vision_dim, vision_hidden, kernel_audio, dim)
        self.proj_t = nn.Conv1d(self.encoder.hidden_size, dim, kernel_text, bias=False)

        def cross(dropout: float, width: int = dim) -> TransformerEncoder:
            # No positional encoding here — only the text-enhanced blocks get it.
            return TransformerEncoder(
                width, heads, layers, attn_dropout=dropout, relu_dropout=relu_dropout,
                res_dropout=res_dropout, embed_dropout=embed_dropout, attn_mask=True,
            )

        self.t_with_a = cross(attn_dropout_audio)
        self.t_with_v = cross(attn_dropout_vision)
        self.a_with_t = cross(attn_dropout)
        self.v_with_t = cross(attn_dropout)

        def enhanced() -> _TextEnhanced:
            return _TextEnhanced(dim, heads, layers, attn_dropout, relu_dropout,
                                 res_dropout, embed_dropout)

        self.a_with_v, self.v_with_a = enhanced(), enhanced()
        self.mem_t = cross(attn_dropout, 2 * dim)
        self.mem_a = cross(attn_dropout, 2 * dim)
        self.mem_v = cross(attn_dropout, 2 * dim)

        def branch(in_dim: int, width: int, dropout: float) -> nn.ModuleDict:
            return nn.ModuleDict({
                "drop": nn.Dropout(dropout),
                "layer_1": nn.Linear(in_dim, width),
                "layer_2": nn.Linear(width, width),
                "layer_3": nn.Linear(width, 1),
            })

        self.branch_fusion = branch(6 * dim, fusion_dim, fusion_dropout)
        self.branch_text = branch(dim, post_text_dim, post_text_dropout)
        self.branch_audio = branch(dim, post_audio_dim, post_audio_dropout)
        self.branch_vision = branch(dim, post_vision_dim, post_vision_dropout)

        self._register_pseudo_label_buffers(train_size, {
            "fusion": fusion_dim, "text": post_text_dim,
            "audio": post_audio_dim, "vision": post_vision_dim,
        })

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """MMSA's four rates and four decays, expressed relative to the CLI pair.

        Its MOSI values are BERT 3e-5 / 1e-3, audio 5e-4 / 1e-2, vision 3e-4 / 0,
        everything else 3e-4 / 1e-2 — so `--lr 3e-4 --weight-decay 0.01` puts
        each group where the reference has it. Vision decays at zero, which is
        the one asymmetry worth noticing rather than smoothing over.
        """
        encoder = list(self.encoder.parameters())
        audio = list(self.audio_model.parameters())
        vision = list(self.vision_model.parameters())
        seen = {id(p) for p in encoder + audio + vision}
        rest = [p for p in self.parameters() if id(p) not in seen]
        return [
            {"params": encoder, "lr": lr * 0.1, "weight_decay": weight_decay * 0.1},
            {"params": audio, "lr": lr * (5 / 3), "weight_decay": weight_decay},
            {"params": vision, "lr": lr, "weight_decay": 0.0},
            {"params": rest, "lr": lr, "weight_decay": weight_decay},
        ]

    @staticmethod
    def _head(branch: nn.ModuleDict, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = F.relu(branch["layer_1"](branch["drop"](x)))
        prediction = branch["layer_3"](F.relu(branch["layer_2"](representation)))
        return prediction.view(-1), representation

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text = self.encoder(batch["text_bert"])
        audio = self.audio_model(batch["audio"])
        vision = self.vision_model(batch["vision"])

        # (batch, dim, steps) -> (steps, batch, dim), the layout the encoders use.
        x_t = self.proj_t(text.transpose(1, 2)).permute(2, 0, 1)
        x_a = audio.permute(2, 0, 1)
        x_v = vision.permute(2, 0, 1)

        # The unimodal branches read a max over time, not the last step.
        pooled = {"text": x_t.max(dim=0)[0], "audio": x_a.max(dim=0)[0],
                  "vision": x_v.max(dim=0)[0]}

        h_t = self.mem_t(torch.cat([self.t_with_a(x_t, x_a, x_a),
                                    self.t_with_v(x_t, x_v, x_v)], dim=2))[-1]
        # a_with_v(source=vision, target=audio, text): audio reads text-enhanced vision.
        h_a = self.mem_a(torch.cat([self.a_with_t(x_a, x_t, x_t),
                                    self.a_with_v(x_v, x_a, x_t)], dim=2))[-1]
        h_v = self.mem_v(torch.cat([self.v_with_t(x_v, x_t, x_t),
                                    self.v_with_a(x_a, x_v, x_t)], dim=2))[-1]

        fusion, f_fusion = self._head(self.branch_fusion, torch.cat([h_t, h_a, h_v], dim=-1))
        text_out, f_text = self._head(self.branch_text, pooled["text"])
        audio_out, f_audio = self._head(self.branch_audio, pooled["audio"])
        vision_out, f_vision = self._head(self.branch_vision, pooled["vision"])
        return {
            "M": fusion, "T": text_out, "A": audio_out, "V": vision_out,
            "feature_fusion": f_fusion, "feature_text": f_text,
            "feature_audio": f_audio, "feature_vision": f_vision,
        }

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.pseudo_label_loss(outputs, batch)

    def on_train_batch_end(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], epoch: int
    ) -> None:
        self.pseudo_label_step(outputs, batch, epoch)
