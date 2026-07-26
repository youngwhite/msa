"""Tensor Fusion Network (Zadeh et al., EMNLP 2017).

Ported from MMSA (`models/singleTask/TFN.py`, MIT, THUIAR) — architecture and
default hyper-parameters follow it so the numbers are comparable to its reported
MOSI result (MAE 0.947). Two documented deviations, both selectable:

* `use_lengths=True` reads the text LSTM at the last real token instead of after
  the [PAD] tail (repo-wide convention, see docs/decisions.md).
* `mask_pooling=True` averages audio/vision over real frames only. MMSA divides
  by the padded width, which multiplies each sample's features by
  `valid_len / padded_width` — on unaligned MOSI that is ~39/375.

Set both to False for a bit-faithful port.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from .base import MSAModel
from .functional import last_valid_state, masked_mean


class _SubNet(nn.Module):
    """BatchNorm -> dropout -> 3 x (linear + ReLU), for the pooled audio/vision vector."""

    def __init__(self, in_size: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.norm(x))
        y = F.relu(self.linear_1(y))
        y = F.relu(self.linear_2(y))
        return F.relu(self.linear_3(y))


class _TextSubNet(nn.Module):
    def __init__(self, in_size: int, hidden: int, out_size: int, dropout: float) -> None:
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden, out_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        outputs, _ = self.rnn(x)
        return self.linear_1(self.dropout(last_valid_state(outputs, lengths)))


@register_model("tfn")
class TensorFusionNetwork(MSAModel):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        text_hidden: int = 128,
        audio_hidden: int = 32,
        vision_hidden: int = 128,
        text_out: int = 32,
        post_fusion_dim: int = 64,
        text_dropout: float = 0.3,
        audio_dropout: float = 0.3,
        vision_dropout: float = 0.3,
        post_fusion_dropout: float = 0.5,
        use_lengths: bool = True,
        mask_pooling: bool = True,
        output_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()
        self.use_lengths = use_lengths
        self.mask_pooling = mask_pooling
        self.audio_hidden = audio_hidden
        self.vision_hidden = vision_hidden
        self.text_out = text_out

        self.audio_subnet = _SubNet(audio_dim, audio_hidden, audio_dropout)
        self.vision_subnet = _SubNet(vision_dim, vision_hidden, vision_dropout)
        self.text_subnet = _TextSubNet(text_dim, text_hidden, text_out, text_dropout)

        fusion_size = (text_out + 1) * (vision_hidden + 1) * (audio_hidden + 1)
        self.post_fusion_dropout = nn.Dropout(p=post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(fusion_size, post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(post_fusion_dim, post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(post_fusion_dim, 1)

        # Buffers, not Parameters: MMSA stores these as requires_grad=False
        # Parameters and then has to build its optimiser from
        # `list(model.parameters())[2:]` to skip them — a positional hack that
        # silently drops real parameters if anyone reorders the constructor.
        low, high = output_range
        self.register_buffer("output_shift", torch.tensor(low))
        self.register_buffer("output_scale", torch.tensor(high - low))

    def _pool(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return masked_mean(x, lengths if self.mask_pooling else None)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        audio_h = self.audio_subnet(self._pool(batch["audio"], batch["audio_length"]))
        vision_h = self.vision_subnet(self._pool(batch["vision"], batch["vision_length"]))
        text_h = self.text_subnet(
            batch["text"], batch["text_length"] if self.use_lengths else None
        )

        # Tensor fusion: append a 1 to each modality so the outer product also
        # contains the unimodal and bimodal terms, then flatten.
        ones = torch.ones(audio_h.shape[0], 1, dtype=audio_h.dtype, device=audio_h.device)
        a = torch.cat((ones, audio_h), dim=1)
        v = torch.cat((ones, vision_h), dim=1)
        t = torch.cat((ones, text_h), dim=1)

        fusion = torch.bmm(a.unsqueeze(2), v.unsqueeze(1))
        fusion = fusion.view(-1, (self.audio_hidden + 1) * (self.vision_hidden + 1), 1)
        fusion = torch.bmm(fusion, t.unsqueeze(1)).view(audio_h.shape[0], -1)

        y = self.post_fusion_dropout(fusion)
        y = F.relu(self.post_fusion_layer_1(y))
        y = F.relu(self.post_fusion_layer_2(y))
        y = self.post_fusion_layer_3(y)
        # TFN constrains the output to the label range with a sigmoid.
        y = torch.sigmoid(y) * self.output_scale + self.output_shift
        return {
            "M": y.squeeze(-1),
            "feature_t": text_h,
            "feature_a": audio_h,
            "feature_v": vision_h,
        }
