"""The two pre-TFN baselines: early fusion and late fusion.

They bracket the storyline's starting point. EF-LSTM concatenates the three
modalities at every timestep and runs one LSTM over the result — fusion before
any modality has been understood on its own terms. LF-DNN does the opposite: each
modality is encoded separately and the vectors are concatenated at the very end,
so nothing models how the modalities interact.

TFN exists because neither is satisfying: the first cannot separate modalities,
the second cannot relate them.

Ported from MMSA (`models/singleTask/EF_LSTM.py`, `LF_DNN.py`, MIT, THUIAR).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from .base import MSAModel
from .blocks import SubNet, TextSubNet
from .functional import masked_mean


@register_model("ef_lstm")
class EarlyFusionLSTM(MSAModel):
    """Concatenate the modalities per timestep, then one LSTM over the whole thing.

    Uses aligned data — the three streams have to share a clock for a per-step
    concatenation to mean anything.

    The BatchNorm is over the *time* axis, not the feature axis: MMSA constructs
    `BatchNorm1d(seq_len)` and applies it to a (batch, seq, feature) tensor, so
    each timestep is a channel. Unusual, and kept because it is what produced the
    reference number.
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        seq_len: int = 50,
        hidden_size: int = 128,
        num_layers: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(seq_len)
        self.lstm = nn.LSTM(
            text_dim + audio_dim + vision_dim, hidden_size, num_layers=num_layers,
            dropout=dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = torch.cat([batch["text"], batch["audio"], batch["vision"]], dim=-1)
        _, (h, _) = self.lstm(self.norm(x))
        y = self.dropout(h[-1])
        y = self.dropout(F.relu(self.linear(y)))
        return {"M": self.out(y).view(-1)}


@register_model("lf_dnn")
class LateFusionDNN(MSAModel):
    """Encode each modality separately, concatenate at the end, then an MLP."""

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        text_hidden: int = 128,
        audio_hidden: int = 16,
        vision_hidden: int = 128,
        text_out: int = 256,
        post_fusion_dim: int = 128,
        text_dropout: float = 0.2,
        audio_dropout: float = 0.2,
        vision_dropout: float = 0.2,
        post_fusion_dropout: float = 0.2,
        use_lengths: bool = True,
        mask_pooling: bool = True,
    ) -> None:
        super().__init__()
        self.use_lengths = use_lengths
        self.mask_pooling = mask_pooling
        self.audio_subnet = SubNet(audio_dim, audio_hidden, audio_dropout)
        self.vision_subnet = SubNet(vision_dim, vision_hidden, vision_dropout)
        self.text_subnet = TextSubNet(text_dim, text_hidden, text_out, text_dropout)
        self.post_fusion_dropout = nn.Dropout(p=post_fusion_dropout)
        self.layer_1 = nn.Linear(text_out + vision_hidden + audio_hidden, post_fusion_dim)
        self.layer_2 = nn.Linear(post_fusion_dim, post_fusion_dim)
        self.layer_3 = nn.Linear(post_fusion_dim, 1)

    def _pool(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return masked_mean(x, lengths if self.mask_pooling else None)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        audio_h = self.audio_subnet(self._pool(batch["audio"], batch["audio_length"]))
        vision_h = self.vision_subnet(self._pool(batch["vision"], batch["vision_length"]))
        text_h = self.text_subnet(
            batch["text"], batch["text_length"] if self.use_lengths else None
        )
        fused = torch.cat([audio_h, vision_h, text_h], dim=-1)
        y = self.post_fusion_dropout(fused)
        y = F.relu(self.layer_1(y))
        y = F.relu(self.layer_2(y))
        return {"M": self.layer_3(y).view(-1)}
