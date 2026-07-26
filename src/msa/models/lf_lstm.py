"""Late-fusion LSTM baseline (LF-LSTM).

Deliberately plain: one LSTM per modality, concatenate the last *valid* hidden
state of each, regress the sentiment score with an MLP. It exists to validate the
data/train/eval pipeline and to give later models a reference number, not to be
competitive.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import register_model
from .base import MSAModel
from .functional import last_valid_state


class _ModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # Read the state at the last real step. Padding is not harmless: BERT
        # gives [PAD] tokens non-zero embeddings, and even all-zero audio/vision
        # frames keep driving the recurrence, so the final step is a state that
        # has run ~35 of 50 steps past the end of the utterance.
        out, _ = self.lstm(self.norm(x))
        return self.dropout(last_valid_state(out, lengths))


@register_model("lf_lstm")
class LateFusionLSTM(MSAModel):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        hidden_text: int = 128,
        hidden_audio: int = 32,
        hidden_vision: int = 32,
        fusion_dim: int = 128,
        dropout: float = 0.2,
        modalities: str = "tav",
        use_lengths: bool = True,
    ) -> None:
        super().__init__()
        modalities = "".join(m for m in "tav" if m in modalities.lower())
        if not modalities:
            raise ValueError("at least one of t/a/v must be enabled")
        self.modalities = modalities
        self.use_lengths = use_lengths
        self.text_enc = (
            _ModalityEncoder(text_dim, hidden_text, dropout) if "t" in modalities else None
        )
        self.audio_enc = (
            _ModalityEncoder(audio_dim, hidden_audio, dropout) if "a" in modalities else None
        )
        self.vision_enc = (
            _ModalityEncoder(vision_dim, hidden_vision, dropout) if "v" in modalities else None
        )
        fused_in = sum(
            h for m, h in (("t", hidden_text), ("a", hidden_audio), ("v", hidden_vision))
            if m in modalities
        )
        self.head = nn.Sequential(
            nn.Linear(fused_in, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        parts = []
        for key, encoder in (
            ("text", self.text_enc), ("audio", self.audio_enc), ("vision", self.vision_enc)
        ):
            if encoder is None:
                continue
            lengths = batch[f"{key}_length"] if self.use_lengths else None
            parts.append(encoder(batch[key], lengths))
        return {"M": self.head(torch.cat(parts, dim=-1)).squeeze(-1)}
