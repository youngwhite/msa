"""Building blocks shared by the ported models.

These come from the TFN reference implementation and are reused verbatim by LMF
and others in MMSA, so they live here rather than being copied per model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import last_valid_state


class SubNet(nn.Module):
    """BatchNorm -> dropout -> 3 x (linear + ReLU), for a pooled audio/vision vector.

    Note BatchNorm1d needs a batch of at least 2 in training mode; a run whose
    last batch has a single sample will raise.
    """

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


class TextSubNet(nn.Module):
    """LSTM over the token embeddings, read at the last real step, then projected."""

    def __init__(self, in_size: int, hidden: int, out_size: int, dropout: float) -> None:
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden, out_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        outputs, _ = self.rnn(x)
        return self.linear_1(self.dropout(last_valid_state(outputs, lengths)))
