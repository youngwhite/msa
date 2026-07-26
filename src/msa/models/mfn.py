"""Memory Fusion Network (Zadeh et al., AAAI 2018).

TFN and LMF collapse each modality to one vector before fusing, so any
interaction that plays out *over time* — a smile arriving a beat after the word
it undercuts — is gone before fusion starts. MFN keeps three LSTMCells running in
lockstep and, at every step, attends over the modalities' memory cells before and
after the update, writing the result into a shared memory through two gates.

Ported from MMSA (`models/singleTask/MFN.py`, MIT, THUIAR).

**A quirk of MMSA's configuration worth understanding before reading the
numbers.** MFN's entry sets both `need_normalized` and `need_model_aligned`. The
first mean-pools audio and vision over time down to a single step; the second
then re-expands that step back to the text length by repetition. The audio and
vision streams MFN receives are therefore *constant over time* — only the text
stream actually varies — which removes most of what MFN was built to model.

We replicate that for the acceptance run (`collapse_av_to_mean=True`, the
default) because reproducing the reference means reproducing what it ran. Set it
to False to feed the real per-step sequences, which is the model the paper
describes; that comparison is reported in docs/storyline.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model
from .base import MSAModel
from .functional import masked_mean


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    """The fc1 -> relu -> dropout -> fc2 block MFN repeats four times."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
    )


@register_model("mfn")
class MemoryFusionNetwork(MSAModel):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        text_hidden: int = 256,
        audio_hidden: int = 32,
        vision_hidden: int = 256,
        memsize: int = 400,
        window: int = 2,
        attention_hidden: int = 32,
        attention_dropout: float = 0.2,
        memory_hidden: int = 64,
        memory_dropout: float = 0.0,
        gamma1_hidden: int = 256,
        gamma1_dropout: float = 0.7,
        gamma2_hidden: int = 32,
        gamma2_dropout: float = 0.5,
        out_hidden: int = 128,
        out_dropout: float = 0.0,
        collapse_av_to_mean: bool = True,
    ) -> None:
        super().__init__()
        self.collapse_av_to_mean = collapse_av_to_mean
        total_hidden = text_hidden + audio_hidden + vision_hidden
        attention_in = total_hidden * window   # the cells before and after the step

        self.lstm_t = nn.LSTMCell(text_dim, text_hidden)
        self.lstm_a = nn.LSTMCell(audio_dim, audio_hidden)
        self.lstm_v = nn.LSTMCell(vision_dim, vision_hidden)

        # Which parts of the joint memory-cell window matter at this step...
        self.attention = _mlp(attention_in, attention_hidden, attention_in, attention_dropout)
        # ...and what to propose writing into the shared memory.
        self.proposal = _mlp(attention_in, memory_hidden, memsize, memory_dropout)
        # Two gates: how much old memory to keep, how much of the proposal to add.
        self.gamma1 = _mlp(attention_in + memsize, gamma1_hidden, memsize, gamma1_dropout)
        self.gamma2 = _mlp(attention_in + memsize, gamma2_hidden, memsize, gamma2_dropout)

        self.head = _mlp(total_hidden + memsize, out_hidden, 1, out_dropout)
        self.memsize = memsize
        self.hidden_sizes = (text_hidden, audio_hidden, vision_hidden)

    def _prepare(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        text = batch["text"]
        audio, vision = batch["audio"], batch["vision"]
        if self.collapse_av_to_mean:
            # MMSA's need_normalized + need_model_aligned in one step: average
            # over the padded width, then repeat across the text timeline.
            steps = text.shape[1]
            audio = masked_mean(audio, None).unsqueeze(1).expand(-1, steps, -1)
            vision = masked_mean(vision, None).unsqueeze(1).expand(-1, steps, -1)
        return text, audio, vision

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text, audio, vision = self._prepare(batch)
        batch_size, steps = text.shape[0], text.shape[1]
        device, dtype = text.device, text.dtype
        dh_t, dh_a, dh_v = self.hidden_sizes

        def zeros(width: int) -> torch.Tensor:
            return torch.zeros(batch_size, width, device=device, dtype=dtype)

        h_t, c_t = zeros(dh_t), zeros(dh_t)
        h_a, c_a = zeros(dh_a), zeros(dh_a)
        h_v, c_v = zeros(dh_v), zeros(dh_v)
        memory = zeros(self.memsize)

        for step in range(steps):
            previous = torch.cat([c_t, c_a, c_v], dim=1)
            h_t, c_t = self.lstm_t(text[:, step], (h_t, c_t))
            h_a, c_a = self.lstm_a(audio[:, step], (h_a, c_a))
            h_v, c_v = self.lstm_v(vision[:, step], (h_v, c_v))
            window = torch.cat([previous, torch.cat([c_t, c_a, c_v], dim=1)], dim=1)

            attended = F.softmax(self.attention(window), dim=1) * window
            proposal = torch.tanh(self.proposal(attended))
            gated = torch.cat([attended, memory], dim=1)
            memory = (torch.sigmoid(self.gamma1(gated)) * memory
                      + torch.sigmoid(self.gamma2(gated)) * proposal)

        return {"M": self.head(torch.cat([h_t, h_a, h_v, memory], dim=1)).view(-1)}
