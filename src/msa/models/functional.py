"""Sequence pooling shared by models. Length handling lives here, once."""

from __future__ import annotations

import torch


def last_valid_state(outputs: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """(batch, time, hidden) -> (batch, hidden) at each sample's last real step.

    With `lengths=None` this is the plain final step, i.e. whatever the recurrence
    drifted to after the padding — the behaviour of the original implementations.
    """
    if lengths is None:
        return outputs[:, -1]
    idx = lengths.clamp(min=1, max=outputs.shape[1]) - 1
    idx = idx.view(-1, 1, 1).expand(-1, 1, outputs.shape[-1])
    return outputs.gather(1, idx).squeeze(1)


def masked_mean(x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """Average over real frames only: (batch, time, dim) -> (batch, dim).

    `lengths=None` averages over the padded width instead, which is what MMSA's
    `__normalize()` does. That is not equivalent: dividing by the padded width
    scales every sample by `valid_len / padded_width`, so utterance length leaks
    into the feature magnitude (on unaligned MOSI only ~39 of 375 audio frames
    are real). Kept selectable so the difference can be measured, not assumed.
    """
    if lengths is None:
        return x.mean(dim=1)
    time = x.shape[1]
    steps = torch.arange(time, device=x.device).unsqueeze(0)
    mask = (steps < lengths.clamp(min=1, max=time).unsqueeze(1)).unsqueeze(-1).to(x.dtype)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
