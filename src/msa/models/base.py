"""The contract every model must satisfy to be trainable by `msa.trainer`."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DatasetSpec


class MSAModel(nn.Module):
    """A multimodal sentiment model.

    The trainer knows nothing about a model beyond this interface, which is what
    keeps one training loop honest across every architecture:

    * `forward(batch)` takes the whole batch dict (already on the right device)
      and returns a dict of predictions. The key ``"M"`` — the multimodal
      sentiment score, shape (batch,) — is mandatory. Multi-task models may add
      their own keys (e.g. ``"T"``/``"A"``/``"V"`` unimodal heads) and use them
      in `compute_loss`.
    * `compute_loss(outputs, batch)` defaults to L1 on ``"M"``; override it for
      models with auxiliary objectives rather than special-casing the trainer.
    * `param_groups` lets a model ask for per-module learning rates (a fine-tuned
      text encoder usually wants a smaller one) without the trainer knowing why.
    """

    #: Filled in by @register_model.
    name: str = "unnamed"

    @classmethod
    def build(cls, spec: DatasetSpec, **kwargs) -> MSAModel:
        """Construct from a dataset spec. Override when the signature differs."""
        return cls(
            text_dim=spec.text_dim,
            audio_dim=spec.audio_dim,
            vision_dim=spec.vision_dim,
            **kwargs,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return F.l1_loss(outputs["M"], batch["label"])

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        return [{"params": list(self.parameters()), "lr": lr, "weight_decay": weight_decay}]
