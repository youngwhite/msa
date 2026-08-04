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
    * `on_train_epoch_start` / `on_train_batch_end` exist for models that carry
      state between batches. Self-MM is the only one so far: it rewrites its own
      unimodal training targets as it goes.
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

    def on_train_batch_end(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], epoch: int
    ) -> None:
        """Called after each optimiser step, for models that carry state across
        batches. Self-MM regenerates its unimodal pseudo-labels here. No-op by
        default, so the loop stays identical for every other model."""

    def on_train_epoch_start(self, epoch: int) -> None:
        """Called before each training epoch."""

    def on_run_start(self, seed: int) -> None:
        """Called once after construction, before training, with this run's seed.

        For models whose weights depend on a stage this repository runs
        separately: ConFEDE pretrains three unimodal encoders per seed and its
        text encoder is 418MB, more than ten of which will not fit on this disk,
        so it produces the current seed's stage one here and drops the previous
        seed's. No-op by default, so the loop is identical for every other
        model."""

    def auxiliary_optimizer(
        self, lr: float, weight_decay: float
    ) -> torch.optim.Optimizer | None:
        """A second objective, optimised in its own pass over the training data.

        Returning an optimiser makes the trainer run a full extra epoch over the
        training split *before* each main epoch, stepping only this optimiser on
        the loss from `auxiliary_loss`. Returning None — the default — leaves the
        loop exactly as it was for every other model.

        MMIM needs this: its mutual-information bound is fitted in a separate
        stage each epoch, and folding that into the main batch loop would change
        the algorithm rather than reorganise it. The hook is deliberately not
        MMIM-shaped, so the next model with an auxiliary objective needs no
        further change here.
        """
        return None

    def auxiliary_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """The auxiliary objective for one batch, including its own forward.

        The model runs the forward itself because an auxiliary pass usually wants
        different inputs than the main one — MMIM withholds the labels and the
        memory bank here, so it cannot reuse the main forward's output.
        """
        raise NotImplementedError
