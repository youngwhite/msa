"""Self-supervised unimodal pseudo-labels, shared by Self-MM and TETFN.

Both models generate their own per-modality training targets during training:
each sample's unimodal label drifts from the true multimodal one according to how
close that modality's representation sits to the positive versus the negative
class centre. MMSA's two trainers implement this identically — `update_labels`,
`update_centers` and `weighted_loss` are **byte-for-byte the same** in
`trains/multiTask/SELF_MM.py` and `trains/multiTask/TETFN.py` (verified by
hashing the three function bodies).

So it lives in one place here. A second copy is exactly the failure this project
keeps recording: an implementation written twice drifts, and the drift is silent.

A model mixes this in and supplies:

* `MODES` — ("fusion", "text", "audio", "vision"), in that order
* `_register_pseudo_label_buffers(train_size, widths)` from its `__init__`
* representations under `feature_fusion` / `feature_text` / ... in its forward
  output, and predictions under `M` / `T` / `A` / `V`

Everything else — the loss weighting, the centre updates, the momentum on the
labels — is inherited and identical for both models by construction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: Order matters: `fusion` is the anchor the other three are measured against.
MODES = ("fusion", "text", "audio", "vision")

#: Keys the forward output uses for each mode's prediction.
PREDICTION_KEYS = {"text": "T", "audio": "A", "vision": "V"}


class PseudoLabelMixin(nn.Module):
    """The label-generation half of Self-MM, without any architecture.

    `label_bound` is MMSA's `H`: pseudo-labels are clamped to +-H. `exclude_zero`
    is its `excludeZero`, which decides whether an exactly-zero label counts
    towards the positive centre.
    """

    label_bound: float = 3.0
    exclude_zero: bool = True

    def _register_pseudo_label_buffers(
        self, train_size: int, widths: dict[str, int]
    ) -> None:
        """Per-sample labels, the representation each came from, and the centres.

        Buffers rather than plain tensors: they move with `.to(device)` and land
        in the checkpoint, so a resumed run keeps the labels it had drifted to.
        """
        for mode in MODES:
            self.register_buffer(f"label_{mode}", torch.zeros(train_size))
            self.register_buffer(f"feature_{mode}", torch.zeros(train_size, widths[mode]))
            self.register_buffer(f"center_pos_{mode}", torch.zeros(widths[mode]))
            self.register_buffer(f"center_neg_{mode}", torch.zeros(widths[mode]))
        # Per sample, not global: a sample's labels start at its true label the
        # first time it is seen. One global flag would leave every sample after
        # the first batch initialised to zero.
        self.register_buffer("label_seen", torch.zeros(train_size, dtype=torch.bool))

    def pseudo_label_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """L1 on the fused head plus three weighted unimodal terms.

        A unimodal branch is trusted in proportion to how far its pseudo-label
        has drifted from the fused one — the samples where a modality says
        something different from the fusion are the ones worth learning from.
        """
        index, label = batch["index"], batch["label"]
        fresh = ~self.label_seen[index]
        if fresh.any():
            for mode in MODES:
                getattr(self, f"label_{mode}")[index[fresh]] = label[fresh]
        loss = torch.abs(outputs["M"] - self.label_fusion[index]).mean()
        for mode, key in PREDICTION_KEYS.items():
            target = getattr(self, f"label_{mode}")[index]
            weight = torch.tanh(torch.abs(target - self.label_fusion[index]))
            loss = loss + (weight * torch.abs(outputs[key] - target)).mean()
        return loss

    @torch.no_grad()
    def pseudo_label_step(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], epoch: int
    ) -> None:
        """Update labels, then features, then centres — MMSA's order.

        Labels are updated *before* this batch's features are written, so they
        are computed against the centres as they stood beforehand. Skipped
        entirely in epoch 1, when no centres exist yet.
        """
        index = batch["index"]
        if epoch > 1 and bool(self.label_seen.all()):
            self._update_pseudo_labels(outputs, index, epoch)
        for mode in MODES:
            getattr(self, f"feature_{mode}")[index] = outputs[f"feature_{mode}"].detach()
        self._update_centers()
        self.label_seen[index] = True

    def _update_centers(self) -> None:
        for mode in MODES:
            labels = getattr(self, f"label_{mode}")
            features = getattr(self, f"feature_{mode}")
            positive = labels > 0 if self.exclude_zero else labels >= 0
            negative = labels < 0
            if positive.any():
                getattr(self, f"center_pos_{mode}").copy_(features[positive].mean(0))
            if negative.any():
                getattr(self, f"center_neg_{mode}").copy_(features[negative].mean(0))

    def _relative_distance(self, features: torch.Tensor, mode: str) -> torch.Tensor:
        """(distance to the negative centre - to the positive one), scaled.

        Large and positive when the sample looks clearly positive.
        """
        to_pos = torch.norm(features - getattr(self, f"center_pos_{mode}"), dim=-1)
        to_neg = torch.norm(features - getattr(self, f"center_neg_{mode}"), dim=-1)
        return (to_neg - to_pos) / (to_pos + 1e-8)

    def _update_pseudo_labels(
        self, outputs: dict[str, torch.Tensor], index: torch.Tensor, epoch: int
    ) -> None:
        fused = self.label_fusion[index]
        delta_fusion = self._relative_distance(outputs["feature_fusion"].detach(), "fusion")
        for mode in ("text", "audio", "vision"):
            delta = self._relative_distance(outputs[f"feature_{mode}"].detach(), mode)
            alpha = delta / (delta_fusion + 1e-8)
            proposed = 0.5 * alpha * fused + 0.5 * (fused + delta - delta_fusion)
            proposed = torch.clamp(proposed, -self.label_bound, self.label_bound)
            # Momentum: later epochs move the label less.
            buffer = getattr(self, f"label_{mode}")
            buffer[index] = ((epoch - 1) / (epoch + 1) * buffer[index]
                             + 2 / (epoch + 1) * proposed)
