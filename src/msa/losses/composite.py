"""Weighted combinations of contrastive terms, and how the weights move.

Three weighting schemes, matching the three cases phase 2 asks about:

* ``equal`` — fixed, equal weights. This is the control the other two are
  measured against, so it is not a placeholder: "does adapting the weights help"
  is only answerable if the non-adaptive version was run under otherwise
  identical conditions.
* ``grad_rate`` — the no-crossing case. Every ``window`` epochs, weight shifts
  towards the terms whose gradient is still moving. Defined precisely below,
  because "adjust by the gradient change rate" has several plausible readings
  and they do not agree.
* ``linear_ramp`` — the crossing case. Weight moves linearly towards a named
  term over training, for when one objective is better early and another late.

The task loss is not one of these terms. It enters as ``task + lambda * combo``
with ``lambda`` fixed, so the schemes redistribute weight *within* the
contrastive part and can never trade the task objective away. That is deliberate:
a scheme free to down-weight the task loss would improve total loss by ignoring
the task, which is the failure the whole phase is trying to avoid measuring.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .contrastive import ContrastiveLoss, Views


class WeightScheme:
    """How the weight on each contrastive term evolves.

    `observe` is called at most once per epoch with that epoch's mean term
    values and, when the scheme asked for them, per-term gradient norms.
    `weights` is then read every batch, so it must be cheap and must not depend
    on anything but what `observe` recorded.
    """

    #: Whether the trainer needs to pay for per-term gradient norms. Each one
    #: costs an extra backward pass, so schemes that do not use them say so.
    needs_grad_norms: bool = False
    name: str = "unnamed"

    def __init__(self, terms: Sequence[str]):
        self.terms = list(terms)
        self._weights = dict.fromkeys(self.terms, 1.0 / max(len(self.terms), 1))

    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def observe(
        self, epoch: int, values: dict[str, float], grad_norms: dict[str, float] | None
    ) -> None:
        pass

    def state(self) -> dict[str, object]:
        """What to record per epoch, so the weight trajectory is auditable."""
        return {"scheme": self.name, "weights": self.weights()}


class EqualWeights(WeightScheme):
    """Fixed 1/k on every term. The control."""

    name = "equal"


class GradientRateWeights(WeightScheme):
    """The no-crossing case: shift weight towards terms still producing signal.

    Concretely, every `window` epochs, for each term i:

        rate_i = |g_i(t) - g_i(t - window)| / (g_i(t - window) + eps)

    where g_i is the L2 norm of the gradient term i contributes to the shared
    representation. Weight is then set proportional to ``rate_i``, floored at
    `floor` and renormalised to sum to 1.

    **Why proportional and not inversely proportional.** A term whose gradient
    norm has stopped changing has converged as far as it is going to under the
    current weight: extra weight on it buys a smaller change in the
    representation than the same weight spent on a term still moving. So weight
    follows the terms that are still moving. The opposite reading -- upweight
    the converged terms to "finish" them -- is also defensible, and this is
    exactly the kind of choice that has to be fixed before looking at results
    rather than after, so it is written down here and not tuned.

    `floor` exists so a term can recover: a weight of exactly zero makes its
    gradient zero, which makes its rate zero, which pins the weight at zero
    permanently. That is an absorbing state, not a decision.
    """

    name = "grad_rate"
    needs_grad_norms = True

    def __init__(self, terms: Sequence[str], window: int = 10, floor: float = 0.05,
                 eps: float = 1e-8):
        super().__init__(terms)
        self.window, self.floor, self.eps = window, floor, eps
        self._history: dict[int, dict[str, float]] = {}
        self._rates: dict[str, float] = {}

    def observe(self, epoch, values, grad_norms):
        if grad_norms is None:
            raise ValueError(f"{self.name} needs per-term gradient norms")
        self._history[epoch] = dict(grad_norms)
        previous = self._history.get(epoch - self.window)
        if previous is None:
            return
        rates = {
            term: abs(grad_norms[term] - previous[term]) / (previous[term] + self.eps)
            for term in self.terms
        }
        total = sum(rates.values())
        if total <= 0:
            # Every term is perfectly flat. Nothing distinguishes them, so equal
            # weights is the honest answer rather than an arbitrary one.
            self._weights = dict.fromkeys(self.terms, 1.0 / len(self.terms))
        else:
            raw = {t: max(rates[t] / total, self.floor) for t in self.terms}
            scale = sum(raw.values())
            self._weights = {t: raw[t] / scale for t in self.terms}
        self._rates = rates

    def state(self):
        return {**super().state(), "rates": dict(self._rates)}


class LinearRampWeights(WeightScheme):
    """The crossing case: move weight linearly towards `favour` late in training.

    Starts at equal weights and interpolates towards `target` by `end_fraction`
    of the run, holding there. Linear rather than a step because the case this
    is for is a loss that is *becoming* better, not one that switches.

    `total_epochs` is the planned length, not the observed one. Early stopping
    can end a run before the ramp completes, and that is a property of the
    schedule rather than a problem with it: the recorded weight trajectory says
    where it got to.
    """

    name = "linear_ramp"

    def __init__(self, terms: Sequence[str], favour: str, total_epochs: int,
                 target: float = 0.7, end_fraction: float = 1.0):
        super().__init__(terms)
        if favour not in self.terms:
            raise ValueError(f"cannot favour {favour!r}, not among {self.terms}")
        if not 0.0 < target <= 1.0:
            raise ValueError(f"target must be in (0, 1], got {target}")
        self.favour, self.total_epochs = favour, max(total_epochs, 1)
        self.target, self.end_fraction = target, end_fraction
        self._progress = 0.0

    def observe(self, epoch, values, grad_norms):
        span = max(self.total_epochs * self.end_fraction, 1.0)
        self._progress = min(epoch / span, 1.0)
        start = 1.0 / len(self.terms)
        favoured = start + (self.target - start) * self._progress
        others = (1.0 - favoured) / max(len(self.terms) - 1, 1)
        self._weights = {
            term: favoured if term == self.favour else others for term in self.terms
        }

    def state(self):
        return {**super().state(), "progress": self._progress, "favour": self.favour}


SCHEMES = {
    "equal": EqualWeights,
    "grad_rate": GradientRateWeights,
    "linear_ramp": LinearRampWeights,
}


class ContrastiveHead(nn.Module):
    """Projections onto a shared space, plus the weighted contrastive terms.

    A projection head is not optional here: TFN's modality features are 32, 32
    and 128 dimensional, and every objective in `contrastive.py` compares views
    directly. It is also what SimCLR found matters -- contrasting on the
    projection rather than on the representation the task head reads keeps the
    contrastive objective from deleting information the task needs.

    One projection per modality, not shared, because the inputs are not
    commensurate: a shared projection would have to be a function of whichever
    modality's geometry dominated.

    Terms are divided by a running estimate of their own magnitude before being
    weighted (see `_rescale`), which is what makes a weight mean a share of
    influence and a single lambda comparable across candidates. Raw, unrescaled
    values are what `forward` returns for the record.
    """

    def __init__(
        self,
        feature_dims: dict[str, int],
        losses: dict[str, ContrastiveLoss],
        scheme: WeightScheme,
        projection_dim: int = 64,
        hidden_dim: int | None = None,
        lambda_: float = 0.1,
        normalise_terms: bool = True,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        if not losses:
            raise ValueError("a contrastive head with no objectives does nothing")
        hidden = hidden_dim or projection_dim
        self.projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, projection_dim)
            )
            for modality, dim in sorted(feature_dims.items())
        })
        self.losses = nn.ModuleDict(dict(sorted(losses.items())))
        self.scheme = scheme
        self.lambda_ = lambda_
        self.projection_dim = projection_dim
        self.normalise_terms = normalise_terms
        self.ema_decay = ema_decay
        # Buffers, so the running scales are part of state_dict: a resumed or
        # reloaded head that re-learned them from scratch would weight its terms
        # differently from the one that was saved.
        for name in self.losses:
            self.register_buffer(f"scale_{name}", torch.ones(()))
            self.register_buffer(f"scale_ready_{name}", torch.zeros((), dtype=torch.bool))

    def rescale(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Divide a term by a running estimate of its own magnitude.

        Without this, a weight is not a share of influence. The terms here differ
        by four orders of magnitude on real batches -- barlow_twins' gradient
        norm is ~195 where rnc's is ~0.024 -- so "equal weights" would hand
        barlow_twins the whole combination, and a single lambda would be
        measuring which objective's natural scale happened to suit it rather
        than which objective helps. That is the same category error as ranking
        objectives by their raw loss values, moved inside the sum.

        The estimate is an EMA of |value|, detached, so it rescales the term
        without contributing gradient of its own. First observation seeds it
        exactly, so the very first batch is already on a unit scale rather than
        divided by 1.
        """
        if not self.normalise_terms:
            return value
        scale = getattr(self, f"scale_{name}")
        ready = getattr(self, f"scale_ready_{name}")
        magnitude = value.detach().abs()
        if self.training:
            if bool(ready):
                scale.mul_(self.ema_decay).add_((1 - self.ema_decay) * magnitude)
            else:
                scale.copy_(magnitude)
                ready.fill_(True)
        return value / scale.clamp_min(1e-8)

    def project(self, features: dict[str, torch.Tensor]) -> Views:
        return {
            modality: self.projections[modality](features[modality])
            for modality in self.projections
        }

    def term_values(
        self, views: Views, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {name: loss(views, labels) for name, loss in self.losses.items()}

    def forward(
        self, features: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], Views]:
        """Returns ``lambda * sum(w_i * rescaled_i)``, the *raw* terms, and the views.

        The raw values are handed back unweighted and unrescaled on purpose: they
        are what the crossing-point analysis reads, and multiplying them by the
        weights that analysis is trying to explain would make the plot circular.
        The views come back because the gradient norms the weighting scheme reads
        are measured at the projection, and recomputing them would mean a second
        forward pass through it.
        """
        views = self.project(features)
        terms = self.term_values(views, labels)
        weights = self.scheme.weights()
        total = sum(
            weights[name] * self.rescale(name, value) for name, value in terms.items()
        )
        return self.lambda_ * total, terms, views
