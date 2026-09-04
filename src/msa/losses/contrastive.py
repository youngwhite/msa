"""Contrastive objectives, as auxiliary terms on modality representations.

Read this before comparing any two numbers produced here.

**These are not alternatives to the task loss.** InfoNCE alone predicts no
sentiment score; it shapes the representation the regression head reads. Every
one of these enters training as ``task_loss + lambda * contrastive_loss``, and a
candidate is judged by what that does to *validation MAE/Corr*, never by its own
value. Loss values across different objectives are not comparable — different
scales, different units, different minima, and most of them move with batch size
and temperature. Screening on "whose loss number is lower" would rank them by
their normalisation constants.

**Positive pairs on a continuous target.** MOSI's label is a real number in
[-3, 3], so the usual "same class = positive" construction has no direct
definition. Two constructions are available and this module keeps them separate:

* *cross-modal* — the text, audio and vision views of the same clip are
  positives of each other, everything else in the batch is negative. Needs no
  labels, applies unchanged to regression, and is what MMIM already does in this
  repository (its CPC term). Most losses here take this route.
* *label-aware* — positives come from label proximity. For discrete labels that
  is SupCon; for a continuous target it needs an objective defined on ordering
  rather than on classes, which is what ``rnc`` is. Anything label-aware that
  requires *classes* (``supcon``, ``arcface``) has to bin the target first, and
  that binning is an approximation the screening must state rather than hide.

Each loss takes ``views``: a dict of modality name -> (batch, dim) embeddings,
already pooled to one vector per clip. Whether they are L2-normalised is the
loss's own business — several of them require it and several are defined on
unnormalised vectors.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_loss

Views = dict[str, torch.Tensor]


class ContrastiveLoss(nn.Module):
    """An auxiliary objective on modality representations.

    Subclasses implement `pair_loss` (called for each ordered modality pair) or
    override `forward` outright when the objective is not pairwise.
    """

    #: Filled in by @register_loss.
    name: str = "unnamed"
    #: True if the objective needs a *discrete* label, which a [-3, 3] target
    #: only has after binning. Reported by the screening so the approximation is
    #: visible in the results table rather than buried here.
    needs_class_labels: bool = False
    #: True if the objective is defined on the continuous target directly.
    uses_continuous_labels: bool = False
    #: True if the objective owns parameters, which then need to reach the
    #: optimiser — see `MSAModel.param_groups`.
    parametric: bool = False

    def pair_loss(
        self, x: torch.Tensor, y: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, views: Views, labels: torch.Tensor) -> torch.Tensor:
        """Mean of `pair_loss` over every unordered pair of modalities."""
        keys = sorted(views)
        if len(keys) < 2:
            raise ValueError(f"{self.name} needs at least two views, got {keys}")
        terms = [
            self.pair_loss(views[a], views[b], labels)
            for a, b in itertools.combinations(keys, 2)
        ]
        return torch.stack(terms).mean()


def _normalise(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-8)


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Every element of a square matrix except the diagonal, flattened."""
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# --------------------------------------------------------------------------
# Cross-modal: positives are the other modalities of the same clip.
# --------------------------------------------------------------------------


@register_loss("infonce")
class InfoNCE(ContrastiveLoss):
    """Symmetric cross-modal InfoNCE, as CLIP uses it (Radford et al. 2021).

    For a pair of modalities the (batch, batch) similarity matrix has the true
    correspondences on its diagonal, so this is cross-entropy against
    ``arange(batch)`` in both directions. Symmetric because the two directions
    are different problems: retrieving audio from text is not retrieving text
    from audio, and averaging them is what makes the objective modality-neutral.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def pair_loss(self, x, y, labels):
        logits = _normalise(x) @ _normalise(y).t() / self.temperature
        target = torch.arange(x.shape[0], device=x.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


@register_loss("nt_xent")
class NTXent(ContrastiveLoss):
    """SimCLR's NT-Xent (Chen et al. 2020), with modalities as the two views.

    The difference from `infonce` is not cosmetic: NT-Xent pools both views into
    one set of 2N examples, so each anchor's negatives include the *other
    samples of its own modality*, not only the other modality's. That makes it
    push apart clips within a modality as well as across, which is a stronger
    (and sometimes harmful) constraint than CLIP's.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def pair_loss(self, x, y, labels):
        batch = x.shape[0]
        z = torch.cat([_normalise(x), _normalise(y)], dim=0)
        sim = z @ z.t() / self.temperature
        # Mask self-similarity, which would otherwise dominate every row.
        sim.fill_diagonal_(float("-inf"))
        # Positive of i is i+batch, and of i+batch is i.
        target = torch.cat([
            torch.arange(batch, 2 * batch, device=x.device),
            torch.arange(0, batch, device=x.device),
        ])
        return F.cross_entropy(sim, target)


@register_loss("dcl")
class DecoupledContrastive(ContrastiveLoss):
    """Decoupled Contrastive Learning (Yeh et al. 2022).

    InfoNCE's gradient contains a positive-negative coupling term that shrinks
    the learning signal exactly when the positive pair is already close, which
    is why it needs large batches. DCL removes the positive from the
    denominator, leaving ``-sim_pos + logsumexp(negatives)``. Same information,
    no coupling, and it is the one variant here with a documented reason to work
    at the batch sizes this project actually uses (16-64).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def _directional(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        sim = a @ b.t() / self.temperature
        positive = sim.diagonal()
        negatives = sim.clone()
        negatives.fill_diagonal_(float("-inf"))
        return (-positive + torch.logsumexp(negatives, dim=1)).mean()

    def pair_loss(self, x, y, labels):
        a, b = _normalise(x), _normalise(y)
        return 0.5 * (self._directional(a, b) + self._directional(b, a))


@register_loss("hcl")
class HardNegativeContrastive(ContrastiveLoss):
    """Hard-negative sampling for contrastive learning (Robinson et al. 2021).

    Reweights negatives towards the hard ones by ``exp(beta * sim)`` and applies
    the debiasing correction of Chuang et al. (2020) with class prior ``tau_plus``
    — without it, upweighting hard negatives mostly upweights false negatives.
    The correction is clamped at ``exp(-1/temperature)``, the smallest value the
    negative term can legitimately take, which is the reference implementation's
    guard against a negative estimate.
    """

    def __init__(self, temperature: float = 0.5, beta: float = 1.0, tau_plus: float = 0.1):
        super().__init__()
        self.temperature, self.beta, self.tau_plus = temperature, beta, tau_plus

    def pair_loss(self, x, y, labels):
        batch = x.shape[0]
        a, b = _normalise(x), _normalise(y)
        positive = torch.exp((a * b).sum(dim=-1) / self.temperature)

        sim = torch.exp(a @ b.t() / self.temperature)
        mask = ~torch.eye(batch, dtype=torch.bool, device=x.device)
        negatives = sim[mask].view(batch, batch - 1)

        weight = torch.exp(self.beta * torch.log(negatives.clamp_min(1e-12)) * self.temperature)
        weight = weight * (batch - 1) / weight.sum(dim=1, keepdim=True)
        reweighted = (weight * negatives).sum(dim=1)

        n = batch - 1
        debiased = (reweighted - n * self.tau_plus * positive) / (1 - self.tau_plus)
        debiased = debiased.clamp_min(n * torch.exp(torch.tensor(-1.0 / self.temperature)))
        return -torch.log(positive / (positive + debiased)).mean()


@register_loss("cpc")
class ContrastivePredictiveCoding(ContrastiveLoss):
    """CPC (Oord et al. 2018): InfoNCE through a learned bilinear score.

    The scoring function is ``x W y^T`` rather than a cosine, so the two views
    need not live in a shared geometry — the projection is learned. MMIM already
    uses this shape in this repository (its `cpc_zt`/`cpc_zv`/`cpc_za` terms), so
    it doubles as a sanity check: a candidate that cannot match MMIM's own term
    on MMIM's own backbone would say something is wrong with this harness.

    Parametric, so it must reach the optimiser. See `CompositeLoss`.
    """

    parametric = True

    def __init__(self, dim: int, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.score = nn.Linear(dim, dim, bias=False)

    def pair_loss(self, x, y, labels):
        logits = _normalise(self.score(x)) @ _normalise(y).t() / self.temperature
        target = torch.arange(x.shape[0], device=x.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


@register_loss("triplet")
class TripletMargin(ContrastiveLoss):
    """Triplet loss with batch-hard negative mining (Schroff et al. 2015).

    Anchor is a clip in one modality, positive is the same clip in the other,
    negative is the *hardest* other clip in the batch — the closest one that is
    wrong. Batch-hard rather than random because with a batch of 16 a random
    negative is almost always already far enough for the margin to be inactive,
    which would make this loss silently zero.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def pair_loss(self, x, y, labels):
        a, b = _normalise(x), _normalise(y)
        distance = torch.cdist(a, b)
        positive = distance.diagonal()
        hardest = distance + torch.eye(x.shape[0], device=x.device) * 1e9
        negative = hardest.min(dim=1).values
        return F.relu(positive - negative + self.margin).mean()


@register_loss("max_margin")
class MaxMargin(ContrastiveLoss):
    """The original contrastive loss (Hadsell et al. 2006).

    Pull positives to zero distance, push negatives beyond a margin, both
    squared. Predates InfoNCE and has no temperature or partition function,
    which is exactly why it is worth having in the screen: if the InfoNCE family
    wins here, it should be possible to say it beat something that is not a
    softmax over the batch.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def pair_loss(self, x, y, labels):
        a, b = _normalise(x), _normalise(y)
        distance = torch.cdist(a, b)
        eye = torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
        positive = distance[eye].pow(2).mean()
        negative = F.relu(self.margin - distance[~eye]).pow(2).mean()
        return positive + negative


@register_loss("align_uniform")
class AlignmentUniformity(ContrastiveLoss):
    """Alignment + uniformity (Wang & Isola 2020).

    Contrastive learning's two asymptotic properties, written down directly
    instead of being induced by a softmax: positives close (alignment), the
    representation spread over the sphere (uniformity). Useful in this screen as
    a diagnostic — if a candidate helps, these two terms say which half of the
    effect it came from.
    """

    def __init__(self, alpha: float = 2.0, t: float = 2.0, uniformity_weight: float = 1.0):
        super().__init__()
        self.alpha, self.t, self.uniformity_weight = alpha, t, uniformity_weight

    def _uniformity(self, z: torch.Tensor) -> torch.Tensor:
        return torch.pdist(z, p=2).pow(2).mul(-self.t).exp().mean().clamp_min(1e-12).log()

    def pair_loss(self, x, y, labels):
        a, b = _normalise(x), _normalise(y)
        alignment = (a - b).norm(dim=1).pow(self.alpha).mean()
        uniformity = 0.5 * (self._uniformity(a) + self._uniformity(b))
        return alignment + self.uniformity_weight * uniformity


# --------------------------------------------------------------------------
# Negative-free: no negatives at all, collapse prevented structurally.
# --------------------------------------------------------------------------


@register_loss("barlow_twins")
class BarlowTwins(ContrastiveLoss):
    """Barlow Twins (Zbontar et al. 2021): redundancy reduction, no negatives.

    Drives the cross-correlation matrix between the two views towards the
    identity — diagonal to 1 (invariance), off-diagonal to 0 (decorrelation).
    Batch-size-insensitive by construction, which matters here: this project
    trains at batch 16-64, where every softmax-over-the-batch objective is at
    its weakest.
    """

    def __init__(self, lambd: float = 5e-3):
        super().__init__()
        self.lambd = lambd

    def pair_loss(self, x, y, labels):
        batch = x.shape[0]
        # Standardise per feature; the objective is defined on correlations.
        a = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
        b = (y - y.mean(0)) / y.std(0).clamp_min(1e-6)
        c = (a.t() @ b) / batch
        on_diagonal = (c.diagonal() - 1).pow(2).sum()
        return on_diagonal + self.lambd * _off_diagonal(c).pow(2).sum()


@register_loss("vicreg")
class VICReg(ContrastiveLoss):
    """VICReg (Bardes et al. 2022): variance, invariance, covariance.

    Barlow Twins' idea split into three explicit terms, which is why it is here
    alongside it rather than instead of it: the variance term (a hinge on each
    feature's standard deviation) is an anti-collapse mechanism no other loss in
    this set has, and if the negative-free family works on this data, this loss
    is the one that says whether it was the anti-collapse term doing it.
    """

    def __init__(self, sim_weight: float = 25.0, var_weight: float = 25.0,
                 cov_weight: float = 1.0):
        super().__init__()
        self.sim_weight, self.var_weight, self.cov_weight = sim_weight, var_weight, cov_weight

    def _variance(self, z: torch.Tensor) -> torch.Tensor:
        return F.relu(1.0 - torch.sqrt(z.var(dim=0) + 1e-4)).mean()

    def _covariance(self, z: torch.Tensor) -> torch.Tensor:
        centred = z - z.mean(dim=0)
        cov = (centred.t() @ centred) / (z.shape[0] - 1)
        return _off_diagonal(cov).pow(2).sum() / z.shape[1]

    def pair_loss(self, x, y, labels):
        invariance = F.mse_loss(x, y)
        variance = self._variance(x) + self._variance(y)
        covariance = self._covariance(x) + self._covariance(y)
        return (self.sim_weight * invariance
                + self.var_weight * variance
                + self.cov_weight * covariance)


@register_loss("simsiam")
class SimSiam(ContrastiveLoss):
    """SimSiam (Chen & He 2021): a predictor and a stop-gradient, no negatives.

    Collapse is prevented by asymmetry alone — one branch goes through a
    predictor MLP, the other is detached. The stop-gradient is not an
    optimisation detail; without it this loss has a trivial constant minimum and
    will find it. Parametric, like `cpc`.
    """

    parametric = True

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(dim // 4, 1)
        self.predictor = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def _half(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return -(_normalise(p) * _normalise(z.detach())).sum(dim=1).mean()

    def pair_loss(self, x, y, labels):
        return 0.5 * (self._half(self.predictor(x), y) + self._half(self.predictor(y), x))


# --------------------------------------------------------------------------
# Label-aware. Read the module docstring on what "positive" means here.
# --------------------------------------------------------------------------


@register_loss("supcon")
class SupervisedContrastive(ContrastiveLoss):
    """SupCon (Khosla et al. 2020), over the pooled multi-view batch.

    Positives are every view of every clip sharing the anchor's class. On a
    [-3, 3] regression target there are no classes, so `bins` discretises it —
    7 bins matching the acc7 metric by default. **That binning is the
    approximation**: two clips at 0.9 and 1.1 land in different bins and become
    negatives of each other, which is a claim about the label this data does not
    make. Reported as `needs_class_labels` so the screening table can say so.
    """

    needs_class_labels = True

    def __init__(self, temperature: float = 0.07, bins: int = 7):
        super().__init__()
        self.temperature, self.bins = temperature, bins

    def _classes(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.dtype in (torch.int64, torch.int32):
            return labels
        # Same binning as metrics.acc7: round into 7 cells over [-3, 3].
        return torch.round(labels.clamp(-3, 3)).long() + 3

    def forward(self, views: Views, labels: torch.Tensor) -> torch.Tensor:
        keys = sorted(views)
        z = _normalise(torch.cat([views[k] for k in keys], dim=0))
        classes = self._classes(labels).repeat(len(keys))

        sim = z @ z.t() / self.temperature
        # Subtract the row max for stability before exponentiating (the standard
        # log-sum-exp trick; SupCon's reference implementation does the same).
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        self_mask = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        positive_mask = (classes[:, None] == classes[None, :]) & ~self_mask

        log_prob = sim - torch.log(sim.exp().masked_fill(self_mask, 0).sum(1, keepdim=True))
        counts = positive_mask.sum(dim=1)
        # Anchors whose class is a singleton in this batch have no positive and
        # contribute nothing; dropping them is not the same as counting a zero.
        valid = counts > 0
        if not valid.any():
            return z.sum() * 0.0
        mean_log_prob = (log_prob * positive_mask).sum(1)[valid] / counts[valid]
        return -mean_log_prob.mean()


@register_loss("rnc")
class RankNContrast(ContrastiveLoss):
    """Rank-N-Contrast (Zha et al. 2023), for continuous targets.

    The one label-aware objective here that needs no binning. For an anchor, all
    other samples are ranked by how far their *label* is from the anchor's, and
    each is contrasted against everything at least that far away. So it learns
    an ordering of the representation that matches the ordering of the target,
    which is the structure a regression head can actually use — and the reason
    it belongs in a screen where SupCon has to bin.
    """

    uses_continuous_labels = True

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, views: Views, labels: torch.Tensor) -> torch.Tensor:
        keys = sorted(views)
        z = _normalise(torch.cat([views[k] for k in keys], dim=0))
        y = labels.reshape(-1).repeat(len(keys))
        n = z.shape[0]

        sim = -torch.cdist(z, z) / self.temperature
        label_distance = (y[:, None] - y[None, :]).abs()
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device)

        losses = []
        for j in range(n):
            # Everything at least as far from the anchor, in label space, as j.
            keep = (label_distance >= label_distance[:, j : j + 1]) & ~self_mask
            keep[:, j] = True
            keep = keep & ~self_mask
            denominator = torch.logsumexp(sim.masked_fill(~keep, float("-inf")), dim=1)
            valid = keep.any(dim=1) & ~self_mask[:, j]
            if valid.any():
                losses.append((denominator - sim[:, j])[valid].mean())
        if not losses:
            return z.sum() * 0.0
        return torch.stack(losses).mean()


@register_loss("arcface")
class ArcFace(ContrastiveLoss):
    """ArcFace's additive angular margin (Deng et al. 2019).

    A margin on the angle to a learned class centre rather than a pairwise
    objective, so unlike everything else here it is only defined once classes
    exist. Included for the classification arm of phase 2; on the regression arm
    it carries the same binning approximation as `supcon` and should be read
    with the same caution.
    """

    needs_class_labels = True
    parametric = True

    def __init__(self, dim: int, num_classes: int = 7, scale: float = 16.0,
                 margin: float = 0.2):
        super().__init__()
        self.scale, self.margin, self.num_classes = scale, margin, num_classes
        self.centres = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.xavier_uniform_(self.centres)

    def _classes(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.dtype in (torch.int64, torch.int32):
            return labels
        return torch.round(labels.clamp(-3, 3)).long() + 3

    def pair_loss(self, x, y, labels):
        classes = self._classes(labels)
        terms = []
        for z in (x, y):
            cosine = (_normalise(z) @ _normalise(self.centres).t()).clamp(-1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cosine)
            target = F.one_hot(classes, self.num_classes).bool()
            margined = torch.where(target, torch.cos(theta + self.margin), cosine)
            terms.append(F.cross_entropy(self.scale * margined, classes))
        return torch.stack(terms).mean()
