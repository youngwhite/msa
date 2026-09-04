"""Attach contrastive objectives to any model, without touching the model.

The alternative — editing fourteen models to add an auxiliary term — is what
this repository was built to avoid, and it would also make every existing
prediction hash unverifiable. So this is a wrapper: it delegates `forward` to
the wrapped model, reads the modality features that model already returns, and
adds the contrastive terms in `compute_loss`.

**The wrapped model's own loss is unchanged.** `compute_loss` calls the base
model's `compute_loss` and adds to it, so MISA keeps its CMD and orthogonality
terms and MMIM keeps its InfoNCE — a wrapped MMIM has two contrastive
objectives, which is a fact about that comparison rather than a bug.

Feature naming is not uniform across the models: TFN and LMF return
``feature_t``/``feature_a``/``feature_v``, Self-MM and TETFN return
``feature_text``/``feature_audio``/``feature_vision``, and MISA returns
``shared``/``private`` dicts instead. `FEATURE_KEYS` maps each convention onto
``t``/``a``/``v``; a model whose features cannot be found raises rather than
silently contrasting nothing.
"""

from __future__ import annotations

import torch

from ..losses.composite import ContrastiveHead
from .base import MSAModel

#: Each entry maps modality -> the key that model family puts it under. Tried in
#: order; the first whose keys are all present wins.
FEATURE_KEYS: tuple[dict[str, str], ...] = (
    {"t": "feature_t", "a": "feature_a", "v": "feature_v"},
    {"t": "feature_text", "a": "feature_audio", "v": "feature_vision"},
)


def find_features(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The per-modality vectors a contrastive term can be computed on.

    Raises when none of the known conventions matches. That is deliberate: the
    quiet alternative is a head that contrasts a single view, or an empty dict,
    and either would train to a number that looks like a result.
    """
    for mapping in FEATURE_KEYS:
        if all(key in outputs for key in mapping.values()):
            return {modality: outputs[key] for modality, key in mapping.items()}
    available = sorted(k for k, v in outputs.items() if isinstance(v, torch.Tensor))
    raise KeyError(
        "no per-modality features in this model's output; looked for "
        f"{[sorted(m.values()) for m in FEATURE_KEYS]}, found {available}. "
        "A model must return pooled modality vectors to be wrapped."
    )


def feature_dims(model: MSAModel, batch: dict[str, torch.Tensor]) -> dict[str, int]:
    """Run one batch to learn the feature widths, rather than trusting kwargs.

    TFN's are 32, 32 and 128 and follow from three separate constructor
    arguments; reading them off a forward pass cannot drift out of date.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        features = find_features(model(batch))
    model.train(was_training)
    return {modality: int(value.shape[-1]) for modality, value in features.items()}


class ContrastiveModel(MSAModel):
    """A model plus a contrastive head on its modality features."""

    def __init__(self, base: MSAModel, head: ContrastiveHead) -> None:
        super().__init__()
        self.base = base
        self.head = head
        self._sums: dict[str, float] = {}
        self._seen = 0
        self._grad_norms: dict[str, float] = {}
        self._measure = False

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"{self.base.name}+contrastive"

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.base(batch)

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        task = self.base.compute_loss(outputs, batch)
        auxiliary, terms, views = self.head(find_features(outputs), batch["label"])

        if self._measure and self.head.scheme.needs_grad_norms:
            self._grad_norms = self._measure_grad_norms(terms, views)
            self._measure = False

        for name, value in terms.items():
            self._sums[name] = self._sums.get(name, 0.0) + float(value.detach())
        self._seen += 1

        return task + auxiliary

    @staticmethod
    def _measure_grad_norms(
        terms: dict[str, torch.Tensor], views: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """L2 norm of the gradient each term sends into the shared projection.

        Measured at the projection rather than at the model's parameters so the
        terms are compared on one common quantity: parameter gradients would mix
        in each modality encoder's own scale, and the weighting scheme would then
        be reacting to how big TFN's vision subnet happens to be.

        `torch.autograd.grad` does not accumulate into `.grad`, so this leaves
        the real backward pass untouched. It costs one extra backward per term
        and runs once per epoch.
        """
        targets = list(views.values())
        norms = {}
        for name, value in terms.items():
            grads = torch.autograd.grad(
                value, targets, retain_graph=True, allow_unused=True
            )
            squared = sum(
                float(g.detach().pow(2).sum()) for g in grads if g is not None
            )
            norms[name] = squared ** 0.5
        return norms

    def contrastive_terms(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Per-term values with the graph intact, for gradient-norm measurement."""
        return self.head.term_values(
            self.head.project(find_features(outputs)), batch["label"]
        )

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """The base model's groups, plus the head as its own group.

        The head is listed separately rather than folded into the base's groups
        because several models ask for a smaller learning rate on a fine-tuned
        text encoder, and a freshly initialised projection head is the opposite
        case. It gets the plain `lr`.
        """
        groups = self.base.param_groups(lr, weight_decay)
        return [*groups, {"params": list(self.head.parameters()), "lr": lr,
                          "weight_decay": weight_decay}]

    def on_train_epoch_start(self, epoch: int) -> None:
        self.base.on_train_epoch_start(epoch)
        self._sums, self._seen = {}, 0
        # Measure on the first batch of the epoch, not the last: the last one is
        # usually partial, and a gradient norm computed on a third of a batch is
        # not comparable to the ones before it.
        self._measure = True

    def on_train_epoch_end(self, epoch: int) -> dict[str, float]:
        """Record this epoch's term values, gradient norms and weights.

        `observe` is called here rather than per batch because the schemes are
        defined on epoch-level quantities — a weight that moved every batch
        would be reacting to batch noise, and the 10-epoch window phase 2 asks
        for would have no meaning.
        """
        means = {
            name: total / max(self._seen, 1) for name, total in self._sums.items()
        }
        self.head.scheme.observe(
            epoch, means, self._grad_norms if self._grad_norms else None
        )
        recorded = {f"loss_{name}": value for name, value in means.items()}
        recorded.update({f"gradnorm_{k}": v for k, v in self._grad_norms.items()})
        recorded.update({
            f"weight_{k}": v for k, v in self.head.scheme.weights().items()
        })
        return {**self.base.on_train_epoch_end(epoch), **recorded}

    def on_train_batch_end(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], epoch: int
    ) -> None:
        self.base.on_train_batch_end(outputs, batch, epoch)
