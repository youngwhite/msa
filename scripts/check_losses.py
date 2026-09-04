"""Do the contrastive objectives behave like contrastive objectives?

"It runs without raising" is not evidence. A loss that returns a constant, or
whose gradient does not reach the encoder, or that scores a shuffled batch as
well as an aligned one, would train perfectly quietly and produce a screening
table full of noise. This checks the properties that would make each candidate's
number mean something, before any of them is trained.

Per loss:

1. **Finite and scalar** on a plausible batch.
2. **Gradient reaches the inputs**, and is not all zero. A silently detached
   objective is the failure this project has already seen in another form --
   convention 5's "declared but inert layer".
3. **Aligned beats shuffled.** With views that genuinely correspond, the loss
   must be lower than with one view's rows permuted. This is the discriminating
   test: it is what separates an objective that uses the pairing from one that
   merely consumes it. Losses that do not read the pairing at all are exempt and
   say so.
4. **Deterministic** under a fixed seed, twice in a row -- the project reports
   bit-identical runs, and a loss with hidden randomness would break that.

Exit code 0 if every loss passes, non-zero otherwise.
"""

from __future__ import annotations

import torch

from msa.losses import available_losses, get_loss_class
from msa.repro import set_seed

BATCH, DIM = 24, 32

#: Objectives that are invariant to which row pairs with which, by construction.
#: `supcon` and `arcface` group by label, not by correspondence; `align_uniform`
#: is exempt only in its uniformity half, but its alignment half does read the
#: pairing, so it stays in the test.
PAIRING_BLIND = {"supcon", "arcface"}

#: Steps of Adam given to a parametric objective before the pairing test. Its
#: score function is *learned*, so at initialisation it carries no information
#: about the correspondence and the test would be asking the wrong question --
#: not "does this objective use the pairing" but "can an untrained random
#: projection use it", whose answer is no for any of them. Small, because the
#: claim under test is that the signal is learnable at all, not how fast.
FIT_STEPS = 100


def build(name: str) -> torch.nn.Module:
    cls = get_loss_class(name)
    kwargs = {}
    if getattr(cls, "parametric", False):
        kwargs["dim"] = DIM
    return cls(**kwargs)


def make_views(shuffle: bool = False) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Three correlated views of one batch, plus a continuous [-3, 3] label.

    Correlated, not independent: a real encoder's modality outputs share the
    signal the label carries, and an objective that only ever sees orthogonal
    noise would look identical for every candidate.
    """
    set_seed(0)
    shared = torch.randn(BATCH, DIM)
    views = {
        modality: shared + 0.5 * torch.randn(BATCH, DIM)
        for modality in ("t", "a", "v")
    }
    if shuffle:
        views["a"] = views["a"][torch.randperm(BATCH)]
    labels = torch.empty(BATCH).uniform_(-3, 3)
    for view in views.values():
        view.requires_grad_(True)
    return views, labels


def pairing_scores(name: str) -> tuple[float, float]:
    """Loss on corresponding views vs on views with one modality permuted.

    A parametric objective is given `FIT_STEPS` of Adam on the aligned batch
    first; then the *same fitted* objective scores both, so the comparison is
    between two inputs to one function rather than between two functions.
    """
    set_seed(1)
    loss_fn = build(name)
    if getattr(get_loss_class(name), "parametric", False):
        optimiser = torch.optim.Adam(loss_fn.parameters(), lr=1e-2)
        views, labels = make_views()
        for _ in range(FIT_STEPS):
            optimiser.zero_grad()
            loss_fn(views, labels).backward()
            optimiser.step()
    loss_fn.eval()
    with torch.no_grad():
        aligned = loss_fn(*make_views(shuffle=False)).item()
        shuffled = loss_fn(*make_views(shuffle=True)).item()
    return aligned, shuffled


def check(name: str) -> list[str]:
    problems = []
    views, labels = make_views()
    loss_fn = build(name)

    set_seed(1)
    value = loss_fn(views, labels)
    if value.ndim != 0:
        problems.append(f"not a scalar: shape {tuple(value.shape)}")
    if not torch.isfinite(value):
        problems.append(f"not finite: {value.item()}")
    if problems:
        return problems

    value.backward()
    grads = [v.grad for v in views.values() if v.grad is not None]
    if len(grads) < 2:
        problems.append("gradient reached fewer than two views")
    elif all(g.abs().max() == 0 for g in grads):
        problems.append("gradient is identically zero at every view")

    if name not in PAIRING_BLIND:
        aligned, shuffled = pairing_scores(name)
        if not shuffled > aligned:
            note = " (after fitting its score function)" if getattr(
                get_loss_class(name), "parametric", False) else ""
            problems.append(
                f"does not read the pairing{note}: aligned {aligned:.5f} "
                f"vs shuffled {shuffled:.5f}"
            )

    set_seed(1)
    first = build(name)(*make_views()).item()
    set_seed(1)
    second = build(name)(*make_views()).item()
    if first != second:
        problems.append(f"not deterministic: {first!r} then {second!r}")

    return problems


def main() -> int:
    names = available_losses()
    print(f"== {len(names)} registered objective(s) ==\n")
    failed = {}
    for name in names:
        cls = get_loss_class(name)
        tags = [t for t, on in (
            ("class-labels", cls.needs_class_labels),
            ("continuous-labels", cls.uses_continuous_labels),
            ("parametric", cls.parametric),
        ) if on]
        problems = check(name)
        status = "ok" if not problems else "FAIL"
        print(f"  {status:4s} {name:14s} {' '.join(tags)}")
        for problem in problems:
            print(f"         - {problem}")
        if problems:
            failed[name] = problems

    print()
    if failed:
        print(f"{len(failed)} objective(s) failed: {', '.join(sorted(failed))}")
        return 1
    print("every objective is finite, differentiable, pairing-sensitive and deterministic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
