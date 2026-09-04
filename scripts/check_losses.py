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

All of it runs on the resolved device, CUDA included. The first version ran only
on the CPU and so did not catch `hcl` building a scalar with `torch.tensor(...)`
-- a CPU tensor, which `clamp_min` refuses against a CUDA input. That failed on
the seventh group of a screening sweep rather than in the gate written to
prevent exactly this.

Exit code 0 if every loss passes, non-zero otherwise.
"""

from __future__ import annotations

import torch

from msa.device import describe_device, resolve_device
from msa.losses import available_losses, get_loss_class
from msa.repro import set_seed

BATCH, DIM = 24, 32
DEVICE = resolve_device("auto")

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
    return cls(**kwargs).to(DEVICE)


def make_views(shuffle: bool = False) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Three correlated views of one batch, plus a continuous [-3, 3] label.

    Correlated, not independent: a real encoder's modality outputs share the
    signal the label carries, and an objective that only ever sees orthogonal
    noise would look identical for every candidate.
    """
    set_seed(0)
    shared = torch.randn(BATCH, DIM)
    views = {
        modality: (shared + 0.5 * torch.randn(BATCH, DIM)).to(DEVICE)
        for modality in ("t", "a", "v")
    }
    if shuffle:
        views["a"] = views["a"][torch.randperm(BATCH, device=DEVICE)]
    labels = torch.empty(BATCH).uniform_(-3, 3).to(DEVICE)
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


def check_composite() -> list[str]:
    """The properties the weighted combination has to have to be interpretable.

    1. **Rescaling equalises influence.** The whole point of dividing each term
       by its running magnitude is that a weight becomes a share of influence.
       Tested on values with a deliberately extreme spread (1e-3 against 1e3)
       rather than on whatever spread a synthetic batch happens to produce --
       the real one is ~1e4 between barlow_twins and rnc, and a check that
       depended on reproducing that would be measuring the test batch.
    2. **Weights sum to 1** under every scheme, at every epoch. A scheme whose
       weights drifted off 1 would silently change the effective lambda, and the
       comparison between schemes would be measuring that instead.
    3. **`equal` never moves.** It is the control; if it drifts, the adaptive
       arms have nothing to be compared against.
    4. **`grad_rate` holds still for `window` epochs, then moves.** Its whole
       claim is that it reacts to a 10-epoch trend, so reacting sooner would
       mean it is reacting to noise.
    5. **`linear_ramp` reaches its target and does not overshoot.**
    """
    problems = []
    from msa.losses.composite import SCHEMES, ContrastiveHead

    names = ["infonce", "barlow_twins", "rnc"]
    views, labels = make_views()
    dims = {m: DIM for m in views}

    head = ContrastiveHead(
        feature_dims=dims,
        losses={n: build(n) for n in names},
        scheme=SCHEMES["equal"](names),
        projection_dim=DIM,
    ).to(DEVICE)
    head.train()
    # Six orders of magnitude apart, which no real pair of objectives reaches.
    extreme = {"infonce": 1e-3, "barlow_twins": 1e3, "rnc": 1.0}
    rescaled = {
        name: float(head.rescale(name, torch.tensor(value, device=DEVICE)))
        for name, value in extreme.items()
    }
    for name, value in rescaled.items():
        if abs(value - 1.0) > 1e-3:
            problems.append(
                f"rescale({name}, {extreme[name]:g}) = {value:.6f}, expected ~1.0 "
                f"on the first observation"
            )

    # And it must track, not freeze: a term that grows tenfold should come back
    # towards 1 rather than staying at 10 forever.
    for _ in range(2000):
        head.rescale("infonce", torch.tensor(1e-2, device=DEVICE))
    tracked = float(head.rescale("infonce", torch.tensor(1e-2, device=DEVICE)))
    if abs(tracked - 1.0) > 0.05:
        problems.append(
            f"running scale did not track a tenfold change: rescaled to {tracked:.4f}"
        )

    # The head must still run end to end on real term values.
    _, raw, _ = head(views, labels)
    if not all(torch.isfinite(v) for v in raw.values()):
        problems.append(f"non-finite term in the combination: {raw}")

    for scheme_name in sorted(SCHEMES):
        kwargs = {}
        if scheme_name == "grad_rate":
            kwargs = {"window": 3}
        elif scheme_name == "linear_ramp":
            kwargs = {"favour": "infonce", "total_epochs": 10, "target": 0.7}
        scheme = SCHEMES[scheme_name](names, **kwargs)
        seen = []
        for epoch in range(1, 11):
            scheme.observe(
                epoch,
                dict.fromkeys(names, 1.0),
                {n: 1.0 + 0.5 * epoch * (i + 1) for i, n in enumerate(names)},
            )
            weights = scheme.weights()
            total = sum(weights.values())
            if abs(total - 1.0) > 1e-6:
                problems.append(f"{scheme_name}: weights sum to {total} at epoch {epoch}")
                break
            seen.append(weights)

        if scheme_name == "equal" and any(w != seen[0] for w in seen):
            problems.append("equal: weights moved, so it is not a control")
        if scheme_name == "grad_rate":
            if any(w != seen[0] for w in seen[:3]):
                problems.append("grad_rate: moved before its window elapsed")
            elif all(w == seen[0] for w in seen):
                problems.append("grad_rate: never moved at all")
        if scheme_name == "linear_ramp":
            final = seen[-1]["infonce"]
            if abs(final - 0.7) > 1e-6:
                problems.append(f"linear_ramp: reached {final:.4f}, target was 0.7")
            if max(w["infonce"] for w in seen) > 0.7 + 1e-6:
                problems.append("linear_ramp: overshot its target")

    return problems


def main() -> int:
    names = available_losses()
    print(f"== {len(names)} registered objective(s) on {describe_device(DEVICE)} ==\n")
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

    print("\n== weighted combination ==")
    composite = check_composite()
    for problem in composite:
        print(f"  FAIL {problem}")
    if not composite:
        print("  ok   rescaling equalises influence; every scheme sums to 1 and "
              "moves when it should")

    print()
    if failed or composite:
        if failed:
            print(f"{len(failed)} objective(s) failed: {', '.join(sorted(failed))}")
        if composite:
            print(f"{len(composite)} problem(s) in the weighted combination")
        return 1
    print("every objective is finite, differentiable, pairing-sensitive and deterministic; "
          "the combination weights are interpretable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
