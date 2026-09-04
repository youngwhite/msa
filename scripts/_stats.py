"""Shared statistics for the phase-2 experiment scripts.

Three of them needed the same Welch test, the same BH correction and the same
"read a run group off disk" logic, so it lives here once. The parts that are
*not* shared are deliberate: each experiment's arms, seeds and criterion are
written into its own script, because those are the things that must be fixed
before it runs and read afterwards without hunting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

METRICS = ("mae", "corr")
LOWER_IS_BETTER = {"mae"}


def read_group(directory: Path, expected_seeds: int) -> dict | None:
    """Validation metrics per seed, from the epoch each run selected.

    Returns None -- with a warning -- when the group has the wrong number of
    seeds. A sweep killed partway leaves a partial group, and averaging whatever
    is present into a row labelled n=20 is a quietly wrong sample size in the
    half of the pipeline that decides what counts as signal.
    """
    if not directory.exists():
        return None
    per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
    seeds, trajectory = [], {}
    for path in sorted(directory.glob("seed*/result.json")):
        result = json.loads(path.read_text())
        if result.get("env", {}).get("git", {}).get("dirty", True):
            print(f"    warning: {path.parent.name} from a dirty tree", file=sys.stderr)
        best = next(r for r in result["history"] if r["epoch"] == result["best_epoch"])
        for metric in METRICS:
            per_metric[metric].append(best[f"valid_{metric}"])
        seeds.append(result["seed"])
        for key, value in result["history"][-1].items():
            if key.startswith(("weight_", "loss_", "gradnorm_")):
                trajectory.setdefault(key, []).append(value)
    if not seeds:
        return None
    if len(seeds) != expected_seeds:
        print(f"    warning: {directory.name} has {len(seeds)}/{expected_seeds} "
              f"seeds — excluded", file=sys.stderr)
        return None
    return {
        "group": directory.name,
        "seeds": seeds,
        "valid": {
            m: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                "per_seed": v}
            for m, v in per_metric.items()
        },
        "final": {k: float(np.mean(v)) for k, v in sorted(trajectory.items())},
    }


def welch_one_sided(candidate: list[float], control: list[float], metric: str) -> dict:
    """One-sided Welch test for "candidate is better than control" on `metric`.

    Welch rather than Student because the two arms have no reason to share a
    variance -- an auxiliary loss can make a run more or less seed-sensitive,
    and in the screen `simsiam` roughly halved the spread.

    Unpaired even when the arms share seeds. Adding a loss term changes the whole
    trajectory, and the per-seed correlation between arms was measured at a
    median of about zero with a range spanning -0.99 to +0.98, so the paired
    premise does not hold -- the same conclusion phase 1 reached across
    implementations.
    """
    from scipy import stats

    delta = float(np.mean(candidate) - np.mean(control))
    improvement = -delta if metric in LOWER_IS_BETTER else delta
    statistic, p_two = stats.ttest_ind(candidate, control, equal_var=False)
    if metric in LOWER_IS_BETTER:
        statistic = -statistic
    pooled = float(np.sqrt(
        (np.var(candidate, ddof=1) + np.var(control, ddof=1)) / 2
    ))
    return {
        "improvement": improvement,
        "p_one_sided": float(p_two / 2 if statistic > 0 else 1 - p_two / 2),
        "cohens_d": improvement / pooled if pooled else float("nan"),
        "pooled_sd": pooled,
    }


def benjamini_hochberg(p_values: list[float], q: float) -> list[bool]:
    """Which hypotheses BH rejects at false-discovery rate `q`.

    FDR rather than Bonferroni throughout phase 2: these are screens and
    confirmations, where the cost of a false positive is one wasted follow-up
    run rather than a wrong published claim.
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    n, threshold_rank = len(p_values), 0
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= q * rank / n:
            threshold_rank = rank
    rejected = [False] * n
    for rank, index in enumerate(order, start=1):
        if rank <= threshold_rank:
            rejected[index] = True
    return rejected


def minimum_detectable_effect(pooled_sd: float, n_per_arm: int,
                              alpha: float = 0.05, power: float = 0.80) -> float:
    """The smallest true effect this design would find, at `power`.

    Worth computing *before* a sweep, not after. Phase 2's screen ran 140 tests
    whose minimum detectable effect was 0.076 MAE against observed effects of
    0.017 -- arithmetic that depends only on n and sd, and would have said so in
    advance. See docs/investigations.md#screen-underpowered.
    """
    from scipy import stats

    df = 2 * n_per_arm - 2
    se = pooled_sd * np.sqrt(2 / n_per_arm)
    return float((stats.t.ppf(1 - alpha, df) + stats.t.ppf(power, df)) * se)
