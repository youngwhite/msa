"""Phase 2, step 2: does adapting the loss weights beat holding them fixed?

This is the question the phase actually asks, and it is not the same question
the screen was asking. It compares *weighting rules* on one fixed set of terms,
so it does not require each term to be individually beneficial -- which matters,
because the screen could not establish that and, at n=5 on this backbone, could
not have (docs/investigations.md#screen-underpowered).

Five arms, seeds 100-119, everything else identical to the screen:

    weight_control   plain L1, no contrastive part at all
    weight_equal     fixed 1/4 on each term          <- the control
    weight_grad10    grad_rate, window 10            (the brief's wording)
    weight_grad5     grad_rate, window 5
    weight_ramp      linear_ramp towards infonce, completing at epoch 20

Terms: hcl, infonce, simsiam, supcon at lambda 0.1, chosen by the stated rule in
`screen_contrastive.py combine` before any of this ran. See decisions.md.

**Fresh seeds on purpose.** 42-46 selected the terms; reusing them here would
stamp approval on this experiment's own selection bias. 20 of them rather than 5
brings the minimum detectable effect to about 0.019 MAE, which is the size of
effect anything here has shown.

**Why a second grad_rate arm.** Across 355 screen runs the median run trains for
24 epochs -- 9% reach 40, 70% reach 20. A 10-epoch window therefore gets one or
two chances to act on a median run, so without a shorter-window arm "the rule
does not help" and "the rule never ran" are the same observation. Window 5 gives
a median run about four.

The primary comparison is each adaptive arm against `weight_equal`: Welch
one-sided on validation MAE and Corr, BH over those six tests. `weight_control`
answers a different question -- whether the combination does anything at all --
and is reported separately rather than folded into the primary claim.

    scripts/weighting_experiment.py run
    scripts/weighting_experiment.py report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
PYTHON = REPO / ".venv" / "bin" / "python"
TRAIN = REPO / "scripts" / "train.py"
REPORT = REPO / "docs" / "weighting_experiment.json"

BACKBONE = ["--model", "tfn", "--unaligned", "--lr", "1e-3", "--weight-decay", "0"]
SEEDS = [str(s) for s in range(100, 120)]
TERMS = ["hcl", "infonce", "simsiam", "supcon"]
LAMBDA = "0.1"
FAVOUR = "infonce"
METRICS = ("mae", "corr")
LOWER_IS_BETTER = {"mae"}
FDR_Q = 0.10
CONTROL_ARM = "weight_equal"

#: Arm -> the contrastive options that define it. The backbone, seeds, terms and
#: lambda are identical across every arm that has a contrastive part, so the only
#: thing that differs is the weighting rule.
ARMS: dict[str, list[str]] = {
    "weight_control": [],
    "weight_equal": ["--weight-scheme", "equal"],
    "weight_grad10": ["--weight-scheme", "grad_rate", "--weight-window", "10"],
    "weight_grad5": ["--weight-scheme", "grad_rate", "--weight-window", "5"],
    # end-fraction 0.5: the ramp completes at epoch 20, which 70% of runs reach.
    # Left at 1.0 it would span 40 epochs, and the median run's 24 would get it
    # only 60% of the way -- an arm that barely differs from `weight_equal`.
    "weight_ramp": ["--weight-scheme", "linear_ramp", "--favour", FAVOUR,
                    "--favour-target", "0.7", "--favour-end-fraction", "0.5"],
}


def run_one(arm: str, epochs: int, force: bool) -> None:
    directory = OUTPUTS / arm
    if directory.exists() and not force:
        present = sorted(path.name for path in directory.glob("seed*"))
        if len(present) == len(SEEDS):
            print(f"  {arm}: already on disk, skipping")
            return
        print(f"  {arm}: incomplete ({len(present)}/{len(SEEDS)} seeds), redoing")

    command = [str(PYTHON), str(TRAIN), *BACKBONE,
               "--seeds", *SEEDS, "--epochs", str(epochs),
               "--run-group", arm, "--quiet"]
    if arm != "weight_control":
        command += ["--contrastive", *TERMS, "--contrastive-lambda", LAMBDA]
        command += ARMS[arm]
    print(f"  {arm}: {' '.join(ARMS[arm]) or 'plain L1'}")
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode})")
        print("\n".join(result.stdout.strip().splitlines()[-8:]))
        print(result.stderr.strip()[-800:])


def read_arm(arm: str) -> dict | None:
    directory = OUTPUTS / arm
    if not directory.exists():
        return None
    per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
    seeds, weights, terms = [], {}, {}
    for path in sorted(directory.glob("seed*/result.json")):
        result = json.loads(path.read_text())
        if result.get("env", {}).get("git", {}).get("dirty", True):
            print(f"    warning: {path.parent.name} from a dirty tree", file=sys.stderr)
        best = next(r for r in result["history"] if r["epoch"] == result["best_epoch"])
        for metric in METRICS:
            per_metric[metric].append(best[f"valid_{metric}"])
        seeds.append(result["seed"])
        final = result["history"][-1]
        for key, value in final.items():
            if key.startswith("weight_"):
                weights.setdefault(key, []).append(value)
            elif key.startswith("loss_"):
                terms.setdefault(key, []).append(value)
    if len(seeds) != len(SEEDS):
        print(f"    warning: {arm} has {len(seeds)}/{len(SEEDS)} seeds — excluded",
              file=sys.stderr)
        return None
    return {
        "arm": arm,
        "seeds": seeds,
        "valid": {
            m: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                "per_seed": v}
            for m, v in per_metric.items()
        },
        "final_weights": {k: float(np.mean(v)) for k, v in sorted(weights.items())},
        "final_terms": {k: float(np.mean(v)) for k, v in sorted(terms.items())},
    }


def welch_one_sided(candidate: list[float], control: list[float], metric: str
                    ) -> tuple[float, float]:
    from scipy import stats

    delta = float(np.mean(candidate) - np.mean(control))
    improvement = -delta if metric in LOWER_IS_BETTER else delta
    statistic, p_two = stats.ttest_ind(candidate, control, equal_var=False)
    if metric in LOWER_IS_BETTER:
        statistic = -statistic
    return improvement, float(p_two / 2 if statistic > 0 else 1 - p_two / 2)


def benjamini_hochberg(p_values: list[float], q: float) -> list[bool]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    n, threshold_rank = len(p_values), -1
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= q * rank / n:
            threshold_rank = rank
    rejected = [False] * n
    for rank, index in enumerate(order, start=1):
        if rank <= threshold_rank:
            rejected[index] = True
    return rejected


def report() -> int:
    arms = {name: read_arm(name) for name in ARMS}
    control = arms.get(CONTROL_ARM)
    if control is None:
        print(f"No {CONTROL_ARM} on disk. Run `weighting_experiment.py run` first.")
        return 1

    adaptive = [n for n, d in arms.items()
                if d is not None and n not in (CONTROL_ARM, "weight_control")]
    tests = [(name, metric) for name in adaptive for metric in METRICS]
    results = {}
    for name, metric in tests:
        results[(name, metric)] = welch_one_sided(
            arms[name]["valid"][metric]["per_seed"],
            control["valid"][metric]["per_seed"], metric,
        )
    rejected = benjamini_hochberg([results[k][1] for k in tests], FDR_Q)
    significant = dict(zip(tests, rejected, strict=True))

    n = len(control["seeds"])
    se = {m: float(control["valid"][m]["sd"] / np.sqrt(n)) for m in METRICS}
    print(f"backbone: {' '.join(BACKBONE)}   terms: {' '.join(TERMS)}   "
          f"lambda {LAMBDA}")
    print(f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={n})   "
          f"primary: each adaptive arm vs {CONTROL_ARM}, "
          f"Welch one-sided, BH q={FDR_Q} over {len(tests)} tests")
    print(f"control SE: mae {se['mae']:.4f}  corr {se['corr']:.4f}   "
          f"(MDE at 80% power ≈ {2.5 * se['mae'] * np.sqrt(2):.4f} mae)\n")

    header = f"{'arm':16s}{'valid mae':>19s}{'valid corr':>19s}   verdict"
    print(header)
    print("-" * len(header))
    for name in ARMS:
        data = arms.get(name)
        if data is None:
            print(f"{name:16s}{'not run':>19s}")
            continue
        note = ""
        if name == CONTROL_ARM:
            note = "<- control"
        elif name == "weight_control":
            note = "(separate question: does the combination do anything)"
        else:
            marks = [m for m in METRICS if significant[(name, m)]]
            note = ("BH sig: " + ",".join(marks)) if marks else "no difference"
        print(f"{name:16s}"
              f"{data['valid']['mae']['mean']:12.4f} ± {data['valid']['mae']['sd']:.4f}"
              f"{data['valid']['corr']['mean']:12.4f} ± {data['valid']['corr']['sd']:.4f}"
              f"   {note}")

    print(f"\n{'arm':16s}{'Δmae vs equal':>16s}{'p':>8s}{'Δcorr vs equal':>17s}{'p':>8s}")
    for name in adaptive:
        d_mae, p_mae = results[(name, "mae")]
        d_corr, p_corr = results[(name, "corr")]
        print(f"{name:16s}{d_mae:+16.4f}{p_mae:8.3f}{d_corr:+17.4f}{p_corr:8.3f}")

    print(f"\nBH rejected {sum(rejected)}/{len(tests)} test(s) at q={FDR_Q}")

    print("\nfinal mean weight per term (says whether the schemes actually moved):")
    for name in ARMS:
        data = arms.get(name)
        if data is None or not data["final_weights"]:
            continue
        parts = " ".join(f"{k.replace('weight_', '')}={v:.3f}"
                         for k, v in data["final_weights"].items())
        print(f"  {name:16s}{parts}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "backbone": BACKBONE, "terms": TERMS, "lambda": float(LAMBDA),
        "seeds": [int(s) for s in SEEDS], "control_arm": CONTROL_ARM,
        "criterion": {"test": "Welch one-sided, unpaired",
                      "correction": f"Benjamini-Hochberg q={FDR_Q}",
                      "n_tests": len(tests)},
        "arms": {k: v for k, v in arms.items() if v is not None},
        "comparisons": [
            {"arm": name, "metric": metric,
             "improvement": results[(name, metric)][0],
             "p_one_sided": results[(name, metric)][1],
             "bh_significant": bool(significant[(name, metric)])}
            for name, metric in tests
        ],
    }, indent=2) + "\n")
    print(f"\nwrote {REPORT.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "report"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--only", nargs="+", default=None, choices=sorted(ARMS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.command == "report":
        return report()

    wanted = args.only or list(ARMS)
    print(f"== {len(wanted)} arm(s) x {len(SEEDS)} seeds ==")
    for arm in wanted:
        run_one(arm, args.epochs, args.force)
    print("\nDone. `weighting_experiment.py report` for the table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
