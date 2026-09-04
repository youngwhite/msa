"""Phase 2, step 3: does the four-term combination really beat plain L1?

The five-arm experiment observed it as a *secondary* comparison -- +0.0131 MAE
(one-sided p=0.045) and +0.0088 Corr (p=0.027), Cohen's d around 0.55-0.63,
against plain L1 on seeds 100-119. That was not part of its primary criterion,
it was tested after the fact, and it did not pass the BH correction the primary
comparison used. So it is a lead, not a result.

It also sits almost exactly on the detection boundary: with n=20 the minimum
detectable effect was 0.0133 MAE and the observed effect was 0.0131. That is
the position where an independent repeat matters most, because it is the
position where a single sample is least informative.

Two arms, two metrics, two tests, and a **third disjoint seed set**:

    confirm_l1      plain L1
    confirm_combo   hcl + infonce + simsiam + supcon, equal weights, lambda 0.1

Seeds 120-139. Seeds 42-46 chose the four terms and 100-119 produced the lead,
so both are spent -- using either again would be confirming this work's own
selection rather than testing it.

Everything else is identical to the five-arm experiment. The criterion is fixed
here before the runs exist: Welch one-sided on validation MAE and Corr, BH at
q=0.10 over the two tests. If it does not hold, the lead is recorded as an
unreplicated observation and is not a conclusion.

    scripts/confirm_combination.py run
    scripts/confirm_combination.py report
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
REPORT = REPO / "docs" / "confirm_combination.json"

BACKBONE = ["--model", "tfn", "--unaligned", "--lr", "1e-3", "--weight-decay", "0"]
SEEDS = [str(s) for s in range(120, 140)]
TERMS = ["hcl", "infonce", "simsiam", "supcon"]
LAMBDA = "0.1"
METRICS = ("mae", "corr")
LOWER_IS_BETTER = {"mae"}
FDR_Q = 0.10

#: The lead being tested, from docs/weighting_experiment.json on seeds 100-119.
PRIOR = {"mae": 0.0131, "corr": 0.0088}

ARMS = {
    "confirm_l1": [],
    "confirm_combo": ["--contrastive", *TERMS, "--contrastive-lambda", LAMBDA,
                      "--weight-scheme", "equal"],
}


def run_one(arm: str, epochs: int, force: bool) -> None:
    directory = OUTPUTS / arm
    if directory.exists() and not force:
        present = sorted(path.name for path in directory.glob("seed*"))
        if len(present) == len(SEEDS):
            print(f"  {arm}: already on disk, skipping")
            return
        print(f"  {arm}: incomplete ({len(present)}/{len(SEEDS)}), redoing")
    command = [str(PYTHON), str(TRAIN), *BACKBONE, "--seeds", *SEEDS,
               "--epochs", str(epochs), "--run-group", arm, "--quiet", *ARMS[arm]]
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
    seeds = []
    for path in sorted(directory.glob("seed*/result.json")):
        result = json.loads(path.read_text())
        if result.get("env", {}).get("git", {}).get("dirty", True):
            print(f"    warning: {path.parent.name} from a dirty tree", file=sys.stderr)
        best = next(r for r in result["history"] if r["epoch"] == result["best_epoch"])
        for metric in METRICS:
            per_metric[metric].append(best[f"valid_{metric}"])
        seeds.append(result["seed"])
    if len(seeds) != len(SEEDS):
        print(f"    warning: {arm} has {len(seeds)}/{len(SEEDS)} seeds — excluded",
              file=sys.stderr)
        return None
    return {
        "arm": arm, "seeds": seeds,
        "valid": {m: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                      "per_seed": v} for m, v in per_metric.items()},
    }


def report() -> int:
    from scipy import stats

    combo, l1 = read_arm("confirm_combo"), read_arm("confirm_l1")
    if combo is None or l1 is None:
        print("Both arms must be complete. Run `confirm_combination.py run`.")
        return 1

    outcomes = {}
    for metric in METRICS:
        a, b = combo["valid"][metric]["per_seed"], l1["valid"][metric]["per_seed"]
        delta = float(np.mean(a) - np.mean(b))
        improvement = -delta if metric in LOWER_IS_BETTER else delta
        statistic, p_two = stats.ttest_ind(a, b, equal_var=False)
        if metric in LOWER_IS_BETTER:
            statistic = -statistic
        p_one = float(p_two / 2 if statistic > 0 else 1 - p_two / 2)
        pooled = float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2))
        outcomes[metric] = {
            "improvement": improvement, "p_one_sided": p_one,
            "cohens_d": improvement / pooled if pooled else float("nan"),
            "prior_estimate": PRIOR[metric],
        }

    # BH over exactly two tests.
    ordered = sorted(METRICS, key=lambda m: outcomes[m]["p_one_sided"])
    threshold_rank = 0
    for rank, metric in enumerate(ordered, start=1):
        if outcomes[metric]["p_one_sided"] <= FDR_Q * rank / len(METRICS):
            threshold_rank = rank
    for rank, metric in enumerate(ordered, start=1):
        outcomes[metric]["bh_significant"] = rank <= threshold_rank

    print(f"backbone: {' '.join(BACKBONE)}   terms: {' '.join(TERMS)}  lambda {LAMBDA}")
    print(f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)}), disjoint from 42-46 "
          f"and 100-119")
    print(f"criterion fixed before these runs: Welch one-sided, BH q={FDR_Q}, "
          f"2 tests\n")
    for arm, data in (("confirm_l1 (plain L1)", l1), ("confirm_combo", combo)):
        print(f"  {arm:24s}"
              f"mae {data['valid']['mae']['mean']:.4f} ± {data['valid']['mae']['sd']:.4f}"
              f"   corr {data['valid']['corr']['mean']:.4f} ± "
              f"{data['valid']['corr']['sd']:.4f}")

    print(f"\n{'metric':8s}{'Δ (combo - L1)':>16s}{'p':>8s}{'d':>7s}"
          f"{'prior Δ':>10s}   BH")
    for metric in METRICS:
        o = outcomes[metric]
        print(f"{metric:8s}{o['improvement']:+16.4f}{o['p_one_sided']:8.4f}"
              f"{o['cohens_d']:+7.2f}{o['prior_estimate']:+10.4f}   "
              f"{'yes' if o['bh_significant'] else 'no'}")

    confirmed = [m for m in METRICS if outcomes[m]["bh_significant"]]
    print()
    if confirmed:
        print(f"CONFIRMED on {', '.join(confirmed)}: the combination beats plain L1 "
              f"on a third\nindependent seed set. Still bound to TFN + MOSI and to "
              f"these four terms.")
    else:
        print("NOT CONFIRMED. The five-arm lead does not replicate on seeds "
              f"{SEEDS[0]}-{SEEDS[-1]};\nit is an unreplicated observation, not a "
              "result. Do not report it as one.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "backbone": BACKBONE, "terms": TERMS, "lambda": float(LAMBDA),
        "seeds": [int(s) for s in SEEDS],
        "criterion": {"test": "Welch one-sided, unpaired",
                      "correction": f"Benjamini-Hochberg q={FDR_Q}", "n_tests": 2},
        "arms": {"confirm_l1": l1, "confirm_combo": combo},
        "outcomes": outcomes,
        "confirmed_on": confirmed,
    }, indent=2) + "\n")
    print(f"\nwrote {REPORT.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "report"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.command == "report":
        return report()
    print(f"== 2 arms x {len(SEEDS)} seeds ==")
    for arm in ARMS:
        run_one(arm, args.epochs, args.force)
    print("\nDone. `confirm_combination.py report` for the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
