"""Phase 2, step 6: does DCL's gain on MMIM/MOSEI replicate?

The screen on MMIM(contrast=False)/MOSEI eliminated all four candidates under
its pre-registered criterion -- BH rejected 0 of 8 -- but every one of them
improved on the control, and `dcl` came in at +0.0100 MAE with a raw two-sided
p of 0.037. That is a lead, not a result, and it sits almost exactly on the
design's detection boundary: observed 0.0100 against an MDE of 0.0129.

That position is where this project has already been wrong once. The four-term
combination beat plain L1 by +0.0131 MAE at raw p=0.045 on TFN/MOSI, and on a
third seed set the sign flipped (docs/experiments.md, confirm_combination). So
the same discipline applies here.

Two arms, two metrics, two tests, and a **fresh seed set**:

    dclconfirm_mosei_control   MMIM(contrast=False)
    dclconfirm_mosei_dcl       MMIM(contrast=False) + dcl, lambda 0.5

Seeds 120-139. Seeds 100-119 produced the lead and are spent -- reusing them
would confirm this work's own selection rather than test it. Both arms rerun, so
the control is on the same seeds as the candidate rather than inherited from the
screen.

At n=20 per arm the minimum detectable effect is 0.0089 against the observed
0.0100, so the design can resolve the lead if it is real. Criterion is fixed
before these runs: Welch one-sided for improvement and two-sided for either
direction, BH at q=0.10 over the two metrics. If it does not hold, the lead is
recorded as an unreplicated observation and DCL is not carried into the
combination step.

    scripts/confirm_dcl.py run
    scripts/confirm_dcl.py report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from _stats import (  # noqa: E402
    METRICS,
    benjamini_hochberg,
    minimum_detectable_effect,
    read_group,
    welch_one_sided,
)

OUTPUTS = REPO / "outputs"
PYTHON = REPO / ".venv" / "bin" / "python"
TRAIN = REPO / "scripts" / "train.py"
REPORT = REPO / "docs" / "confirm_dcl.json"

BACKBONE = ["--model", "mmim", "--dataset", "mosei", "--unaligned",
            "--model-arg", "contrast=False"]
SEEDS = [str(s) for s in range(120, 140)]
LAMBDA = "0.5"
FDR_Q = 0.10
#: The lead under test, from docs/screen_on_mmim.json on seeds 100-109.
PRIOR = {"mae": 0.0100, "corr": 0.0103}

ARMS = {
    "dclconfirm_mosei_control": [],
    "dclconfirm_mosei_dcl": ["--contrastive", "dcl", "--contrastive-lambda", LAMBDA],
}


def run_one(arm: str, epochs: int, force: bool) -> None:
    directory = OUTPUTS / arm
    wanted = SEEDS if force else [
        s for s in SEEDS if not (directory / f"seed{s}" / "result.json").exists()
    ]
    if not wanted:
        print(f"  {arm}: all {len(SEEDS)} seeds on disk, skipping")
        return
    if len(wanted) < len(SEEDS):
        print(f"  {arm}: resuming, {len(wanted)} seed(s) left")
    command = [str(PYTHON), str(TRAIN), *BACKBONE, "--epochs", str(epochs),
               *ARMS[arm], "--seeds", *wanted, "--run-group", arm, "--quiet"]
    print(f"  {arm}: {' '.join(ARMS[arm]) or 'plain (contrast off)'}")
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode})")
        print("\n".join(result.stdout.strip().splitlines()[-8:]))
        print(result.stderr.strip()[-800:])
        return
    if len(wanted) < len(SEEDS):
        from train_parallel import rebuild_summary

        rebuild_summary(directory)
        print(f"    rebuilt summary.json over all {len(SEEDS)} seeds")


def report() -> int:
    from scipy import stats

    control = read_group(OUTPUTS / "dclconfirm_mosei_control", len(SEEDS))
    subject = read_group(OUTPUTS / "dclconfirm_mosei_dcl", len(SEEDS))
    if control is None or subject is None:
        print("Both arms must be complete. Run `confirm_dcl.py run`.")
        return 1

    outcomes = {}
    for metric in METRICS:
        outcome = welch_one_sided(
            subject["valid"][metric]["per_seed"],
            control["valid"][metric]["per_seed"], metric,
        )
        _, p_two = stats.ttest_ind(
            subject["valid"][metric]["per_seed"],
            control["valid"][metric]["per_seed"], equal_var=False,
        )
        outcome["p_two_sided"] = float(p_two)
        outcome["mde"] = minimum_detectable_effect(outcome["pooled_sd"], len(SEEDS))
        outcome["prior_estimate"] = PRIOR[metric]
        outcomes[metric] = outcome

    for metric, ok in zip(
        METRICS,
        benjamini_hochberg([outcomes[m]["p_one_sided"] for m in METRICS], FDR_Q),
        strict=True,
    ):
        outcomes[metric]["bh_helps"] = bool(ok)

    print(f"backbone: {' '.join(BACKBONE)}   lambda {LAMBDA}")
    print(f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)}), disjoint from 100-119")
    print(f"criterion fixed before these runs: Welch, BH q={FDR_Q}, "
          f"{len(METRICS)} tests\n")
    for label, data in (("control (contrast off)", control), ("+ dcl", subject)):
        print(f"  {label:24s}"
              f"mae {data['valid']['mae']['mean']:.4f} ± {data['valid']['mae']['sd']:.4f}"
              f"   corr {data['valid']['corr']['mean']:.4f} ± "
              f"{data['valid']['corr']['sd']:.4f}")

    print(f"\n{'metric':8s}{'Δ (dcl - control)':>19s}{'p1':>8s}{'p2':>10s}"
          f"{'d':>7s}{'MDE':>9s}{'prior Δ':>10s}   BH")
    for metric in METRICS:
        o = outcomes[metric]
        print(f"{metric:8s}{o['improvement']:+19.4f}{o['p_one_sided']:8.3f}"
              f"{o['p_two_sided']:10.2e}{o['cohens_d']:+7.2f}{o['mde']:9.4f}"
              f"{o['prior_estimate']:+10.4f}   "
              f"{'yes' if o['bh_helps'] else 'no'}")

    confirmed = [m for m in METRICS if outcomes[m]["bh_helps"]]
    print()
    if confirmed:
        print(f"CONFIRMED on {', '.join(confirmed)}. DCL improves MMIM on MOSEI on a")
        print("second independent seed set, so it is the first candidate this phase")
        print("has earned the right to carry into the combination step. Still bound")
        print(f"to MMIM, MOSEI and lambda {LAMBDA}.")
    else:
        print("NOT CONFIRMED. The screen's lead does not replicate on seeds "
              f"{SEEDS[0]}-{SEEDS[-1]};")
        print("it is an unreplicated observation and DCL is not carried forward.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "backbone": BACKBONE, "lambda": float(LAMBDA),
        "seeds": [int(s) for s in SEEDS],
        "criterion": {"correction": f"Benjamini-Hochberg q={FDR_Q}",
                      "n_tests": len(METRICS)},
        "arms": {"control": control, "dcl": subject},
        "outcomes": outcomes, "confirmed_on": confirmed,
    }, indent=2, default=float) + "\n")
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
    print(f"== 2 arms x {len(SEEDS)} seeds on MMIM/MOSEI ==")
    for arm in ARMS:
        run_one(arm, args.epochs, args.force)
    print("\nDone. `confirm_dcl.py report` for the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
