"""Phase 2, step 9: does label_dcl's edge over dcl hold on a third seed batch?

`label_dcl` beat no contrastive term convincingly on seeds 120-139 -- +0.0100
MAE and +0.0097 Corr, both above that design's detection floor, d of 1.19 and
0.87. That claim is established. What is not is the increment over `dcl`, the
term it degenerates to when every weight is 1: +0.0044 MAE and +0.0092 Corr,
which passed BH while sitting *below* the floor the design can resolve. Sampling
helped, and those two numbers are probably inflated.

That increment is the whole mechanism. The two losses differ in exactly one
thing -- whether a negative's weight follows its label distance -- and
`check_losses.py` asserts the degeneracy numerically. So this is the experiment
that decides whether the designed loss is a contribution or a relabelling of
dcl.

This project has been wrong in this exact position twice: dcl's own confirmation
came in at half the lead's size, and the four-term combination's +0.0131 at
p=0.045 flipped sign on a third batch. Hence a third batch here.

    ldclconfirm_mosei_dcl        MMIM(contrast=False) + dcl
    ldclconfirm_mosei_label_dcl  MMIM(contrast=False) + label_dcl

Seeds 140-159, both arms rerun so they share a batch. Criterion fixed before the
runs: validation MAE and Corr, Welch one-sided, BH at q=0.10 over the two tests.

The plain no-contrastive arm is **not** rerun, to keep this at the two arms the
question needs. Its mean has been 0.5445 on seeds 100-119 and 0.5433 on 120-139,
a range of 0.0012, so it is the best-characterised quantity here -- but any
comparison against it from this batch is across seed batches and is printed as
context, never as part of the verdict.

    scripts/confirm_label_dcl.py run
    scripts/confirm_label_dcl.py report
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
REPORT = REPO / "docs" / "confirm_label_dcl.json"

BACKBONE = ["--model", "mmim", "--dataset", "mosei", "--unaligned",
            "--model-arg", "contrast=False"]
SEEDS = [str(s) for s in range(140, 160)]
LAMBDA = "0.5"
FDR_Q = 0.10
#: The lead under test, from docs/new_loss.json on seeds 120-139.
PRIOR = {"mae": 0.0044, "corr": 0.0092}
#: Earlier no-contrastive arms, for context only -- different seed batches.
CONTEXT_CONTROLS = ("mmim_diag_mosei_unaligned_off", "dclconfirm_mosei_control")

ARMS = {
    "ldclconfirm_mosei_dcl": ["--contrastive", "dcl"],
    "ldclconfirm_mosei_label_dcl": ["--contrastive", "label_dcl",
                                    "--contrastive-arg", "distance_scale=1.0"],
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
               *ARMS[arm], "--contrastive-lambda", LAMBDA,
               "--seeds", *wanted, "--run-group", arm, "--quiet"]
    print(f"  {arm}: {' '.join(ARMS[arm])} at lambda {LAMBDA}")
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
    plain = read_group(OUTPUTS / "ldclconfirm_mosei_dcl", len(SEEDS))
    designed = read_group(OUTPUTS / "ldclconfirm_mosei_label_dcl", len(SEEDS))
    if plain is None or designed is None:
        print("Both arms must be complete. Run `confirm_label_dcl.py run`.")
        return 1

    outcomes = {}
    for metric in METRICS:
        outcome = welch_one_sided(
            designed["valid"][metric]["per_seed"],
            plain["valid"][metric]["per_seed"], metric,
        )
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
    print(f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)}), disjoint from 100-139")
    print(f"criterion fixed before these runs: Welch one-sided, BH q={FDR_Q}, "
          f"{len(METRICS)} tests\n")
    for label, data in (("dcl", plain), ("label_dcl", designed)):
        print(f"  {label:12s}mae {data['valid']['mae']['mean']:.4f} ± "
              f"{data['valid']['mae']['sd']:.4f}   corr "
              f"{data['valid']['corr']['mean']:.4f} ± {data['valid']['corr']['sd']:.4f}")

    print(f"\n{'metric':8s}{'Δ (label_dcl - dcl)':>21s}{'p1':>8s}{'d':>7s}{'MDE':>9s}"
          f"{'prior Δ':>10s}   BH")
    for metric in METRICS:
        o = outcomes[metric]
        print(f"{metric:8s}{o['improvement']:+21.4f}{o['p_one_sided']:8.3f}"
              f"{o['cohens_d']:+7.2f}{o['mde']:9.4f}{o['prior_estimate']:+10.4f}   "
              f"{'yes' if o['bh_helps'] else 'no'}")

    # Context only: the no-contrastive arms live on other seed batches.
    print("\ncontext (across seed batches, not part of the verdict) — "
          "no-contrastive arms:")
    for name in CONTEXT_CONTROLS:
        data = read_group(OUTPUTS / name, 20)
        if data is not None:
            print(f"  {name:34s}mae {data['valid']['mae']['mean']:.4f}")
    print(f"  {'this batch: label_dcl':34s}mae "
          f"{designed['valid']['mae']['mean']:.4f}   dcl "
          f"{plain['valid']['mae']['mean']:.4f}")

    confirmed = [m for m in METRICS if outcomes[m]["bh_helps"]]
    print()
    if confirmed:
        print(f"CONFIRMED on {', '.join(confirmed)}. The label weighting beats the")
        print("plain decoupled form on an independent seed batch, and the two differ")
        print("in exactly one mechanism -- so the designed loss is a contribution,")
        print(f"bound to MMIM, MOSEI and lambda {LAMBDA}.")
    else:
        print("NOT CONFIRMED. The edge over dcl does not replicate on seeds "
              f"{SEEDS[0]}-{SEEDS[-1]}.")
        print("What survives is the established claim -- label_dcl beats no")
        print("contrastive term -- with the mechanism's own contribution recorded as")
        print("an unreplicated observation.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "backbone": BACKBONE, "lambda": float(LAMBDA),
        "seeds": [int(s) for s in SEEDS],
        "criterion": {"correction": f"Benjamini-Hochberg q={FDR_Q}",
                      "n_tests": len(METRICS)},
        "arms": {"dcl": plain, "label_dcl": designed},
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
    print(f"== {len(ARMS)} arms x {len(SEEDS)} seeds ==")
    for arm in ARMS:
        run_one(arm, args.epochs, args.force)
    print("\nDone. `confirm_label_dcl.py report` for the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
