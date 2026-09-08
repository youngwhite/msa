"""Phase 2, step 8: does weighting negatives by label distance beat not doing so?

`label_dcl` is this phase's designed loss, and its ablation is unusually clean:
with every weight at 1 it is exactly `dcl`, which is the one term already
confirmed on this setup (+0.0056 MAE, replicated). `check_losses.py` asserts
that degeneracy numerically. So the comparison against `dcl` isolates a single
mechanism -- treating "negative" as a matter of degree on a continuous target
rather than as a binary or ordinal fact.

Three arms, two already on disk from confirm_dcl at seeds 120-139:

    A  MMIM(contrast=False)                 no contrastive term
    B  MMIM(contrast=False) + dcl           the confirmed term, lambda 0.5
    D  MMIM(contrast=False) + label_dcl     the designed term, lambda 0.5

**The threshold is stated before the run, not after.** dcl beats A by +0.0056 on
this seed batch. At n=20 this design resolves about 0.0075 of MAE, so for
`label_dcl` to be judged better than `dcl` it needs roughly that much again on
top -- an effect near +0.013 against A. Anything smaller will be real or not
without this experiment being able to say, and that outcome gets recorded as
"cannot tell", not as "the mechanism does nothing".

Criterion: validation MAE and Corr, Welch one-sided, BH at q=0.10 over the four
tests (D-vs-A and D-vs-B, two metrics each). D-vs-A asks whether the designed
loss works at all; D-vs-B asks whether the label weighting earns its place.

    scripts/test_new_loss.py run
    scripts/test_new_loss.py report
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
REPORT = REPO / "docs" / "new_loss.json"

BACKBONE = ["--model", "mmim", "--dataset", "mosei", "--unaligned",
            "--model-arg", "contrast=False"]
SEEDS = [str(s) for s in range(120, 140)]
LAMBDA = "0.5"
FDR_Q = 0.10
#: d0 = 1.0 on a [-3, 3] target: pairs a sentiment point apart or more are full
#: negatives, closer ones are discounted in proportion. One hyper-parameter, set
#: once, not tuned -- tuning it against the metric would make the comparison
#: against dcl a search rather than a test.
LOSS_ARGS = ["--contrastive-arg", "distance_scale=1.0"]

ARM_A = "dclconfirm_mosei_control"
ARM_B = "dclconfirm_mosei_dcl"
ARM_D = "newloss_mosei_label_dcl"


def run_arm(epochs: int, force: bool) -> None:
    directory = OUTPUTS / ARM_D
    wanted = SEEDS if force else [
        s for s in SEEDS if not (directory / f"seed{s}" / "result.json").exists()
    ]
    if not wanted:
        print(f"  {ARM_D}: all {len(SEEDS)} seeds on disk, skipping")
        return
    if len(wanted) < len(SEEDS):
        print(f"  {ARM_D}: resuming, {len(wanted)} seed(s) left")
    command = [str(PYTHON), str(TRAIN), *BACKBONE, "--epochs", str(epochs),
               "--contrastive", "label_dcl", "--contrastive-lambda", LAMBDA,
               *LOSS_ARGS, "--seeds", *wanted, "--run-group", ARM_D, "--quiet"]
    print(f"  {ARM_D}: label_dcl at lambda {LAMBDA}, {' '.join(LOSS_ARGS)}")
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
    arms = {n: read_group(OUTPUTS / n, len(SEEDS)) for n in (ARM_A, ARM_B, ARM_D)}
    missing = [k for k, v in arms.items() if v is None]
    if missing:
        print(f"Incomplete: {', '.join(missing)}. Run `test_new_loss.py run`.")
        return 1

    comparisons = (("D vs A", ARM_D, ARM_A, "the designed loss works at all"),
                   ("D vs B", ARM_D, ARM_B, "label weighting earns its place"))
    outcomes, tests = {}, []
    for label, subject, control, _ in comparisons:
        for metric in METRICS:
            outcome = welch_one_sided(
                arms[subject]["valid"][metric]["per_seed"],
                arms[control]["valid"][metric]["per_seed"], metric,
            )
            outcome["mde"] = minimum_detectable_effect(
                outcome["pooled_sd"], len(SEEDS)
            )
            outcomes[(label, metric)] = outcome
            tests.append((label, metric, outcome["p_one_sided"]))

    for (label, metric, _), ok in zip(
        tests, benjamini_hochberg([p for _, _, p in tests], FDR_Q), strict=True
    ):
        outcomes[(label, metric)]["bh_helps"] = bool(ok)

    print(f"backbone: {' '.join(BACKBONE)}   lambda {LAMBDA}   {' '.join(LOSS_ARGS)}")
    print(f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})   "
          f"criterion fixed before these runs: BH q={FDR_Q} over {len(tests)} tests\n")
    for name, label in ((ARM_A, "A  no contrastive"), (ARM_B, "B  dcl"),
                        (ARM_D, "D  label_dcl")):
        data = arms[name]
        print(f"  {label:20s}mae {data['valid']['mae']['mean']:.4f} ± "
              f"{data['valid']['mae']['sd']:.4f}   corr "
              f"{data['valid']['corr']['mean']:.4f} ± {data['valid']['corr']['sd']:.4f}")

    print(f"\n{'comparison':12s}{'metric':7s}{'Δ':>9s}{'p1':>8s}{'d':>7s}{'MDE':>9s}"
          f"   BH   asks")
    for label, _, _, question in comparisons:
        for metric in METRICS:
            o = outcomes[(label, metric)]
            print(f"{label:12s}{metric:7s}{o['improvement']:+9.4f}"
                  f"{o['p_one_sided']:8.3f}{o['cohens_d']:+7.2f}{o['mde']:9.4f}   "
                  f"{'yes' if o['bh_helps'] else 'no':4s} "
                  f"{question if metric == METRICS[0] else ''}")

    beats_none = [m for m in METRICS if outcomes[("D vs A", m)]["bh_helps"]]
    beats_dcl = [m for m in METRICS if outcomes[("D vs B", m)]["bh_helps"]]
    print()
    if beats_dcl:
        print(f"The label weighting earns its place on {', '.join(beats_dcl)}. The "
              f"designed loss\nbeats the confirmed term it degenerates to, and the "
              f"two differ in exactly one\nmechanism -- so that mechanism is what "
              f"did it.")
    elif beats_none:
        print(f"label_dcl beats no-contrastive on {', '.join(beats_none)} but not "
              f"dcl.\nSo the designed loss works, and the label weighting is not "
              f"shown to add\nanything over the plain decoupled form at this "
              f"resolution -- 'cannot tell',\nnot 'does nothing'.")
    else:
        print("label_dcl beats neither arm. Since dcl beat A on this same seed "
              "batch, the\nlabel weighting did not merely fail to help -- it cost "
              "the decoupled form its\nconfirmed gain, which is a finding about "
              "the mechanism rather than about power.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "backbone": BACKBONE, "lambda": float(LAMBDA), "loss_args": LOSS_ARGS,
        "seeds": [int(s) for s in SEEDS],
        "criterion": {"correction": f"Benjamini-Hochberg q={FDR_Q}",
                      "n_tests": len(tests)},
        "arms": arms,
        "comparisons": [
            {"comparison": label, "metric": metric, **outcomes[(label, metric)]}
            for label, _, _, _ in comparisons for metric in METRICS
        ],
        "beats_no_contrastive_on": beats_none, "beats_dcl_on": beats_dcl,
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
    print(f"== 1 arm x {len(SEEDS)} seeds (A and B already on disk) ==")
    run_arm(args.epochs, args.force)
    print("\nDone. `test_new_loss.py report` for the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
