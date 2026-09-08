"""Phase 2, step 7: does splitting the budget across two mechanisms beat one?

This is the deliverable the phase asks for -- a composite loss -- and it is
tested against both of the things it has to beat:

    A  MMIM(contrast=False)                     no contrastive term at all
    B  MMIM(contrast=False) + dcl               the one confirmed single term
    C  MMIM(contrast=False) + dcl + vicreg      the composite

A and B already exist on seeds 120-139 from confirm_dcl, so only C runs.

**Total lambda stays 0.5, split equally.** Arm C carries the same total
contrastive contribution as arm B, half to each term, so the comparison asks
whether spreading one budget over two mechanisms helps -- not whether more
contrastive weight helps, which is a different question that would confound this
one.

**vicreg rather than infonce, for mechanism complementarity.** dcl is
cross-modal contrast with negatives and decoupled positives; vicreg is
negative-free with an explicit anti-collapse variance term. infonce sits in the
same family as dcl (and as MMIM's own CPC), so pairing them would test one idea
twice. vicreg's own screen lead was only +0.0014 and confirming it individually
would need about 600 seeds -- but a composite does not require each term to be
individually significant, which is the whole reason to test the composite
directly.

Criterion, fixed before these runs: validation MAE and Corr, Welch one-sided for
improvement, BH at q=0.10 over the four tests (two comparisons x two metrics).
C-vs-A says whether the composite works at all; C-vs-B says whether the second
mechanism earns its place. At n=20 the detectable effect is about 0.008 MAE, so
C-vs-A is answerable if the composite matches or beats dcl's confirmed 0.0056
plus a little, while a small increment over dcl alone will not be resolvable --
recorded here so a null on C-vs-B is not read as "the second term is useless".

    scripts/test_composite.py run
    scripts/test_composite.py report
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
REPORT = REPO / "docs" / "composite.json"

BACKBONE = ["--model", "mmim", "--dataset", "mosei", "--unaligned",
            "--model-arg", "contrast=False"]
SEEDS = [str(s) for s in range(120, 140)]
LAMBDA = "0.5"
TERMS = ["dcl", "vicreg"]
FDR_Q = 0.10

ARM_A = "dclconfirm_mosei_control"      # no contrastive term
ARM_B = "dclconfirm_mosei_dcl"          # dcl alone, lambda 0.5
ARM_C = "composite_mosei_dcl_vicreg"    # dcl + vicreg, lambda 0.5 total


def run_composite(epochs: int, force: bool) -> None:
    directory = OUTPUTS / ARM_C
    wanted = SEEDS if force else [
        s for s in SEEDS if not (directory / f"seed{s}" / "result.json").exists()
    ]
    if not wanted:
        print(f"  {ARM_C}: all {len(SEEDS)} seeds on disk, skipping")
        return
    if len(wanted) < len(SEEDS):
        print(f"  {ARM_C}: resuming, {len(wanted)} seed(s) left")
    command = [str(PYTHON), str(TRAIN), *BACKBONE, "--epochs", str(epochs),
               "--contrastive", *TERMS, "--contrastive-lambda", LAMBDA,
               "--weight-scheme", "equal",
               "--seeds", *wanted, "--run-group", ARM_C, "--quiet"]
    print(f"  {ARM_C}: {' + '.join(TERMS)} at total lambda {LAMBDA}")
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
    arms = {name: read_group(OUTPUTS / name, len(SEEDS))
            for name in (ARM_A, ARM_B, ARM_C)}
    missing = [k for k, v in arms.items() if v is None]
    if missing:
        print(f"Incomplete: {', '.join(missing)}. Run `test_composite.py run`.")
        return 1

    comparisons = (("C vs A", ARM_C, ARM_A, "composite beats no contrastive term"),
                   ("C vs B", ARM_C, ARM_B, "second mechanism earns its place"))
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

    print(f"backbone: {' '.join(BACKBONE)}   total lambda {LAMBDA}, split equally")
    print(f"seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})   "
          f"criterion fixed before these runs: BH q={FDR_Q} over {len(tests)} tests\n")
    for name, label in ((ARM_A, "A  no contrastive"), (ARM_B, "B  dcl"),
                        (ARM_C, "C  dcl + vicreg")):
        data = arms[name]
        print(f"  {label:20s}mae {data['valid']['mae']['mean']:.4f} ± "
              f"{data['valid']['mae']['sd']:.4f}   corr "
              f"{data['valid']['corr']['mean']:.4f} ± {data['valid']['corr']['sd']:.4f}")

    header = (f"\n{'comparison':12s}{'metric':7s}{'Δ':>9s}{'p1':>8s}{'d':>7s}"
              f"{'MDE':>9s}   BH   asks")
    print(header)
    for label, _, _, question in comparisons:
        for metric in METRICS:
            o = outcomes[(label, metric)]
            print(f"{label:12s}{metric:7s}{o['improvement']:+9.4f}"
                  f"{o['p_one_sided']:8.3f}{o['cohens_d']:+7.2f}{o['mde']:9.4f}   "
                  f"{'yes' if o['bh_helps'] else 'no':4s} "
                  f"{question if metric == METRICS[0] else ''}")

    beats_none = [m for m in METRICS if outcomes[("C vs A", m)]["bh_helps"]]
    beats_dcl = [m for m in METRICS if outcomes[("C vs B", m)]["bh_helps"]]
    print()
    if beats_dcl:
        print(f"The composite beats dcl alone on {', '.join(beats_dcl)}: two "
              f"mechanisms at one budget\nbeat one. That is the phase's "
              f"deliverable working.")
    elif beats_none:
        print(f"The composite beats no-contrastive on {', '.join(beats_none)} but "
              f"not dcl alone.\nAt n=20 an increment over dcl below about 0.008 is "
              f"not resolvable, so this is\nnot evidence the second term is "
              f"useless -- it is the absence of evidence either way.")
    else:
        print("The composite beats neither. Since dcl alone did beat A on this "
              "same seed\nbatch, splitting the budget across two mechanisms cost "
              "something rather than\nadding -- which is itself a finding about "
              "composite design.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "backbone": BACKBONE, "lambda_total": float(LAMBDA), "terms": TERMS,
        "weight_scheme": "equal", "seeds": [int(s) for s in SEEDS],
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
    run_composite(args.epochs, args.force)
    print("\nDone. `test_composite.py report` for the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
