"""Phase 2, step 5: eliminate candidates on a test bed that is known to react.

Elimination needs a setup that can tell "this loss does nothing" from "this loss
had no chance to act here". TFN could not: on MOSEI at lambda 0.1 all fourteen
candidates landed inside +-0.0035 with BH rejecting 0 of 28, which eliminates
nothing (docs/investigations.md#mosei-screen-control-precision). TFN is also not
a contrastive-learning architecture -- its modality vectors go straight into an
outer product, so the space an auxiliary contrastive term shapes is not the space
the model was built around.

The one setup this project has measured a real contrastive effect in is MMIM on
MOSEI: its own CPC and mutual-information terms move validation MAE by |d|=1.39,
and `check_mmim_equivalence.py` proves that model is numerically identical to
MMSA's. So the test bed is MMIM with its own contrast switched off, and a
candidate put in the slot its terms vacated:

    control    MMIM(contrast=False)              -- 20 seeds, already run as
                                                    mmim_diag_mosei_unaligned_off
    candidate  MMIM(contrast=False) + candidate  -- 10 seeds, 100-109

Three properties at once: a contrastive-learning architecture, no second
contrastive objective competing with the one under test, and a slot demonstrated
to be sensitive.

**Four candidates, chosen by mechanism rather than by rank.** Fourteen would be
32 hours at 820s a seed. The TFN screen's ordering cannot be used to cut the list
-- it is noise, and cutting by it would repeat the mistake that round already
made. So one representative per mechanism: `infonce` for cross-modal pairing
(the family MMIM's own CPC belongs to), `dcl` for the variant that removes
InfoNCE's positive-negative coupling and is the only one with a documented reason
to work at batch 32, `vicreg` for the negative-free family and its explicit
anti-collapse variance term, and `rnc` as the only objective here defined on a
continuous target's ordering without binning it.

**lambda 0.5, calibrated rather than inherited.** Terms are rescaled to unit
magnitude, so lambda is the contrastive part's absolute contribution. MOSEI's
task L1 runs about 0.53 and MMIM's own alpha*nce about 0.43 -- roughly 80% of the
task loss. lambda 0.1 would be 19%, four times weaker than the configuration
that produced the only effect anyone has measured here, and plausibly part of why
the TFN round saw nothing at all.

Criterion, fixed before these runs: validation MAE and Corr, Welch one-sided for
improvement plus two-sided for either direction, BH at q=0.10 over the eight
tests. A candidate is eliminated when neither direction is significant -- that
is the finding this step exists to produce. Harm is reported, not filtered.

    scripts/screen_on_mmim.py run
    scripts/screen_on_mmim.py report
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
REPORT = REPO / "docs" / "screen_on_mmim.json"

BACKBONE = ["--model", "mmim", "--dataset", "mosei", "--unaligned",
            "--model-arg", "contrast=False"]
SEEDS = [str(s) for s in range(100, 110)]
#: Already on disk: MMIM(contrast=False) on MOSEI, seeds 100-119, same config.
CONTROL_GROUP = "mmim_diag_mosei_unaligned_off"
CONTROL_SEEDS = 20
CANDIDATES = ("infonce", "dcl", "vicreg", "rnc")
LAMBDA = "0.5"
FDR_Q = 0.10


def group_name(candidate: str) -> str:
    return f"mmimscreen_mosei_{candidate}"


def run_one(candidate: str, epochs: int, force: bool) -> None:
    group = group_name(candidate)
    directory = OUTPUTS / group
    wanted = SEEDS if force else [
        s for s in SEEDS if not (directory / f"seed{s}" / "result.json").exists()
    ]
    if not wanted:
        print(f"  {group}: all {len(SEEDS)} seeds on disk, skipping")
        return
    if len(wanted) < len(SEEDS):
        print(f"  {group}: resuming, {len(wanted)} seed(s) left")
    command = [str(PYTHON), str(TRAIN), *BACKBONE, "--epochs", str(epochs),
               "--contrastive", candidate, "--contrastive-lambda", LAMBDA,
               "--seeds", *wanted, "--run-group", group, "--quiet"]
    print(f"  {group}: lambda {LAMBDA}")
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

    control = read_group(OUTPUTS / CONTROL_GROUP, CONTROL_SEEDS)
    if control is None:
        print(f"No {CONTROL_GROUP} on disk with {CONTROL_SEEDS} seeds.")
        return 1

    rows, one_sided, two_sided = [], [], []
    for candidate in CANDIDATES:
        data = read_group(OUTPUTS / group_name(candidate), len(SEEDS))
        if data is None:
            continue
        entry = {"candidate": candidate, "metrics": {}}
        for metric in METRICS:
            outcome = welch_one_sided(
                data["valid"][metric]["per_seed"],
                control["valid"][metric]["per_seed"], metric,
            )
            _, p_two = stats.ttest_ind(
                data["valid"][metric]["per_seed"],
                control["valid"][metric]["per_seed"], equal_var=False,
            )
            outcome["p_two_sided"] = float(p_two)
            outcome["mde"] = minimum_detectable_effect(
                outcome["pooled_sd"], len(SEEDS)
            )
            entry["metrics"][metric] = outcome
            one_sided.append((len(rows), metric, outcome["p_one_sided"]))
            two_sided.append((len(rows), metric, float(p_two)))
        entry["data"] = data
        rows.append(entry)

    if not rows:
        print("No candidate groups on disk. Run `screen_on_mmim.py run`.")
        return 1

    for (i, m, _), ok in zip(one_sided,
                             benjamini_hochberg([p for _, _, p in one_sided], FDR_Q),
                             strict=True):
        rows[i]["metrics"][m]["bh_helps"] = bool(ok)
    for (i, m, _), ok in zip(two_sided,
                             benjamini_hochberg([p for _, _, p in two_sided], FDR_Q),
                             strict=True):
        rows[i]["metrics"][m]["bh_any"] = bool(ok)

    print(f"backbone: {' '.join(BACKBONE)}   lambda {LAMBDA}")
    print(f"control:  {CONTROL_GROUP} (n={CONTROL_SEEDS})   "
          f"candidates: seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})")
    print(f"criterion fixed before these runs: BH q={FDR_Q} over "
          f"{len(two_sided)} tests, both directions reported")
    print(f"control MAE {control['valid']['mae']['mean']:.4f} ± "
          f"{control['valid']['mae']['sd']:.4f}   Corr "
          f"{control['valid']['corr']['mean']:.4f} ± "
          f"{control['valid']['corr']['sd']:.4f}\n")

    header = (f"{'candidate':12s}{'Δmae':>9s}{'p2':>10s}{'Δcorr':>9s}{'p2':>10s}"
              f"{'MDE(mae)':>10s}   verdict")
    print(header)
    print("-" * len(header))
    survivors = []
    for entry in rows:
        info = entry["metrics"]
        helps = [m for m in METRICS if info[m]["bh_helps"]]
        any_effect = [m for m in METRICS if info[m]["bh_any"]]
        if helps:
            verdict = "KEEP: helps on " + ",".join(helps)
            survivors.append(entry["candidate"])
        elif any_effect:
            verdict = "HURTS on " + ",".join(any_effect)
        else:
            verdict = "eliminated (no effect either way)"
        print(f"{entry['candidate']:12s}"
              f"{info['mae']['improvement']:+9.4f}{info['mae']['p_two_sided']:10.2e}"
              f"{info['corr']['improvement']:+9.4f}{info['corr']['p_two_sided']:10.2e}"
              f"{info['mae']['mde']:10.4f}   {verdict}")

    print(f"\nsurvivors: {', '.join(survivors) if survivors else 'none'}")
    if not survivors:
        print("Nothing to combine from this round. A null here is stronger than the")
        print("TFN round's -- the slot is known to react, since MMIM's own terms move")
        print(f"it by |d|=1.39 -- but it is still specific to lambda {LAMBDA}.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "backbone": BACKBONE, "lambda": float(LAMBDA),
        "control_group": CONTROL_GROUP, "control_seeds": CONTROL_SEEDS,
        "candidate_seeds": [int(s) for s in SEEDS],
        "criterion": {"correction": f"Benjamini-Hochberg q={FDR_Q}",
                      "n_tests": len(two_sided),
                      "directions": "one-sided for improvement, two-sided for either"},
        "control": control, "results": rows, "survivors": survivors,
    }, indent=2, default=float) + "\n")
    print(f"\nwrote {REPORT.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "report"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--only", nargs="+", default=None, choices=CANDIDATES)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.command == "report":
        return report()
    wanted = args.only or list(CANDIDATES)
    print(f"== {len(wanted)} candidate(s) x {len(SEEDS)} seeds on MMIM/MOSEI ==")
    for candidate in wanted:
        run_one(candidate, args.epochs, args.force)
    print("\nDone. `screen_on_mmim.py report` for the table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
