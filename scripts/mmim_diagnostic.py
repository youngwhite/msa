"""Phase 2, step 4: can this protocol detect a contrastive loss that works?

Every negative result phase 2 has produced on TFN carries the same unexcluded
alternative: maybe the protocol and dataset cannot resolve effects of this size,
in which case "no benefit" should read "cannot tell". This answers that, and it
has to be answered before any of those negatives means anything.

MMIM (Han et al., EMNLP 2021) is the reference point available. Its InfoNCE
(CPC) and mutual-information terms are supported by the original paper's own
ablation, it is already ported and accepted in this repository, and its
`contrast` flag turns all of them off at once. So:

    mmim_contrast_on    MMIM as published and as committed here
    mmim_contrast_off   contrast=False, leaving its L1 term alone

Seeds 100-119, everything else the repo default for MMIM. Criterion fixed before
these runs: validation MAE and Corr, Welch one-sided, BH at q=0.10 over two
tests.

How to read either outcome -- also fixed in advance, because this is the kind of
result it is tempting to reinterpret afterwards:

* **Detected** -- the protocol can see effects of this class, so the negative
  results on TFN are about those fourteen candidates, not about the setup.
* **Not detected** -- the protocol cannot resolve this magnitude on MOSI, and
  every phase-2 conclusion here becomes "inconclusive" rather than "no effect".
  The next step is then the dataset (MOSEI), not another loss.

Note it does *not* say whether our four terms would help on MMIM. That is a
different experiment, and MMIM wrapped would carry two contrastive objectives.

    scripts/mmim_diagnostic.py run
    scripts/mmim_diagnostic.py report
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
REPORT = REPO / "docs" / "mmim_diagnostic.json"

SEEDS = [str(s) for s in range(100, 120)]
FDR_Q = 0.10
ARMS = {
    "mmim_contrast_on": ["--model", "mmim"],
    "mmim_contrast_off": ["--model", "mmim", "--model-arg", "contrast=False"],
}
CONTROL_ARM = "mmim_contrast_off"


def run_one(arm: str, force: bool) -> None:
    directory = OUTPUTS / arm
    if directory.exists() and not force:
        present = sorted(p.name for p in directory.glob("seed*"))
        if len(present) == len(SEEDS):
            print(f"  {arm}: already on disk, skipping")
            return
        print(f"  {arm}: incomplete ({len(present)}/{len(SEEDS)}), redoing")
    command = [str(PYTHON), str(TRAIN), *ARMS[arm], "--seeds", *SEEDS,
               "--run-group", arm, "--quiet"]
    print(f"  {arm}: {' '.join(ARMS[arm])}")
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode})")
        print("\n".join(result.stdout.strip().splitlines()[-8:]))
        print(result.stderr.strip()[-800:])


def report() -> int:
    arms = {name: read_group(OUTPUTS / name, len(SEEDS)) for name in ARMS}
    if any(v is None for v in arms.values()):
        missing = [k for k, v in arms.items() if v is None]
        print(f"Incomplete: {', '.join(missing)}. Run `mmim_diagnostic.py run`.")
        return 1

    control = arms[CONTROL_ARM]
    subject = arms["mmim_contrast_on"]
    outcomes = {
        metric: welch_one_sided(subject["valid"][metric]["per_seed"],
                                control["valid"][metric]["per_seed"], metric)
        for metric in METRICS
    }
    rejected = benjamini_hochberg(
        [outcomes[m]["p_one_sided"] for m in METRICS], FDR_Q
    )
    for metric, is_rejected in zip(METRICS, rejected, strict=True):
        outcomes[metric]["bh_significant"] = bool(is_rejected)

    print(f"MMIM, seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})")
    print(f"criterion fixed before these runs: Welch one-sided, BH q={FDR_Q}, "
          f"{len(METRICS)} tests\n")
    for name in ("mmim_contrast_off", "mmim_contrast_on"):
        data = arms[name]
        print(f"  {name:20s}"
              f"mae {data['valid']['mae']['mean']:.4f} ± {data['valid']['mae']['sd']:.4f}"
              f"   corr {data['valid']['corr']['mean']:.4f} ± "
              f"{data['valid']['corr']['sd']:.4f}")

    print(f"\n{'metric':8s}{'Δ (on - off)':>15s}{'p':>8s}{'d':>7s}{'MDE':>9s}   BH")
    for metric in METRICS:
        o = outcomes[metric]
        mde = minimum_detectable_effect(o["pooled_sd"], len(SEEDS))
        o["mde"] = mde
        print(f"{metric:8s}{o['improvement']:+15.4f}{o['p_one_sided']:8.4f}"
              f"{o['cohens_d']:+7.2f}{mde:9.4f}   "
              f"{'yes' if o['bh_significant'] else 'no'}")

    detected = [m for m in METRICS if outcomes[m]["bh_significant"]]
    print()
    if detected:
        print(f"DETECTED on {', '.join(detected)}. This protocol can resolve a")
        print("contrastive effect of this class, so phase 2's negative results on TFN")
        print("are about those candidates rather than about the setup.")
    else:
        print("NOT DETECTED. A contrastive objective with the original paper's own")
        print("ablation behind it is invisible under this protocol on MOSI. Every")
        print("phase-2 conclusion here therefore reads 'inconclusive', not 'no effect',")
        print("and the next step is the dataset, not another loss.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "seeds": [int(s) for s in SEEDS], "control_arm": CONTROL_ARM,
        "criterion": {"test": "Welch one-sided, unpaired",
                      "correction": f"Benjamini-Hochberg q={FDR_Q}",
                      "n_tests": len(METRICS)},
        "arms": arms, "outcomes": outcomes, "detected_on": detected,
    }, indent=2, default=float) + "\n")
    print(f"\nwrote {REPORT.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "report"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.command == "report":
        return report()
    print(f"== {len(ARMS)} arms x {len(SEEDS)} seeds "
          f"(~300s each, so about {len(ARMS) * len(SEEDS) * 300 / 3600:.1f}h) ==")
    for arm in ARMS:
        run_one(arm, args.force)
    print("\nDone. `mmim_diagnostic.py report` for the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
