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

import numpy as np

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


SEEDS = [str(s) for s in range(100, 120)]
FDR_Q = 0.10
#: The two arms, by the only thing that differs between them.
ARMS = {"on": [], "off": ["--model-arg", "contrast=False"]}
CONTROL = "off"

#: MMIM takes **unaligned** data. MMSA's config says `need_data_aligned: false`,
#: and the accepted `mmim_mosi` group in this repository was run with
#: `--unaligned`; convention 5's checklist lists aligned/unaligned by name.
#:
#: The first version of this script omitted the flag, so the whole first
#: diagnostic ran MMIM in a data setting it was never validated in. That is why
#: the setting is now part of every group name -- a mis-specified rerun would
#: land in a differently named directory instead of silently overwriting a
#: correct one. The `mmim_contrast_{on,off}` groups are that first attempt, kept
#: as the record; see docs/investigations.md#mmim-diagnostic-aligned.
SETTING_ARGS = ["--unaligned"]
SETTING = "unaligned"


def group_name(arm: str, dataset: str) -> str:
    return f"mmim_diag_{dataset}_{SETTING}_{arm}"


def report_path(dataset: str) -> Path:
    return REPO / "docs" / f"mmim_diagnostic_{dataset}_{SETTING}.json"


def run_one(arm: str, dataset: str, epochs: int, force: bool) -> None:
    directory = OUTPUTS / group_name(arm, dataset)
    if directory.exists() and not force:
        present = sorted(p.name for p in directory.glob("seed*"))
        if len(present) == len(SEEDS):
            print(f"  {arm}: already on disk, skipping")
            return
        print(f"  {arm}: incomplete ({len(present)}/{len(SEEDS)}), redoing")
    group = group_name(arm, dataset)
    command = [str(PYTHON), str(TRAIN), "--model", "mmim", "--dataset", dataset,
               *SETTING_ARGS, "--epochs", str(epochs), *ARMS[arm], "--seeds", *SEEDS,
               "--run-group", group, "--quiet"]
    print(f"  {group}: {' '.join(ARMS[arm]) or 'MMIM as published'}")
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode})")
        print("\n".join(result.stdout.strip().splitlines()[-8:]))
        print(result.stderr.strip()[-800:])


def epochs_used(arm: str, dataset: str) -> tuple[int, int, int]:
    """(median epochs, max epochs, runs that hit the cap) for one arm.

    Checked rather than assumed: an arm truncated by the epoch cap is a third
    explanation for a null result, alongside power and the flag not working. On
    MOSI this was verified by hand; here it is part of the report.
    """
    lengths, capped = [], 0
    directory = OUTPUTS / group_name(arm, dataset)
    for path in sorted(directory.glob("seed*/result.json")):
        result = json.loads(path.read_text())
        n = len(result["history"])
        lengths.append(n)
        if n >= result["cli"].get("epochs", 10**9):
            capped += 1
    if not lengths:
        return 0, 0, 0
    return int(np.median(lengths)), max(lengths), capped


def report(dataset: str) -> int:
    arms = {arm: read_group(OUTPUTS / group_name(arm, dataset), len(SEEDS))
            for arm in ARMS}
    if any(v is None for v in arms.values()):
        missing = [group_name(k, dataset) for k, v in arms.items() if v is None]
        print(f"Incomplete: {', '.join(missing)}. Run `mmim_diagnostic.py run`.")
        return 1

    for arm in ARMS:
        directory = OUTPUTS / group_name(arm, dataset)
        for path in sorted(directory.glob("seed*/result.json")):
            record = json.loads(path.read_text())
            if record["aligned"] is (SETTING == "unaligned"):
                print(f"{path.parent} was run aligned={record['aligned']}, but this "
                      f"diagnostic is {SETTING}. Delete the group and rerun.")
                return 1

    control = arms[CONTROL]
    subject = arms["on"]
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

    print(f"MMIM on {dataset} ({SETTING}), seeds {SEEDS[0]}-{SEEDS[-1]} "
          f"(n={len(SEEDS)})")
    print(f"criterion fixed before these runs: Welch one-sided, BH q={FDR_Q}, "
          f"{len(METRICS)} tests\n")
    for name in ("off", "on"):
        data = arms[name]
        print(f"  {group_name(name, dataset):26s}"
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

    print("\ntraining length (an arm cut short by the epoch cap would be a third "
          "explanation):")
    truncated = []
    for arm in ARMS:
        median, longest, capped = epochs_used(arm, dataset)
        print(f"  {group_name(arm, dataset):26s}median {median}  max {longest}  "
              f"hit the cap: {capped}/{len(SEEDS)}")
        if capped:
            truncated.append(group_name(arm, dataset))

    detected = [m for m in METRICS if outcomes[m]["bh_significant"]]
    print()
    if truncated:
        print(f"!! {', '.join(truncated)} had runs stopped by --epochs, not by early "
              f"stopping.\n   Raise --epochs and rerun before reading the verdict "
              f"below.\n")
    if detected:
        print(f"DETECTED on {', '.join(detected)}. This protocol can resolve a")
        print("contrastive effect of this class, so phase 2's negative results on TFN")
        print("are about those candidates rather than about the setup.")
    else:
        print("NOT DETECTED. A contrastive objective with the original paper's own")
        print("ablation behind it is invisible under this protocol on MOSI. Every")
        print("phase-2 conclusion here therefore reads 'inconclusive', not 'no effect',")
        print("and the next step is the dataset, not another loss.")

    out = report_path(dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": dataset,
        "seeds": [int(s) for s in SEEDS], "control_arm": group_name(CONTROL, dataset),
        "epoch_cap_hits": {group_name(a, dataset): epochs_used(a, dataset)[2]
                           for a in ARMS},
        "criterion": {"test": "Welch one-sided, unpaired",
                      "correction": f"Benjamini-Hochberg q={FDR_Q}",
                      "n_tests": len(METRICS)},
        "arms": arms, "outcomes": outcomes, "detected_on": detected,
    }, indent=2, default=float) + "\n")
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "report"))
    ap.add_argument("--dataset", default="mosi", choices=("mosi", "mosei"))
    ap.add_argument("--epochs", type=int, default=40,
                    help="cap, not a target: early stopping ends every run well "
                         "short of it on MOSI, and the report says if it ever binds")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.command == "report":
        return report(args.dataset)
    print(f"== {len(ARMS)} arms x {len(SEEDS)} seeds on {args.dataset} ==")
    for arm in ARMS:
        run_one(arm, args.dataset, args.epochs, args.force)
    print("\nDone. `mmim_diagnostic.py report` for the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
