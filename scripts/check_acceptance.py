"""Did a reproduction meet the acceptance criterion?

The rule (docs/roadmap.md, fixed 2026-07-26 and not adjusted per result):

    Run 10 seeds. Compare each metric's mean against MMSA's reported value,
    measuring the shortfall in standard errors of that mean (SE = sigma/sqrt(n)):

        <= 1 SE   reproduced
        1-2 SE    reproduced, but flagged for a look
        >  2 SE   not reproduced — go find the implementation difference

    "Not much worse" needs a number, and the spread of our own seeds is the only
    honest yardstick available: MMSA reports single values with no variance.

MAE is lower-is-better, every other metric is higher-is-better. The primary
metrics are MAE and Acc-2(non0); the rest are reported for context.

The reference is MMSA's table, not the original papers: those used CMU-SDK GloVe
text features and in several cases different splits, so their numbers describe a
different setup.

When a metric fails, the remedy is to find the implementation difference — never
to change the seed set or tune hyper-parameters.

Usage:
    python scripts/check_acceptance.py tfn_mosi --model tfn
    python scripts/check_acceptance.py --all
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from msa.config import OUTPUT_ROOT, PROJECT_ROOT

REFERENCE_PATH = PROJECT_ROOT / "docs" / "mmsa_reference_mosi.json"
ADJUDICATION_PATH = PROJECT_ROOT / "docs" / "acceptance_status.json"
LOWER_IS_BETTER = {"mae"}
PASS_SE = 1.0    # within this many standard errors: reproduced
WARN_SE = 2.0    # beyond this: not reproduced
PRIMARY = ("mae", "acc2_non0")
REQUIRED_SEEDS = 10


def load_reference() -> dict:
    return json.loads(REFERENCE_PATH.read_text())["models"]


def load_adjudications() -> dict:
    """Gaps already investigated and attributed, so they are not re-litigated."""
    if not ADJUDICATION_PATH.exists():
        return {}
    return json.loads(ADJUDICATION_PATH.read_text()).get("groups", {})


def shortfall(mean: float, reference: float, metric: str) -> float:
    """How much worse than the reference, in metric units. Negative means better."""
    return mean - reference if metric in LOWER_IS_BETTER else reference - mean


def judge(group: str, model: str, reference: dict, verbose: bool = True) -> bool:
    summary_path = OUTPUT_ROOT / group / "summary.json"
    if not summary_path.exists():
        print(f"{group}: no results at {summary_path}")
        return False
    summary = json.loads(summary_path.read_text())
    stats = summary["summary"]
    if model not in reference:
        print(f"{group}: no MMSA reference for model {model!r}; "
              f"known: {sorted(reference)}")
        return False
    ref = reference[model]

    n = stats["mae"]["n"]
    setting = "unaligned" if not summary.get("aligned", True) else "aligned"
    print(f"\n=== {group} ({model}, {setting}, {n} seeds) ===")
    if ref.get("data_setting") not in (None, setting):
        print(f"  ! MMSA reports {model} on {ref['data_setting']} data, ours is "
              f"{setting} — not a like-for-like comparison")

    print(f"  {'metric':12s}{'ours':>10s}{'+-sd':>9s}{'MMSA':>10s}"
          f"{'shortfall':>11s}{'in SE':>8s}  verdict")
    failures, flagged = [], []
    for metric in ("mae", "acc2_non0", "acc2_has0", "f1_non0", "acc7", "acc5", "corr"):
        if metric not in stats or metric not in ref:
            continue
        mean, sd = stats[metric]["mean"], stats[metric]["std"]
        se = sd / math.sqrt(n) if n > 1 else float("inf")
        gap = shortfall(mean, ref[metric], metric)
        in_se = gap / se if se > 0 else 0.0
        if in_se <= PASS_SE:
            tag = "pass"
        elif in_se <= WARN_SE:
            tag = "look"
            flagged.append(metric)
        else:
            tag = "FAIL"
            if metric in PRIMARY:
                failures.append(metric)
        marker = " (primary)" if metric in PRIMARY else ""
        print(f"  {metric:12s}{mean:10.4f}{sd:9.4f}{ref[metric]:10.4f}"
              f"{gap:+11.4f}{in_se:+8.1f}  {tag}{marker}")

    if n < REQUIRED_SEEDS:
        print(f"  ! only {n} seeds; the criterion asks for {REQUIRED_SEEDS} "
              f"(seeds 42-51). This verdict is provisional.")
    if failures:
        verdict = load_adjudications().get(group)
        if verdict and verdict.get("status") == "gap_explained":
            print(f"  VERDICT: {', '.join(failures)} short by more than "
                  f"{WARN_SE:.0f} SE, but this gap is ADJUDICATED "
                  f"({verdict['date']}):")
            print(f"    {verdict['attribution']}")
            print(f"    evidence: {verdict['evidence']}")
            print(f"    do not retry: {verdict['do_not_retry']}")
            return True
        print(f"  VERDICT: not reproduced — {', '.join(failures)} short by more than "
              f"{WARN_SE:.0f} standard errors.")
        print("  Next step is to find the implementation difference: compare against "
              "MMSA's config and the original paper's hyper-parameters. Changing the "
              "seed set or tuning to close the gap is forbidden.")
        return False
    if flagged:
        print(f"  VERDICT: reproduced, but {', '.join(flagged)} sits 1-2 SE low — "
              f"worth a look, not a blocker.")
    else:
        print("  VERDICT: reproduced")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("group", nargs="?", help="run group under outputs/")
    ap.add_argument("--model", help="MMSA model name; defaults to the group's model")
    ap.add_argument("--all", action="store_true",
                    help="judge every group whose model has a reference")
    args = ap.parse_args()

    reference = load_reference()
    if args.all:
        results = []
        for summary_path in sorted(OUTPUT_ROOT.glob("*/summary.json")):
            model = json.loads(summary_path.read_text()).get("model", "")
            if model in reference:
                results.append(judge(summary_path.parent.name, model, reference))
        if not results:
            print("no group matches a model in the reference table")
            return
        sys.exit(0 if all(results) else 1)

    if not args.group:
        ap.error("give a run group, or --all")
    model = args.model or json.loads(
        (OUTPUT_ROOT / args.group / "summary.json").read_text()
    ).get("model", "")
    sys.exit(0 if judge(args.group, model, reference) else 1)


if __name__ == "__main__":
    main()
