"""Did a reproduction meet the acceptance criterion?

The rule (docs/roadmap.md, fixed 2026-07-26 and not adjusted per result):

    Run 10 seeds. Compare each metric's mean against MMSA's reported value,
    measuring the shortfall in standard errors of that mean (SE = sigma/sqrt(n)):

        <= 1 SE   reproduced
        1-2 SE    reproduced, but flagged for a look
        >  2 SE   not reproduced — go find the implementation difference

    Since 2026-07-30 there is a floor under that last line: a shortfall smaller
    than this dataset's own seed noise (NOISE_FLOOR, measured on the LF-LSTM
    baseline and registered on day one) is not counted as a failure however many
    SE it spans. Our per-model sigma varies 29-fold, so without a floor the same
    absolute discrepancy meant +0.2 SE on the least stable model and +4.8 SE on
    the most stable — the criterion was strictest exactly where the port was
    best. It does NOT fix the opposite asymmetry; that case is reported loudly.

    "Not much worse" needs a number, and the spread of our own seeds is the only
    honest yardstick available: MMSA reports single values with no variance.

MAE is lower-is-better, every other metric is higher-is-better. The primary
metrics are **MAE and Corr**; everything else is reported for context but does
not decide a verdict.

Acc-2 was primary until 2026-07-28 and is not any more. It is quantised by the
sample count — one test sample is 0.152 points on the non-zero split (656
samples), 0.146 on the full one — while the differences under test are 0.6 to
1.3 points, i.e. four to eight samples. It also throws away magnitude entirely,
and the sign of a sample with |label| < 0.5 is close to arbitrary (62% error
across every model we have). Acc-5 and Acc-7 are quantised the same way.

MAE and Corr are the only two continuous metrics, and they fail differently:
MAE catches calibration, Corr catches ranking. Corr alone would not do — it is
invariant to scale and shift, so a model predicting ten times the right answer
scores the same. Read them together. See docs/decisions.md, 2026-07-28.

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
#: This dataset's own seed noise, measured on the LF-LSTM baseline over seeds
#: 42-46 and registered in docs/roadmap.md's opening argument long before any
#: acceptance run existed. A shortfall smaller than this is smaller than the
#: spread a rerun would produce, so it cannot be evidence of a failed port.
#: Used as a floor under the SE test, not as a replacement for it — see
#: docs/decisions.md, 2026-07-30, including what this rule does NOT fix.
NOISE_FLOOR = {
    "mae": 0.0387, "corr": 0.0110, "acc2_non0": 0.0124, "acc2_has0": 0.0101,
    "f1_non0": 0.0122, "acc7": 0.0262, "acc5": 0.0299,
}

PASS_SE = 1.0    # within this many standard errors: reproduced
WARN_SE = 2.0    # beyond this: not reproduced
PRIMARY = ("mae", "corr")   # see docs/decisions.md 2026-07-28: Acc-2 is quantised at 1/656
#: Below this, a run has not learned anything: MOSI's majority class is ~58% of
#: the non-zero test samples, so an Acc-2 near that is a collapsed run.
COLLAPSE_ACC2 = 0.60
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
    failures, flagged, under_floor, masked = [], [], [], []
    for metric in ("mae", "acc2_non0", "acc2_has0", "f1_non0", "acc7", "acc5", "corr"):
        if metric not in stats or metric not in ref:
            continue
        mean, sd = stats[metric]["mean"], stats[metric]["std"]
        se = sd / math.sqrt(n) if n > 1 else float("inf")
        gap = shortfall(mean, ref[metric], metric)
        in_se = gap / se if se > 0 else 0.0
        floor = NOISE_FLOOR.get(metric)
        below_floor = floor is not None and gap <= floor
        if in_se <= PASS_SE:
            tag = "pass"
        elif in_se <= WARN_SE:
            tag = "look"
            flagged.append(metric)
        elif below_floor:
            # More than 2 SE behind, but by less than this dataset's own seed
            # noise. Calling that a reproduction failure would demand precision
            # neither side's numbers carry. See docs/decisions.md 2026-07-30.
            tag = "pass*"
            under_floor.append(metric)
        else:
            tag = "FAIL"
            if metric in PRIMARY:
                failures.append(metric)
        # The opposite asymmetry, which the floor rule does NOT fix: a gap well
        # above the noise floor that stays under 2 SE only because our own seed
        # spread is wide. Reported loudly rather than silently tolerated.
        if floor is not None and gap > floor and in_se <= WARN_SE:
            masked.append((metric, gap, floor, in_se))
        marker = " (primary)" if metric in PRIMARY else ""
        print(f"  {metric:12s}{mean:10.4f}{sd:9.4f}{ref[metric]:10.4f}"
              f"{gap:+11.4f}{in_se:+8.1f}  {tag}{marker}")

    # A collapsed seed inflates the spread, which inflates SE, which *widens* the
    # tolerance — so instability can buy a pass. Report it; never drop the seed,
    # since selecting seeds is exactly what the criterion forbids.
    collapsed = [
        seed for seed, value in stats["acc2_non0"]["per_seed"].items()
        if value < COLLAPSE_ACC2
    ]
    if collapsed:
        print(f"  ! {len(collapsed)}/{n} seed(s) collapsed to chance "
              f"(Acc-2 < {COLLAPSE_ACC2}): {', '.join(collapsed)}")
        print("    The mean and the standard error both reflect those runs. A wide "
              "spread makes this test weaker, not the model better.")
    for metric, gap, floor, in_se in masked:
        print(f"  ! {metric} is {gap:+.4f} behind, {gap/floor:.1f}x this dataset's "
              f"noise floor ({floor:.4f}), yet only {in_se:+.1f} SE — our own seed "
              f"spread is wide enough to hide it.")
        print("    The noise-floor rule does not catch this direction. Treat it as "
              "a gap to investigate, not as a pass.")
    if under_floor:
        print(f"  * {', '.join(under_floor)} sit(s) beyond 2 SE but within this "
              f"dataset's seed noise, so not counted as a failure "
              f"(docs/decisions.md 2026-07-30).")
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
            payload = json.loads(summary_path.read_text())
            model, group = payload.get("model", ""), summary_path.parent.name
            # Acceptance groups are named <model>_<dataset>. Ablations and
            # diagnostics carry a suffix and are not judged against MMSA — they
            # deliberately deviate from the reference.
            dataset = payload.get("dataset", "").lower().replace("cmu-", "")
            if model in reference and group == f"{model}_{dataset}":
                results.append(judge(group, model, reference))
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
