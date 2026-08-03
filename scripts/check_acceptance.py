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
import statistics
import sys

from msa.config import OUTPUT_ROOT, PROJECT_ROOT

REFERENCE_PATH = PROJECT_ROOT / "docs" / "mmsa_reference_mosi.json"
#: Models MMSA never published a number for. The reference there is our own run
#: of MMSA's code over the same seeds, which unlike the table carries variance —
#: see load_reference and docs/decisions.md, 2026-07-31. Stored per seed, so the
#: mean and spread are computed here with this project's ddof=1 rather than read
#: off MMSA's CSV, which summarises with np.std (ddof=0).
CODE_REFERENCE_PATH = PROJECT_ROOT / "docs" / "mmsa_code_runs_mosi.json"
#: Methods published after MMSA stopped adding models have only their authors'
#: releases as a reference. Separate file because the provenance differs: one is
#: a third-party reimplementation, the other is the authors' own code.
AUTHOR_REFERENCE_PATH = PROJECT_ROOT / "docs" / "author_code_runs_mosi.json"
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
    """MMSA's published table, plus our own MMSA runs for what it never published.

    The table wins on overlap: where MMSA states a number, that number is the
    claim under test. The code-run entries carry `<metric>_sd`, which `judge`
    uses to widen the comparison — see `standard_error`.

    Those spreads are computed here, from the per-seed values, with statistics'
    ddof=1. MMSA's own CSV would have supplied them ready-made, but it summarises
    with np.std (ddof=0) — 5% narrower at n=10 — and this project reports the
    sample standard deviation everywhere else. Adding one convention's variance
    to the other's is wrong in the direction that makes the test stricter than it
    should be.
    """
    reference = json.loads(REFERENCE_PATH.read_text())["models"]
    if CODE_REFERENCE_PATH.exists():
        for model, entry in json.loads(CODE_REFERENCE_PATH.read_text())["models"].items():
            reference.setdefault(model, {"_source": "mmsa_code"} | summarise(entry))
    if AUTHOR_REFERENCE_PATH.exists():
        for model, entry in json.loads(AUTHOR_REFERENCE_PATH.read_text())["models"].items():
            reference.setdefault(model, {"_source": "author_code"} | summarise(entry))
    return reference


def summarise(entry: dict) -> dict:
    """Per-seed reference runs -> the mean/sd/n shape the rest of this file wants."""
    runs = list(entry["runs"].values())
    # Author releases record no data_setting; the check that uses it treats
    # None as "not stated" and skips, which is the honest reading.
    stats = {"data_setting": entry.get("data_setting"), "n": len(runs),
             "seeds": entry["seeds"]}
    for metric in {key for run in runs for key in run}:
        values = [run[metric] for run in runs if metric in run]
        if len(values) != len(runs):   # a metric missing from some seeds would
            continue                   # average over a different denominator
        stats[metric] = statistics.fmean(values)
        if len(values) > 1:
            stats[f"{metric}_sd"] = statistics.stdev(values)
    return stats


def load_adjudications() -> dict:
    """Gaps already investigated and attributed, so they are not re-litigated."""
    if not ADJUDICATION_PATH.exists():
        return {}
    return json.loads(ADJUDICATION_PATH.read_text()).get("groups", {})


def shortfall(mean: float, reference: float, metric: str) -> float:
    """How much worse than the reference, in metric units. Negative means better."""
    return mean - reference if metric in LOWER_IS_BETTER else reference - mean


def standard_error(sd: float, n: int, ref: dict, metric: str) -> tuple[float, bool]:
    """SE of the difference between our mean and the reference.

    Both spreads are capped at the noise floor first, for the reason in `judge`:
    an unstable implementation must not buy tolerance by being unstable, and that
    holds for the reference as much as for us.

    Against MMSA's published table there is only one spread to use — the table
    states single values, so it is treated as exact and this reduces to the
    original sigma/sqrt(n). Against a reference we ran ourselves, the reference
    mean is an estimate from ten seeds like ours, and pretending otherwise claims
    a precision neither side has. Combining the two is the correct test, and it
    is a *looser* one, so it is fixed here before any of the three models it
    applies to has been run. See docs/decisions.md, 2026-07-31.
    """
    floor = NOISE_FLOOR.get(metric)
    cap = (lambda x: min(x, floor)) if floor is not None else (lambda x: x)
    if n <= 1:
        return float("inf"), False
    variance = cap(sd) ** 2 / n
    ref_sd, ref_n = ref.get(f"{metric}_sd"), ref.get("n")
    paired = ref_sd is not None and ref_n
    if paired:
        variance += cap(ref_sd) ** 2 / ref_n
    return math.sqrt(variance), paired


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
    if ref.get("_source") == "mmsa_code":
        print(f"\n  reference for {model} is our own 10-seed run of MMSA's code, not "
              f"MMSA's table — it publishes no number for this model.")
    elif ref.get("_source") == "author_code":
        print(f"\n  reference for {model} is our own 10-seed run of the AUTHORS' "
              f"release, selected on validation as we select — not the number the "
              f"paper reports, which selects on the test set.")

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
        floor = NOISE_FLOOR.get(metric)
        # A model's own spread may make this test stricter, never looser than
        # the dataset's baseline noise. Without the cap, an unstable
        # implementation buys tolerance by being unstable: EF-LSTM's two
        # collapsed seeds pushed sigma to 0.2095 and turned a shortfall of 3.2
        # noise floors into +1.9 SE. See docs/decisions.md, 2026-07-31.
        se, paired = standard_error(sd, n, ref, metric)
        gap = shortfall(mean, ref[metric], metric)
        in_se = gap / se if se > 0 else 0.0
        capped = floor is not None and sd > floor
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
        # Since the cap, this should be unreachable — a gap above the floor can
        # no longer hide under a wide spread. Kept as a tripwire: if it ever
        # fires again, the cap has a hole.
        if floor is not None and gap > floor and in_se <= WARN_SE:
            masked.append((metric, gap, floor, in_se))
        marker = " (primary)" if metric in PRIMARY else ""
        note = " [sd capped]" if capped else ""
        if paired:
            note += " [vs ref sd]"
        print(f"  {metric:12s}{mean:10.4f}{sd:9.4f}{ref[metric]:10.4f}"
              f"{gap:+11.4f}{in_se:+8.1f}  {tag}{marker}{note}")

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
