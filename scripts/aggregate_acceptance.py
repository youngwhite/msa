"""Is this implementation, as a whole, level with the reference?

`check_acceptance.py` asks the per-model question: did *this* port reproduce
*that* number. It cannot see a bias that lives across models — every group can
sit 0.5-1.5 SE behind, each one passing, while the directions agree far past
chance. That is exactly what docs/investigations.md#systematic-bias recorded on
2026-07-27 and left open.

This script asks the aggregate question. The criterion is fixed in
docs/decisions.md, 2026-07-31, *before* any number here was computed:

    Two statistics, per primary metric (MAE, Corr), over every model with a
    reference:

      sign test    how many models we are worse on, two-sided binomial. Direction
                   only, so no single model's magnitude can drive it.
      Stouffer Z   z_i = shortfall_i / SE_i, Z = sum(z_i)/sqrt(k), two-sided
                   normal. SE is check_acceptance's, cap and all, so the whole
                   project uses one definition.

    Z significantly negative  -> AHEAD
    neither test significant  -> LEVEL
    either significant positive -> SYSTEMATICALLY BEHIND

The noise floor is deliberately *not* applied here. In the per-model test it is
an exemption ("a shortfall smaller than a rerun's spread is not a failure");
zeroing those shortfalls would delete the very signal this test is built to
find.

Two caveats that decide how the output may be read, both from the decision entry:

  * The models are not independent — one 686-sample test set, one feature set,
    one training protocol. Stouffer assumes independence and therefore
    understates the variance. That makes BEHIND easier to reach and LEVEL harder,
    so a non-significant Z is the trustworthy direction.
  * Against MMSA's published table there is no reference variance, so the test
    treats the table as exact. It is not: #mmsa-all-eleven found 9 of 11 models
    where MMSA's own code does not reach MMSA's own published Corr. Falling short
    of the table is not, by itself, evidence of a defect on our side.

Usage:
    python scripts/aggregate_acceptance.py
    python scripts/aggregate_acceptance.py --metrics mae corr acc2_non0
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from check_acceptance import (
    CODE_REFERENCE_PATH,
    PRIMARY,
    REFERENCE_PATH,
    load_reference,
    shortfall,
    standard_error,
    summarise,
)

from msa.config import OUTPUT_ROOT

ALPHA = 0.05
#: Reported alongside the verdict but excluded from it, declared in the decision
#: entry before the first run of this script. Its collapsed seeds inflate both
#: the shortfall and the spread, and it would otherwise dominate Z.
SENSITIVITY_DROP = "ef_lstm"


def pick_reference(which: str) -> dict:
    """Which reference the aggregate is taken against.

    The two answer different questions and must not be pooled silently:

      table  MMSA's published numbers — the claim under test, but a claim
             #mmsa-all-eleven found MMSA's own code does not meet on 9 of 11
             models. Being behind here is not evidence of a defect on our side.
      code   MMSA's implementation, run by us on the same seeds and pickles.
             This one *is* implementation against implementation, and it is the
             comparison that can say whether our ports are faithful.
      auto   what check_acceptance uses: the table where it exists, our runs
             where it does not. Correct per model, incoherent in aggregate,
             which is why the first measurement of this script mixed eleven of
             one with three of the other.
    """
    if which == "auto":
        return load_reference()
    if which == "table":
        return json.loads(REFERENCE_PATH.read_text())["models"]
    return {model: {"_source": "mmsa_code"} | summarise(entry)
            for model, entry in json.loads(CODE_REFERENCE_PATH.read_text())["models"].items()}


def binomial_two_sided(worse: int, k: int) -> float:
    """P(a result at least this lopsided) under p=0.5, both tails."""
    if k == 0:
        return 1.0
    pmf = [math.comb(k, i) * 0.5**k for i in range(k + 1)]
    observed = pmf[worse]
    # "At least as extreme" means at least as improbable, which handles the
    # asymmetry when k is odd without picking a tail by hand.
    return min(1.0, sum(p for p in pmf if p <= observed * (1 + 1e-12)))


def normal_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def our_runs(dataset: str) -> dict[str, dict]:
    """Our acceptance groups for one dataset, keyed by model.

    Same rule as check_acceptance --all: only groups named <model>_<dataset>,
    never the ablations, which deviate from the reference on purpose.

    Restricted to a single dataset, which matters now that there is more than
    one. The reference files are MOSI's (`mmsa_reference_mosi.json`,
    `mmsa_code_runs_mosi.json`) and the seed noise floor is per-dataset, so a
    run that accepted whatever `<model>_<dataset>` groups happened to exist
    would eventually average two datasets into one verdict.
    """
    runs = {}
    for summary_path in sorted(OUTPUT_ROOT.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        model, group = payload.get("model", ""), summary_path.parent.name
        found = payload.get("dataset", "").lower().replace("cmu-", "")
        if found == dataset and group == f"{model}_{found}":
            runs[model] = payload["summary"]
    return runs


def mmsa_runs() -> dict[str, dict]:
    """MMSA's own code as the subject under test, in the same shape as ours.

    This is the control that decides how the table comparison may be read. If
    MMSA's implementation falls short of MMSA's published table by as much as we
    do, then the shortfall is a property of the target, not of our ports — and
    #mmsa-all-eleven already found that to be true model by model at n=5. Here it
    is put through the identical aggregate test rather than eyeballed.
    """
    subject = {}
    for model, entry in json.loads(CODE_REFERENCE_PATH.read_text())["models"].items():
        stats = summarise(entry)
        subject[model] = {
            metric: {"mean": stats[metric], "std": stats.get(f"{metric}_sd", 0.0),
                     "n": stats["n"]}
            for metric in stats if isinstance(stats[metric], float)
            and not metric.endswith("_sd")
        }
    return subject


def collect(reference: dict, metrics: tuple[str, ...],
            subject: dict[str, dict], dataset: str) -> dict[str, list[dict]]:
    """One row per (metric, model): the shortfall and the SE it is measured in."""
    rows: dict[str, list[dict]] = {metric: [] for metric in metrics}
    for model, stats in sorted(subject.items()):
        if model not in reference:
            continue
        ref = reference[model]
        for metric in metrics:
            if metric not in stats or metric not in ref:
                continue
            mean, sd, n = stats[metric]["mean"], stats[metric]["std"], stats[metric]["n"]
            se, paired = standard_error(sd, n, ref, metric, dataset)
            gap = shortfall(mean, ref[metric], metric)
            rows[metric].append({
                "model": model, "ours": mean, "ref": ref[metric], "n": n,
                "gap": gap, "se": se, "z": gap / se if se > 0 else 0.0,
                "source": "mmsa_code" if paired else "mmsa_table",
            })
    return rows


def test(rows: list[dict]) -> dict:
    k = len(rows)
    worse = sum(1 for r in rows if r["gap"] > 0)
    z_sum = sum(r["z"] for r in rows)
    Z = z_sum / math.sqrt(k) if k else 0.0
    return {
        "k": k, "worse": worse, "sign_p": binomial_two_sided(worse, k),
        "Z": Z, "z_p": normal_two_sided(Z),
    }


def verdict_of(sign_p: float, z_p: float, Z: float) -> str:
    if z_p < ALPHA and Z < 0:
        return "AHEAD"
    if z_p < ALPHA or sign_p < ALPHA:
        return "SYSTEMATICALLY BEHIND"
    return "LEVEL"


def report(metric: str, rows: list[dict], subject_name: str) -> str:
    print(f"\n=== {metric} ===")
    print(f"  {'model':12s}{subject_name:>10s}{'ref':>10s}{'gap':>11s}{'SE':>9s}"
          f"{'z':>7s}{'n':>4s}  reference")
    for r in sorted(rows, key=lambda r: -r["z"]):
        print(f"  {r['model']:12s}{r['ours']:10.4f}{r['ref']:10.4f}{r['gap']:+11.4f}"
              f"{r['se']:9.4f}{r['z']:+7.2f}{r['n']:4d}  {r['source']}")
    t = test(rows)
    verdict = verdict_of(t["sign_p"], t["z_p"], t["Z"])
    print(f"\n  sign test : worse on {t['worse']}/{t['k']} models, p = {t['sign_p']:.4f}")
    print(f"  Stouffer  : Z = {t['Z']:+.2f}, p = {t['z_p']:.2e}"
          f"   (positive Z = the subject is behind)")
    print(f"  VERDICT   : {verdict}")

    kept = [r for r in rows if r["model"] != SENSITIVITY_DROP]
    if len(kept) != len(rows):
        s = test(kept)
        print(f"  sensitivity (without {SENSITIVITY_DROP}, not part of the verdict): "
              f"worse on {s['worse']}/{s['k']}, sign p = {s['sign_p']:.4f}, "
              f"Z = {s['Z']:+.2f}, p = {s['z_p']:.2e}")
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", default=list(PRIMARY),
                    help=f"default: the primary metrics {list(PRIMARY)}")
    ap.add_argument("--reference", choices=("auto", "table", "code"), default="auto",
                    help="auto: table where published, our MMSA runs elsewhere "
                         "(check_acceptance's rule). table/code: one source only, "
                         "which is what makes the aggregate interpretable")
    ap.add_argument("--subject", choices=("ours", "mmsa_code"), default="ours",
                    help="whose numbers are under test. mmsa_code puts MMSA's own "
                         "implementation through the identical test against the "
                         "table it published — the control for reading our own "
                         "shortfall against that table")
    ap.add_argument("--dataset", default="mosi",
                    help="which dataset's groups and seed-noise floor to use. The "
                         "reference files are MOSI's, so this is the only value "
                         "that currently has a reference to compare against")
    args = ap.parse_args()

    reference = pick_reference(args.reference)
    sources = {"auto": "MMSA's table where published, our runs of its code elsewhere",
               "table": "MMSA's published table only",
               "code": "our own runs of MMSA's code only"}
    subject = (our_runs(args.dataset) if args.subject == "ours" else mmsa_runs())
    if args.subject == "mmsa_code" and args.reference != "table":
        ap.error("--subject mmsa_code only makes sense against --reference table; "
                 "anything else compares MMSA's runs with themselves")
    subject_name = ("our implementation" if args.subject == "ours"
                    else "MMSA implementation, run by us")
    print(f"subject  : {subject_name}")
    print(f"reference: {sources[args.reference]}")
    rows = collect(reference, tuple(args.metrics), subject, args.dataset)
    verdicts = {}
    label = "ours" if args.subject == "ours" else "MMSA"
    for metric in args.metrics:
        if not rows[metric]:
            print(f"\n=== {metric} ===\n  no group has a reference for this metric")
            continue
        verdicts[metric] = report(metric, rows[metric], label)

    primary = [v for m, v in verdicts.items() if m in PRIMARY]
    print("\n" + "=" * 60)
    print(f"primary metrics: {', '.join(f'{m}={verdicts[m]}' for m in PRIMARY if m in verdicts)}")
    if primary and all(v in ("LEVEL", "AHEAD") for v in primary):
        print("level with the reference, or ahead of it, on every primary metric.")
        print("Read this with the two caveats in the module docstring — the "
              "reference is a table, not a distribution.")
        sys.exit(0)
    print("systematically behind on at least one primary metric.")
    print("The remedy is to find the implementation or protocol difference. "
          "Tuning to close it is forbidden (docs/decisions.md).")
    sys.exit(1)


if __name__ == "__main__":
    main()
