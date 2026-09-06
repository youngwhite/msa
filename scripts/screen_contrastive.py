"""Phase 2, step 1: which contrastive objectives are worth combining?

Each candidate is trained as ``task_loss + lambda * candidate`` on one backbone
and compared with a control trained identically but without the candidate. The
comparison is on **validation** MAE and Corr over five seeds, reported as
mean +- sample standard deviation (ddof=1).

Three things about that sentence are load-bearing:

* **Validation, not test.** Convention 2. The test split is for whatever
  survives, once, later. A screen run on test would leave nothing to confirm on.
* **Five seeds.** Convention 1. LF-LSTM's seed-to-seed MAE spread on MOSI is
  0.039, wider than most published improvements, so a single-seed screen would
  mostly rank the seeds.
* **Not on the candidate's own loss value.** An InfoNCE value and an L1 value
  are not comparable quantities; ranking objectives by their own numbers ranks
  them by their normalisation constants. The candidates' loss curves *are*
  recorded, per epoch, because the later weighting steps need them -- but they
  do not decide anything here.

**The criterion, revised 2026-09-04 after the first round.** The original rule
-- keep a candidate that beats the control by more than one standard error on
either metric -- has a defect that its first run demonstrated rather than
survived: it applies no multiple-comparison correction. One-sided, a 1-SE
threshold passes 15.9% of the time under the null; with two metrics per
candidate that is a 29% chance of keeping a candidate that does nothing, so out
of 14 the expected haul under the null is 4.1. The first round kept exactly 4,
and each of those four cleared on one metric while being negative on the other.
That output is indistinguishable from noise, so the rule had to change before
any more data was looked at. What it now requires, all three:

1. **Welch's t-test, one-sided, BH-corrected.** Every (candidate, lambda,
   metric) is one test; Benjamini-Hochberg at q=0.10 across all of them.
   FDR rather than Bonferroni because this is a screen -- the cost of a false
   positive here is one wasted confirmation run, not a wrong published claim.
   Unpaired, even though control and candidate share seeds 42-46: adding a loss
   term changes the whole trajectory, and this project already withdrew a paired
   test once when the same premise failed across implementations
   (docs/roadmap.md). The observed seed correlation is reported as a diagnostic
   instead of being assumed.
2. **An effect-size floor**: at least 0.01 MAE or 0.01 Corr, about half the
   control's seed spread. A significant effect smaller than this cannot matter
   to anything downstream, and the combination step would be built on it.
3. **No trading one metric for the other**: the metric that did not clear must
   not itself be worse than the control by more than one SE. The first round's
   `triplet` gained 1.06 SE of Corr while losing 0.81 SE of MAE, which is not an
   improvement, it is a rotation.

    scripts/screen_contrastive.py run            # train everything
    scripts/screen_contrastive.py run --lambdas 0.01 0.03 0.3 1.0
    scripts/screen_contrastive.py report        # table from what is on disk
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
PYTHON = REPO / ".venv" / "bin" / "python"
TRAIN = REPO / "scripts" / "train.py"
REPORT = REPO / "docs" / "contrastive_screen.json"

#: TFN with the hyper-parameters it reproduces MMSA under. Fast (48s a run, so
#: 14 candidates x 5 seeds is about an hour) and it already returns pooled
#: modality features, so nothing about the backbone had to change to host this.
BACKBONE = ["--model", "tfn", "--unaligned", "--lr", "1e-3", "--weight-decay", "0"]
SEEDS = ["42", "43", "44", "45", "46"]
#: The first round's value. Kept as the unsuffixed group name so its runs are
#: still found after the sweep grew a lambda axis.
BASE_LAMBDA = 0.1
DEFAULT_LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0)
#: Both metrics that the phase-1 acceptance criteria are written on.
METRICS = ("mae", "corr")
LOWER_IS_BETTER = {"mae"}
#: BH false-discovery rate, and the smallest effect worth carrying forward.
FDR_Q = 0.10
EFFECT_FLOOR = {"mae": 0.01, "corr": 0.01}


def lambda_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


#: Set by main(); MOSI's groups keep their original unprefixed names so the
#: committed results docs/experiments.md points at are not orphaned.
DATASET = "mosi"


def group_name(candidate: str | None, lambda_: float = BASE_LAMBDA) -> str:
    """The control has no contrastive term, so lambda does not apply to it."""
    prefix = "screen" if DATASET == "mosi" else f"screen_{DATASET}"
    if candidate is None:
        return f"{prefix}_control"
    if lambda_ == BASE_LAMBDA:
        return f"{prefix}_{candidate}"
    return f"{prefix}_{candidate}_l{lambda_tag(lambda_)}"


def run_one(candidate: str | None, epochs: int, force: bool,
            lambda_: float = BASE_LAMBDA) -> None:
    group = group_name(candidate, lambda_)
    directory = OUTPUTS / group
    if directory.exists() and not force:
        # Complete means all the seeds, not just the directory. A sweep killed
        # partway leaves a group with some of them, and skipping on existence
        # alone would silently fold an n=4 group into a test that says n=5 --
        # which is exactly what a `screen_rnc_l0p03` with four seeds did after
        # this sweep was killed for memory.
        present = sorted(path.name for path in directory.glob("seed*"))
        if len(present) == len(SEEDS):
            print(f"  {group}: already on disk, skipping (use --force to redo)")
            return
        print(f"  {group}: incomplete ({len(present)}/{len(SEEDS)} seeds: "
              f"{' '.join(present)}), redoing")
    command = [
        str(PYTHON), str(TRAIN), *BACKBONE, "--dataset", DATASET,
        "--seeds", *SEEDS, "--epochs", str(epochs),
        "--run-group", group, "--quiet",
    ]
    if candidate is not None:
        command += ["--contrastive", candidate,
                    "--contrastive-lambda", f"{lambda_:g}"]
    print(f"  {group}: {' '.join(command[2:])}")
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stdout.strip().splitlines()[-6:])
        print(f"    FAILED (exit {result.returncode})\n{tail}\n{result.stderr.strip()[-600:]}")


def read_group(candidate: str | None, lambda_: float = BASE_LAMBDA) -> dict | None:
    """Validation metrics per seed, from the epoch each run selected."""
    directory = OUTPUTS / group_name(candidate, lambda_)
    if not directory.exists():
        return None
    per_seed: dict[str, list[float]] = {metric: [] for metric in METRICS}
    seeds, terms = [], {}
    for path in sorted(directory.glob("seed*/result.json")):
        result = json.loads(path.read_text())
        if result.get("env", {}).get("git", {}).get("dirty", True):
            print(f"    warning: {path.parent.name} came from a dirty tree", file=sys.stderr)
        best = next(
            row for row in result["history"] if row["epoch"] == result["best_epoch"]
        )
        for metric in METRICS:
            per_seed[metric].append(best[f"valid_{metric}"])
        seeds.append(result["seed"])
        for key, value in best.items():
            if key.startswith("loss_"):
                terms.setdefault(key, []).append(value)
    if not seeds:
        return None
    if len(seeds) != len(SEEDS):
        print(f"    warning: {directory.name} has {len(seeds)} seed(s), expected "
              f"{len(SEEDS)} — excluded from the table", file=sys.stderr)
        return None
    return {
        "candidate": candidate or "control",
        "lambda": None if candidate is None else lambda_,
        "seeds": seeds,
        "valid": {
            metric: {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                "per_seed": values,
            }
            for metric, values in per_seed.items()
        },
        "own_loss": {k: float(np.mean(v)) for k, v in terms.items()},
    }


def welch_one_sided(candidate: list[float], control: list[float], metric: str
                    ) -> tuple[float, float]:
    """(improvement, one-sided p) for "candidate is better than control".

    Welch rather than Student because the two groups have no reason to share a
    variance -- an auxiliary loss can easily make a run more or less
    seed-sensitive, and `simsiam` in the first round halved the spread.
    """
    from scipy import stats

    delta = float(np.mean(candidate) - np.mean(control))
    improvement = -delta if metric in LOWER_IS_BETTER else delta
    statistic, p_two = stats.ttest_ind(candidate, control, equal_var=False)
    if metric in LOWER_IS_BETTER:
        statistic = -statistic
    # One-sided in the direction of improvement.
    p_one = p_two / 2 if statistic > 0 else 1 - p_two / 2
    return improvement, float(p_one)


def benjamini_hochberg(p_values: list[float], q: float) -> list[bool]:
    """Which hypotheses BH rejects at false-discovery rate `q`."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    n = len(p_values)
    threshold_rank = -1
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= q * rank / n:
            threshold_rank = rank
    rejected = [False] * n
    for rank, index in enumerate(order, start=1):
        if rank <= threshold_rank:
            rejected[index] = True
    return rejected


def seed_correlation(candidate: dict, control: dict, metric: str) -> float:
    """Diagnostic only: is a paired test's premise even plausible here?"""
    a = candidate["valid"][metric]["per_seed"]
    b = control["valid"][metric]["per_seed"]
    if len(a) != len(b) or len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def report(candidates: list[str], lambdas: list[float]) -> int:
    control = read_group(None)
    if control is None:
        print("No control run on disk. Run `screen_contrastive.py run` first.")
        return 1

    from scipy import stats

    tests, two_sided_tests, rows = [], [], []
    for name in candidates:
        for lambda_ in lambdas:
            data = read_group(name, lambda_)
            if data is None:
                continue
            entry = {"data": data, "metrics": {}}
            for metric in METRICS:
                improvement, p_one = welch_one_sided(
                    data["valid"][metric]["per_seed"],
                    control["valid"][metric]["per_seed"],
                    metric,
                )
                # Two-sided as well. The one-sided test asks "does it help" and
                # scores a significantly *harmful* candidate at p near 1, which
                # prints as a plain "drop" -- the same blind spot that made the
                # MMIM diagnostic report NOT DETECTED against an effect of
                # |d|=1.39 (investigations.md#mosei-contrast-harmful). Harm is a
                # finding here too, not an absence.
                _, p_two = stats.ttest_ind(
                    data["valid"][metric]["per_seed"],
                    control["valid"][metric]["per_seed"], equal_var=False,
                )
                entry["metrics"][metric] = {
                    "improvement": improvement,
                    "p_one_sided": p_one,
                    "p_two_sided": float(p_two),
                    "clears_floor": bool(improvement >= EFFECT_FLOOR[metric]),
                    "seed_corr": seed_correlation(data, control, metric),
                }
                tests.append((len(rows), metric, p_one))
                two_sided_tests.append((len(rows), metric, float(p_two)))
            rows.append(entry)

    if not rows:
        print("Nothing to report: no candidate groups on disk.")
        return 1

    rejected = benjamini_hochberg([p for _, _, p in tests], FDR_Q)
    for (row_index, metric, _), is_rejected in zip(tests, rejected, strict=True):
        rows[row_index]["metrics"][metric]["bh_significant"] = bool(is_rejected)
    two_rejected = benjamini_hochberg([p for _, _, p in two_sided_tests], FDR_Q)
    for (row_index, metric, _), is_rejected in zip(
        two_sided_tests, two_rejected, strict=True
    ):
        rows[row_index]["metrics"][metric]["bh_any_direction"] = bool(is_rejected)

    control_se = {
        m: float(control["valid"][m]["sd"] / np.sqrt(len(control["seeds"])))
        for m in METRICS
    }
    for entry in rows:
        keep, reasons = False, []
        for metric in METRICS:
            info = entry["metrics"][metric]
            other = METRICS[1] if metric == METRICS[0] else METRICS[0]
            other_improvement = entry["metrics"][other]["improvement"]
            no_tradeoff = bool(other_improvement >= -control_se[other])
            info["no_tradeoff"] = no_tradeoff
            if info["bh_significant"] and info["clears_floor"] and no_tradeoff:
                keep = True
                reasons.append(metric)
        entry["keep"], entry["kept_on"] = keep, reasons

    print(f"backbone: {' '.join(BACKBONE)}  seeds {' '.join(SEEDS)}")
    print(f"criterion: Welch one-sided, BH q={FDR_Q} over {len(tests)} tests; "
          f"effect floor {EFFECT_FLOOR}; no metric worse than control SE")
    print(f"control (L1): mae {control['valid']['mae']['mean']:.4f} ± "
          f"{control['valid']['mae']['sd']:.4f}   corr "
          f"{control['valid']['corr']['mean']:.4f} ± "
          f"{control['valid']['corr']['sd']:.4f}   "
          f"SE mae {control_se['mae']:.4f} corr {control_se['corr']:.4f}\n")

    header = (f"{'candidate':15s}{'lam':>6s}{'Δmae':>9s}{'p2':>10s}"
              f"{'Δcorr':>9s}{'p2':>10s}  effect     verdict")
    print(header)
    print("-" * len(header))
    rows.sort(key=lambda e: min(
        e["metrics"][m]["p_one_sided"] for m in METRICS
    ))
    kept = []
    for entry in rows:
        data, info = entry["data"], entry["metrics"]
        marks = []
        for metric in METRICS:
            if info[metric]["bh_significant"]:
                marks.append(f"{metric}*")
        note = "KEEP:" + ",".join(entry["kept_on"]) if entry["keep"] else (
            "sig-but-filtered" if marks else "")
        if entry["keep"]:
            kept.append((data["candidate"], data["lambda"]))
        directions = []
        for metric in METRICS:
            if info[metric]["bh_any_direction"]:
                directions.append(
                    f"{metric}:{'+' if info[metric]['improvement'] > 0 else '-'}"
                )
        effect = ",".join(directions) if directions else "none"
        print(
            f"{data['candidate']:15s}{data['lambda']:6g}"
            f"{info['mae']['improvement']:+9.4f}{info['mae']['p_two_sided']:10.2e}"
            f"{info['corr']['improvement']:+9.4f}{info['corr']['p_two_sided']:10.2e}"
            f"  {effect:9s}  {note}"
        )

    correlations = [
        entry["metrics"][metric]["seed_corr"]
        for entry in rows for metric in METRICS
        if not np.isnan(entry["metrics"][metric]["seed_corr"])
    ]
    print(f"\nBH rejected {sum(rejected)}/{len(tests)} test(s) at q={FDR_Q}")
    print(f"kept after the effect floor and the no-tradeoff rule: {len(kept)}")
    for name, lambda_ in kept:
        print(f"  {name} at lambda {lambda_:g}")
    if not kept:
        print("  none")
    if correlations:
        print(f"\nseed correlation with the control, across all runs: "
              f"{min(correlations):+.2f} … {max(correlations):+.2f} "
              f"(median {float(np.median(correlations)):+.2f}) — "
              f"the diagnostic for whether a paired test would have been valid")

    out = (REPORT if DATASET == "mosi"
           else REPORT.with_name(f"contrastive_screen_{DATASET}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": DATASET,
        "backbone": BACKBONE,
        "lambdas": lambdas,
        "seeds": [int(s) for s in SEEDS],
        "criterion": {
            "test": "Welch one-sided, unpaired",
            "correction": f"Benjamini-Hochberg q={FDR_Q}",
            "n_tests": len(tests),
            "effect_floor": EFFECT_FLOOR,
            "no_tradeoff": "other metric not worse than control SE",
        },
        "control": control,
        "results": [
            {**e["data"], "metrics": e["metrics"], "keep": e["keep"],
             "kept_on": e["kept_on"]}
            for e in rows
        ],
        "kept": [{"candidate": c, "lambda": lam} for c, lam in kept],
    }, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


#: The three mechanism groups from msa.losses.contrastive, so the combination
#: does not end up as four variants of one idea.
FAMILIES = {
    "cross_modal": ["infonce", "nt_xent", "dcl", "hcl", "cpc", "triplet",
                    "max_margin", "align_uniform"],
    "negative_free": ["barlow_twins", "vicreg", "simsiam"],
    "label_aware": ["supcon", "rnc", "arcface"],
}


def combine(candidates: list[str], lambdas: list[float]) -> int:
    """Choose the combination's terms by a stated rule, not by significance.

    The screen could not separate these candidates -- with n=5 and a control sd
    of 0.021 MAE its minimum detectable effect is 0.037 even uncorrected, twice
    the largest effect anyone showed (see investigations.md#screen-underpowered).
    So selecting "the significant ones" is not available, and pretending
    otherwise would be the worst of the options.

    What is available is a rule that is written down and applied mechanically:

    1. Rank every candidate at every lambda on each metric; average the ranks.
       Averaging over the whole lambda grid rather than taking each candidate's
       best lambda matters -- `rnc` is the best single cell in the entire sweep
       at lambda 0.03 and among the two worst at 0.1, and a best-cell rule would
       carry that noise straight into the combination.
    2. Take the best-ranked candidate from each of the three mechanism families,
       so the combination tests three different ideas rather than three
       spellings of one.
    3. Fill the fourth slot with the best remaining candidate overall. Four is
       the cap the phase brief sets.
    4. The combination's lambda is the one with the best mean rank across the
       four chosen terms.

    This is a selection, not a finding. Nothing here claims these four are the
    ones that work; they are the four the recorded numbers rank highest, chosen
    before the weighting experiment so that experiment is not also a search.
    """
    control = read_group(None)
    if control is None:
        print("No control run on disk.")
        return 1

    cells = {}
    for name in candidates:
        for lambda_ in lambdas:
            data = read_group(name, lambda_)
            if data is None:
                continue
            cells[(name, lambda_)] = {
                metric: welch_one_sided(
                    data["valid"][metric]["per_seed"],
                    control["valid"][metric]["per_seed"], metric,
                )[0]
                for metric in METRICS
            }
    if not cells:
        print("No candidate groups on disk.")
        return 1

    # Rank within each (lambda, metric) column: 1 is the best improvement.
    rank_at: dict[tuple[str, float], list[int]] = {}
    for lambda_ in lambdas:
        for metric in METRICS:
            column = [(name, cells[(name, lambda_)][metric])
                      for name in candidates if (name, lambda_) in cells]
            column.sort(key=lambda pair: -pair[1])
            for position, (name, _) in enumerate(column, start=1):
                rank_at.setdefault((name, lambda_), []).append(position)

    #: Mean rank of a candidate at one lambda, over both metrics.
    per_lambda = {key: float(np.mean(v)) for key, v in rank_at.items()}
    #: Mean rank of a candidate over the whole grid.
    mean_rank = {
        name: float(np.mean([per_lambda[(name, lam)] for lam in lambdas
                             if (name, lam) in per_lambda]))
        for name in candidates
        if any((name, lam) in per_lambda for lam in lambdas)
    }
    order = sorted(mean_rank, key=lambda n: mean_rank[n])

    chosen, why = [], {}
    for family, members in FAMILIES.items():
        ranked = [n for n in order if n in members]
        if ranked:
            chosen.append(ranked[0])
            why[ranked[0]] = f"best of {family}"
    for name in order:
        if len(chosen) >= 4:
            break
        if name not in chosen:
            chosen.append(name)
            why[name] = "best remaining overall"

    lambda_score = {
        lam: float(np.mean([per_lambda[(n, lam)] for n in chosen
                            if (n, lam) in per_lambda]))
        for lam in lambdas
        if all((n, lam) in per_lambda for n in chosen)
    }
    chosen_lambda = min(lambda_score, key=lambda lam: lambda_score[lam])

    print("mean rank across the whole lambda grid (1 = best; "
          f"{len(mean_rank)} candidates, {len(lambdas)} lambdas x "
          f"{len(METRICS)} metrics):\n")
    for name in order:
        mark = f"  <- {why[name]}" if name in chosen else ""
        family = next(f for f, m in FAMILIES.items() if name in m)
        print(f"  {mean_rank[name]:5.2f}  {name:15s}{family:16s}{mark}")

    print("\nmean rank of the four chosen, per lambda:")
    for lam in sorted(lambda_score):
        star = "  <- chosen" if lam == chosen_lambda else ""
        print(f"  lambda {lam:5g}   {lambda_score[lam]:5.2f}{star}")

    favour = next(n for n in order if n in chosen)
    print(f"\ncombination ({len(chosen)} terms, cap is 4): {' '.join(sorted(chosen))}")
    print(f"lambda: {chosen_lambda:g}")
    print(f"linear_ramp favours: {favour}  (best-ranked of the four)")
    print("\nThis is a selection under a stated rule, not a finding. The screen "
          "could not\nseparate these candidates; see "
          "investigations.md#screen-underpowered.")
    return 0


def main() -> int:
    sys.path.insert(0, str(REPO / "src"))
    from msa.losses import available_losses

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "report", "combine"))
    ap.add_argument("--dataset", default="mosi", choices=("mosi", "mosei"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--only", nargs="+", default=None, help="a subset of candidates")
    ap.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS),
                    help="contrastive weights to sweep (default: "
                         f"{' '.join(f'{x:g}' for x in DEFAULT_LAMBDAS)})")
    ap.add_argument("--force", action="store_true", help="retrain groups already on disk")
    args = ap.parse_args()

    global DATASET
    DATASET = args.dataset
    candidates = args.only or available_losses()
    if args.command == "report":
        return report(candidates, args.lambdas)
    if args.command == "combine":
        return combine(candidates, args.lambdas)

    total = len(candidates) * len(args.lambdas)
    print(f"== control + {len(candidates)} candidate(s) x {len(args.lambdas)} "
          f"lambda(s) = {total} group(s), {len(SEEDS)} seeds each ==")
    run_one(None, args.epochs, args.force)
    for lambda_ in args.lambdas:
        print(f"-- lambda {lambda_:g}")
        for name in candidates:
            run_one(name, args.epochs, args.force, lambda_)
    print("\nDone. `screen_contrastive.py report` for the table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
