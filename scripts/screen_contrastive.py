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

A candidate is kept when it improves the validation metric by more than the
control's own seed spread. That threshold is the control's standard error, not
zero: with n=5 a candidate can beat the control by luck, and "better than the
control on average" is a claim this data cannot support at that scale.

    scripts/screen_contrastive.py run            # train everything (long)
    scripts/screen_contrastive.py report         # table from what is on disk
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
LAMBDA = "0.1"
#: Both metrics that the phase-1 acceptance criteria are written on.
METRICS = ("mae", "corr")
LOWER_IS_BETTER = {"mae"}


def group_name(candidate: str | None) -> str:
    return "screen_control" if candidate is None else f"screen_{candidate}"


def run_one(candidate: str | None, epochs: int, force: bool) -> None:
    group = group_name(candidate)
    directory = OUTPUTS / group
    if directory.exists() and not force:
        print(f"  {group}: already on disk, skipping (use --force to redo)")
        return
    command = [
        str(PYTHON), str(TRAIN), *BACKBONE,
        "--seeds", *SEEDS, "--epochs", str(epochs),
        "--run-group", group, "--quiet",
    ]
    if candidate is not None:
        command += ["--contrastive", candidate, "--contrastive-lambda", LAMBDA]
    print(f"  {group}: {' '.join(command[2:])}")
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stdout.strip().splitlines()[-6:])
        print(f"    FAILED (exit {result.returncode})\n{tail}\n{result.stderr.strip()[-600:]}")


def read_group(candidate: str | None) -> dict | None:
    """Validation metrics per seed, from the epoch each run selected."""
    directory = OUTPUTS / group_name(candidate)
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
    return {
        "candidate": candidate or "control",
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


def verdict(candidate: dict, control: dict) -> dict:
    """Kept if it clears the control's standard error on either metric."""
    out = {}
    keep = False
    for metric in METRICS:
        c, b = candidate["valid"][metric], control["valid"][metric]
        n = len(b["per_seed"])
        standard_error = b["sd"] / np.sqrt(n) if n > 1 else float("nan")
        delta = c["mean"] - b["mean"]
        improvement = -delta if metric in LOWER_IS_BETTER else delta
        clears = bool(improvement > standard_error)
        out[metric] = {
            "delta": float(delta),
            "improvement": float(improvement),
            "control_se": float(standard_error),
            "clears_se": clears,
        }
        keep = keep or clears
    out["keep"] = keep
    return out


def report(candidates: list[str]) -> int:
    control = read_group(None)
    if control is None:
        print("No control run on disk. Run `screen_contrastive.py run` first.")
        return 1

    rows, missing = [], []
    for name in candidates:
        data = read_group(name)
        if data is None:
            missing.append(name)
            continue
        rows.append((data, verdict(data, control)))

    print(f"backbone: {' '.join(BACKBONE)}  lambda {LAMBDA}  seeds {' '.join(SEEDS)}")
    print("selection metric: VALIDATION mae/corr, mean +- sd (ddof=1), n="
          f"{len(control['seeds'])}\n")
    header = f"{'candidate':16s}{'valid mae':>18s}{'valid corr':>18s}  {'own loss':>12s}  verdict"
    print(header)
    print("-" * len(header))
    control_line = (
        f"{'control (L1)':16s}"
        f"{control['valid']['mae']['mean']:11.4f} ± {control['valid']['mae']['sd']:.4f}"
        f"{control['valid']['corr']['mean']:11.4f} ± {control['valid']['corr']['sd']:.4f}"
        f"  {'—':>12s}  —"
    )
    print(control_line)

    rows.sort(key=lambda r: r[0]["valid"]["mae"]["mean"])
    kept = []
    for data, decision in rows:
        own = next(iter(data["own_loss"].values()), float("nan"))
        mark = "KEEP" if decision["keep"] else "drop"
        if decision["keep"]:
            kept.append(data["candidate"])
        print(
            f"{data['candidate']:16s}"
            f"{data['valid']['mae']['mean']:11.4f} ± {data['valid']['mae']['sd']:.4f}"
            f"{data['valid']['corr']['mean']:11.4f} ± {data['valid']['corr']['sd']:.4f}"
            f"  {own:12.4f}  {mark}"
        )

    print(f"\ncontrol SE: mae {control['valid']['mae']['sd']/np.sqrt(5):.4f}, "
          f"corr {control['valid']['corr']['sd']/np.sqrt(5):.4f}")
    print(f"kept {len(kept)}/{len(rows)}: {', '.join(kept) if kept else 'none'}")
    if missing:
        print(f"not run: {', '.join(missing)}")
    print("\nThe 'own loss' column is recorded, not compared: values from different "
          "objectives\nare not commensurate. It is there for the weighting steps, which "
          "read the curves.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "backbone": BACKBONE, "lambda": float(LAMBDA), "seeds": [int(s) for s in SEEDS],
        "selection": "validation mae/corr, mean over 5 seeds, threshold = control SE",
        "control": control,
        "candidates": [{**d, "verdict": v} for d, v in rows],
        "kept": kept,
    }, indent=2) + "\n")
    print(f"\nwrote {REPORT.relative_to(REPO)}")
    return 0


def main() -> int:
    sys.path.insert(0, str(REPO / "src"))
    from msa.losses import available_losses

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "report"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--only", nargs="+", default=None, help="a subset of candidates")
    ap.add_argument("--force", action="store_true", help="retrain groups already on disk")
    args = ap.parse_args()

    candidates = args.only or available_losses()
    if args.command == "report":
        return report(candidates)

    print(f"== control + {len(candidates)} candidate(s), {len(SEEDS)} seeds each ==")
    run_one(None, args.epochs, args.force)
    for name in candidates:
        run_one(name, args.epochs, args.force)
    print("\nDone. `screen_contrastive.py report` for the table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
