"""Audit stored runs: do the recorded numbers follow from the stored predictions?

For every `outputs/*/seed*/result.json` this re-derives the metrics from
`test_predictions.npy` and the dataset labels, re-hashes the predictions, and
re-derives each group's `summary.json` from its per-seed results. Anything that
disagrees is either a corrupted artefact or a metric/summary change that was made
without re-running the experiments — both of which invalidate the write-up.

Usage:
    python scripts/verify_runs.py [--outputs outputs] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from msa.config import OUTPUT_ROOT, get_dataset_spec
from msa.data import MMSADataset
from msa.metrics import eval_sentiment
from msa.trainer import checksum

TOLERANCE = 1e-9


def dataset_labels(dataset: str, aligned: bool) -> np.ndarray:
    spec = get_dataset_spec(dataset)
    return MMSADataset(spec, "test", aligned=aligned).labels.numpy()


def verify_run(run_dir: Path, labels_cache: dict, warnings: list[str]) -> list[str]:
    """Returns a list of problems; empty means the run checks out.

    Provenance gaps (no commit recorded, dirty tree) go to `warnings` instead:
    they make a run harder to reproduce but do not make its stored numbers wrong.
    """
    problems = []
    result = json.loads((run_dir / "result.json").read_text())
    preds = np.load(run_dir / "test_predictions.npy")

    dataset = result.get("dataset") or result.get("cli", {}).get("dataset", "mosi")
    aligned = result.get("aligned", True)
    key = (dataset, aligned)
    if key not in labels_cache:
        labels_cache[key] = dataset_labels(dataset, aligned)
    labels = labels_cache[key]

    if preds.shape != labels.shape:
        return [f"{run_dir}: predictions {preds.shape} do not match the test split "
                f"{labels.shape}"]

    recomputed = eval_sentiment(preds, labels)
    for name, stored in result["test"].items():
        if name not in recomputed:
            problems.append(f"{run_dir}: metric {name!r} is stored but no longer computed")
        elif abs(recomputed[name] - stored) > TOLERANCE:
            problems.append(
                f"{run_dir}: {name} stored {stored:.6f} but recomputes to "
                f"{recomputed[name]:.6f}"
            )
    stored_hash = result.get("test_pred_sha256_16")
    if stored_hash and stored_hash != checksum(preds):
        problems.append(f"{run_dir}: prediction checksum {stored_hash} does not match "
                        f"the stored array ({checksum(preds)})")
    git = result.get("env", {}).get("git") or {}
    if git.get("dirty"):
        warnings.append(f"{run_dir}: produced from a dirty working tree — the exact "
                        "code cannot be recovered from its commit")
    elif not git.get("commit"):
        warnings.append(f"{run_dir}: no git commit recorded")
    return problems


def verify_group(group_dir: Path) -> list[str]:
    """Check summary.json against the per-seed results it claims to summarise."""
    summary_path = group_dir / "summary.json"
    if not summary_path.exists():
        return []
    problems = []
    summary = json.loads(summary_path.read_text())
    per_seed_files = sorted(group_dir.glob("seed*/result.json"))
    stored_by_seed = {
        json.loads(p.read_text())["seed"]: json.loads(p.read_text())["test"]
        for p in per_seed_files
    }
    for metric, stats in summary.get("summary", {}).items():
        values = np.array(
            [stored_by_seed[int(s)][metric] for s in stats["per_seed"]
             if int(s) in stored_by_seed],
            dtype=np.float64,
        )
        if values.size != len(stats["per_seed"]):
            problems.append(f"{group_dir}: summary lists seeds absent from disk")
            break
        expected_std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        if abs(values.mean() - stats["mean"]) > TOLERANCE:
            problems.append(f"{group_dir}: {metric} mean disagrees with the per-seed runs")
        if abs(expected_std - stats["std"]) > 1e-6:
            problems.append(
                f"{group_dir}: {metric} std {stats['std']:.6f} is not the sample "
                f"std of its seeds ({expected_std:.6f}) — summary predates the "
                "ddof=1 convention, re-run or re-summarise"
            )
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--quiet", action="store_true", help="only print problems")
    args = ap.parse_args()

    if not args.outputs.exists():
        raise SystemExit(f"no results directory at {args.outputs}")

    labels_cache: dict = {}
    problems: list[str] = []
    warnings: list[str] = []
    n_runs, n_groups = 0, 0
    for group_dir in sorted(p for p in args.outputs.iterdir() if p.is_dir()):
        runs = sorted(group_dir.glob("seed*/result.json"))
        if not runs:
            continue
        n_groups += 1
        before = len(problems)
        for run in runs:
            n_runs += 1
            problems += verify_run(run.parent, labels_cache, warnings)
        problems += verify_group(group_dir)
        if not args.quiet:
            status = "OK  " if len(problems) == before else "FAIL"
            print(f"[{status}] {group_dir.name}: {len(runs)} run(s)")

    print(f"\nchecked {n_runs} run(s) in {n_groups} group(s)")
    if warnings:
        print(f"\n{len(warnings)} provenance warning(s):")
        for w in warnings:
            print(f"  ! {w}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("every stored metric follows from its stored predictions")


if __name__ == "__main__":
    main()
