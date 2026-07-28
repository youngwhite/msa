"""Run a group's seeds as concurrent processes, then rebuild its summary.

A single seed leaves the GPU mostly idle — MulT sits at ~35% utilisation and
110W of a 300W budget, because batch 16 over a 2.6M-parameter model is bound by
kernel launches, not arithmetic. Seeds are independent, so running several at
once is close to free wall-clock.

This deliberately does not touch `train.py`. Each seed is an ordinary
`train.py --seeds <one>` subprocess, so whatever that produces serially is what
it produces here: determinism comes from seeding plus deterministic kernels,
neither of which knows about other processes on the device. `--verify` checks
that claim rather than trusting it.

The one thing that does need repairing afterwards is `summary.json`: every
subprocess writes one covering only its own seed, so the last to finish leaves a
partial file behind. The rebuild below reuses `msa.trainer.summarize`, the same
function train.py calls, so the two cannot drift apart.

Concurrency is bounded by memory, not by cores. Peak usage per process, measured
on a 16.3 GB card:

    MulT    ~5.6 GB   -> 2 jobs; 3 would exceed the card
    MISA / Self-MM / text-BERT (fine-tuned BERT)  -> 2 jobs
    TFN / LMF / MFN / Graph-MFN / LF-DNN / EF-LSTM -> 4 jobs

Passing more than fits does not fail gracefully; CUDA raises out of memory
partway through and the group is left incomplete. When unsure, use 2.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msa.config import OUTPUT_ROOT  # noqa: E402
from msa.trainer import summarize  # noqa: E402

PY = str(PROJECT_ROOT / ".venv" / "bin" / "python")


def split_seeds(argv: list[str]) -> tuple[list[str], list[str]]:
    """Pull `--seeds a b c` out of an argument list, returning (rest, seeds)."""
    if "--seeds" not in argv:
        raise SystemExit("no --seeds in the arguments; nothing to parallelise")
    i = argv.index("--seeds")
    j = i + 1
    while j < len(argv) and not argv[j].startswith("-"):
        j += 1
    return argv[:i] + argv[j:], argv[i + 1:j]


def rebuild_summary(group_dir: Path) -> dict:
    """Re-derive summary.json from the per-seed results on disk."""
    runs = []
    for path in sorted(group_dir.glob("seed*/result.json"),
                       key=lambda p: int(p.parent.name.removeprefix("seed"))):
        payload = json.loads(path.read_text())
        runs.append(SimpleNamespace(
            seed=payload["config"]["seed"],
            test=payload["test"],
            dataset=payload["dataset"],
            config=payload["config"],
            env=payload["env"],
            cli=payload.get("cli", {}),
            model_kwargs=payload.get("model_kwargs", {}),
            aligned=payload.get("aligned"),
            checksum=payload["test_pred_sha256_16"],
        ))
    if not runs:
        raise SystemExit(f"{group_dir}: no seed results to summarise")
    first = runs[0]
    payload = {
        "model": first.cli.get("model"),
        "dataset": first.dataset,
        "aligned": first.aligned,
        "seeds": [r.seed for r in runs],
        "cli": first.cli,
        "model_kwargs": first.model_kwargs,
        "config": first.config,
        "env": first.env,
        "summary": summarize(runs),
        "checksums": {str(r.seed): r.checksum for r in runs},
    }
    (group_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def run(train_args: list[str], group: str, jobs: int) -> int:
    rest, seeds = split_seeds(train_args)
    group_dir = OUTPUT_ROOT / group
    print(f"{group}: {len(seeds)} seeds, {jobs} at a time", flush=True)

    failures: list[str] = []
    pending = list(seeds)
    while pending:
        batch, pending = pending[:jobs], pending[jobs:]
        procs = []
        for seed in batch:
            cmd = [PY, str(PROJECT_ROOT / "scripts" / "train.py"), *rest,
                   "--seeds", seed, "--quiet", "--run-group", group]
            log = group_dir / f"seed{seed}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            procs.append((seed, subprocess.Popen(
                cmd, stdout=log.open("w"), stderr=subprocess.STDOUT), log))
        for seed, proc, log in procs:
            if proc.wait() != 0:
                failures.append(seed)
                print(f"  seed {seed} FAILED (see {log})", flush=True)
            else:
                print(f"  seed {seed} done", flush=True)
                log.unlink(missing_ok=True)

    if failures:
        print(f"\n{len(failures)} seed(s) failed: {' '.join(failures)}")
        return 1
    payload = rebuild_summary(group_dir)
    n = payload["summary"]["mae"]["n"]
    print(f"\nrebuilt {group_dir/'summary.json'} from {n} seeds")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=2, help="concurrent seeds (memory-bound)")
    ap.add_argument("--run-group", required=True)
    ap.add_argument("--summarise-only", action="store_true",
                    help="rebuild summary.json from what is already on disk")
    known, train_args = ap.parse_known_args()
    if known.summarise_only:
        rebuild_summary(OUTPUT_ROOT / known.run_group)
        print(f"rebuilt {OUTPUT_ROOT / known.run_group / 'summary.json'}")
        return
    if known.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    raise SystemExit(run(train_args, known.run_group, known.jobs))


if __name__ == "__main__":
    main()
