"""Run MMSA's own code as a reference, and keep the per-seed numbers.

Why this exists at all: comparing against MMSA's published table compares against
a single value with no variance, and #mmsa-all-eleven found that table
unreachable by MMSA's own code on 9 of 11 models. A reference that carries
variance — and, better, that carries the *same seeds* ours do — turns the
comparison from "point vs distribution" into a paired one.

Two subcommands:

    run <model> <seeds...>   run MMSA on those seeds, writing its logs under
                             --out (default ./mmsa_runs, git-ignored)
    collect                  parse `Result for seed N:` out of those logs and
                             write docs/mmsa_code_runs_mosi.json

`collect` is separate on purpose. MMSA's own CSV rounds to two decimals after
multiplying by 100 and summarises with np.std (ddof=0), while this project
reports ddof=1 throughout — deriving our reference from that CSV mixed two
conventions in one variance sum. The per-seed lines in the log carry four
decimals and no convention at all, so they are the thing worth keeping.

Nothing inside MMSA is edited. The accommodations are all outside the algorithm
and are listed in the docstrings below where they are applied; the same list is
in docs/investigations.md#mmsa-all-eleven.

Environment: MMSA's model files cannot import under transformers 5, so `run`
needs a transformers 4.x tree on sys.path ahead of ours, plus a handful of small
packages MMSA imports (easydict, einops, pynvml, and for CENET the real
pytorch_transformers). `scripts/setup_mmsa_reference.sh` builds all of it and
leaves the directory at /workspace/mmsa_env/shim, which is where MMSA_SHIM
defaults to; point MMSA_SHIM elsewhere to override. `collect` needs none of this
— it only reads text.

`collect` also overwrites docs/mmsa_code_runs_mosi.json, and the numbers in
there were produced on the RTX 5070 Ti machine. They do not reproduce on another
one — MMSA's code is subject to the same machine dependence ours is
(docs/investigations.md#cross-machine-hash), so rerunning part of the reference
somewhere else and collecting it would silently mix two machines into one
distribution.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MMSA_SRC = Path("/workspace/MMSA/src")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = PROJECT_ROOT / "datasets" / "CMU-MOSI" / "Processed"
DEFAULT_OUT = PROJECT_ROOT / "mmsa_runs"
RUNS_PATH = PROJECT_ROOT / "docs" / "mmsa_code_runs_mosi.json"

#: Which pickle each model's config expects (MMSA's `need_data_aligned`).
FEATURES_FOR = {
    "tfn": "unaligned", "lmf": "unaligned", "lf_dnn": "unaligned",
    "ef_lstm": "aligned", "mfn": "aligned", "graph_mfn": "aligned",
    "mult": "unaligned", "misa": "unaligned", "self_mm": "unaligned",
    "cenet": "unaligned", "tetfn": "aligned", "bert_mag": "aligned",
    "mmim": "unaligned", "almt": "unaligned",
    # MMSA's published MOSI table has no row for these two, so its own code is
    # the only reference they can have. Both declare need_data_aligned.
    "mctn": "aligned", "mfm": "aligned",
}
#: MMSA's metric names -> ours. Its Acc-2/F1 are already fractions; MAE and Corr
#: are too. No rescaling happens anywhere in this file.
METRIC_KEYS = {
    "MAE": "mae", "Corr": "corr",
    "Non0_acc_2": "acc2_non0", "Non0_F1_score": "f1_non0",
    "Has0_acc_2": "acc2_has0", "Has0_F1_score": "f1_has0",
    "Mult_acc_5": "acc5", "Mult_acc_7": "acc7",
}
RESULT_LINE = re.compile(r"Result for seed (\d+): (\{.*\})")


def run(model: str, seeds: list[int], out_root: Path) -> None:
    """Run MMSA's own model on our MOSI pickles, its own default config."""
    import torch

    # Device plumbing only: run.py calls set_device unconditionally, which fails
    # on a CPU-only torch. With a CUDA build present, leave it alone — MMSA's own
    # RNN-on-gpu0 workaround depends on it.
    if not torch.cuda.is_available():
        torch.cuda.set_device = lambda *a, **k: None

    # torch >= 2.9 dropped ReduceLROnPlateau's `verbose`, which MMSA still
    # passes. The argument only ever controlled logging.
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau

    class PlateauCompat(plateau):
        def __init__(self, *a, verbose=None, **k):
            super().__init__(*a, **k)

    torch.optim.lr_scheduler.ReduceLROnPlateau = PlateauCompat

    sys.path.insert(0, str(MMSA_SRC))
    from MMSA import MMSA_run

    features = DATASETS / f"{FEATURES_FOR[model]}_50.pkl"
    out = out_root / model
    out.mkdir(parents=True, exist_ok=True)
    print(f"MMSA {model}/mosi on {features.name}, seeds {seeds}", flush=True)
    MMSA_run(
        model_name=model,
        dataset_name="mosi",
        # featurePath goes through the public config argument because
        # args.update(config) runs after the path is joined. Same pickle our own
        # runs read, byte for byte.
        config={"featurePath": str(features)},
        seeds=seeds,
        gpu_ids=[0] if torch.cuda.is_available() else [],
        num_workers=0,
        verbose_level=1,
        model_save_dir=str(out / "models"),
        res_save_dir=str(out / "results"),
        log_dir=str(out / "logs"),
    )
    # Checkpoints go, logs stay: each BERT-sized model is ~420MB and eleven of
    # them once filled the disk mid-batch. The runs are seeded and repeatable,
    # and the per-seed numbers live in the log. Same rule as our own best.pt.
    freed = sum(c.stat().st_size for c in out.rglob("*.pth"))
    for ckpt in out.rglob("*.pth"):
        ckpt.unlink()
    if freed:
        print(f"removed {freed / 1e6:.0f}MB of checkpoints", flush=True)


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    """Per-seed metrics out of one MMSA log.

    A seed may appear more than once when a model was run in several batches.
    Identical repeats are collapsed; conflicting ones are an error, because
    silently keeping one of two different numbers is how a reference stops
    meaning anything.
    """
    runs: dict[int, dict[str, float]] = {}
    for line in path.read_text().splitlines():
        match = RESULT_LINE.search(line)
        if not match:
            continue
        seed = int(match.group(1))
        # The log holds a repr with np.float64(...) wrappers; strip them rather
        # than eval the line with numpy in scope.
        literal = re.sub(r"np\.float64\(([^)]*)\)", r"\1", match.group(2))
        raw = ast.literal_eval(literal)
        metrics = {ours: round(float(raw[theirs]), 6)
                   for theirs, ours in METRIC_KEYS.items() if theirs in raw}
        if seed in runs and runs[seed] != metrics:
            raise SystemExit(
                f"{path}: seed {seed} appears twice with different results.\n"
                f"  {runs[seed]}\n  {metrics}\n"
                f"Two different runs are hiding in one log; split them before "
                f"collecting.")
        runs[seed] = metrics
    return runs


def collect(out_root: Path, extra_logs: list[Path]) -> None:
    """Parse every log under out_root (plus any given explicitly) into one file."""
    logs = sorted(out_root.glob("*/logs/*.log")) + list(extra_logs)
    if not logs:
        raise SystemExit(f"no logs under {out_root}")
    existing = json.loads(RUNS_PATH.read_text())["models"] if RUNS_PATH.exists() else {}
    models = dict(existing)
    for log in logs:
        model = log.parent.parent.name
        if model not in FEATURES_FOR:
            print(f"  skip {log}: {model!r} is not a model we compare against")
            continue
        runs = {str(seed): metrics for seed, metrics in sorted(parse_log(log).items())}
        if not runs:
            print(f"  skip {log}: no per-seed result lines")
            continue
        merged = dict(models.get(model, {}).get("runs", {}))
        for seed, metrics in runs.items():
            if seed in merged and merged[seed] != metrics:
                raise SystemExit(
                    f"{model} seed {seed}: {log} disagrees with what is already "
                    f"stored.\n  stored: {merged[seed]}\n  log:    {metrics}")
            merged[seed] = metrics
        models[model] = {
            "data_setting": FEATURES_FOR[model],
            "seeds": sorted(int(s) for s in merged),
            "n": len(merged),
            "runs": dict(sorted(merged.items(), key=lambda kv: int(kv[0]))),
        }
        print(f"  {model:10s} {len(merged)} seed(s): "
              f"{sorted(int(s) for s in merged)}")
    payload = {
        "source": "our own runs of MMSA's unmodified model code (THUIAR, MIT)",
        "why": (
            "MMSA's published table states single values with no variance, and "
            "#mmsa-all-eleven found it unreachable by MMSA's own code on 9 of 11 "
            "models. A reference we run ourselves carries variance and shares our "
            "seeds, which makes the comparison paired rather than point-vs-"
            "distribution."),
        "how": (
            "scripts/mmsa_reference.py run <model> <seeds>, then collect. Nothing "
            "in MMSA is edited; the accommodations (set_device, the scheduler's "
            "dropped verbose kwarg, featurePath through the public config "
            "argument, and a transformers 4.x tree on sys.path) are all outside "
            "the algorithm. Same MOSI pickles our runs read, MMSA's own default "
            "hyper-parameters."),
        "caveat": (
            "These are MMSA's numbers for MMSA's implementation. Where MMSA "
            "departs from the original paper, this reference inherits the "
            "departure — see rule 5 in CLAUDE.md. Each model still needs its own "
            "paper comparison in docs/paper_reference_mosi.json."),
        "units": "fractions, not percentages; MAE and Corr in metric units",
        "precision": (
            "four decimals, as MMSA's logger writes them. Its CSV instead rounds "
            "to two decimals after x100 and summarises with np.std (ddof=0); this "
            "project uses ddof=1 throughout, so means and spreads are derived "
            "from these per-seed values rather than read off that CSV."),
        "mmsa_commit": mmsa_commit(),
        "models": dict(sorted(models.items())),
    }
    RUNS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {RUNS_PATH.relative_to(PROJECT_ROOT)} "
          f"({len(models)} models, {sum(m['n'] for m in models.values())} runs)")


def mmsa_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(MMSA_SRC.parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"where MMSA writes logs (default {DEFAULT_OUT})")
    sub = ap.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run", help="run MMSA on some seeds")
    run_cmd.add_argument("model", choices=sorted(FEATURES_FOR))
    run_cmd.add_argument("seeds", nargs="+", type=int)
    collect_cmd = sub.add_parser("collect", help="parse logs into docs/")
    collect_cmd.add_argument("--also", type=Path, nargs="*", default=[],
                             help="extra log files outside --out")
    args = ap.parse_args()

    if args.command == "run":
        shim = os.environ.get("MMSA_SHIM")
        if shim:
            sys.path.insert(0, shim)
        run(args.model, args.seeds, args.out)
    else:
        collect(args.out, args.also)


if __name__ == "__main__":
    main()
