"""Run an author's own code as a reference, and keep the per-seed numbers.

`mmsa_reference.py` does this for the fourteen models MMSA implements. Methods
that MMSA never included -- everything from 2024 onward -- have only their
authors' releases, so this is the parallel runner for those. Kept separate
rather than folded in: MMSA is one codebase invoked through one API, while each
author release has its own entry point, and mixing them would make the stable
path hostage to the unstable one.

    python scripts/author_reference.py run dpdf_lq 42 43 44 45 46 47 48 49 50 51
    python scripts/author_reference.py collect

Two disciplines are carried over verbatim from `mmsa_reference.py`, because
both were paid for:

* **Every run records the machine that produced it.** MMSA's code turned out to
  be as machine-bound as ours -- tfn seed 42 moves 0.0136 MAE between an RTX
  5070 Ti and a 5080 -- so a reference whose provenance is unrecorded cannot be
  paired with anything.
* **`collect` refuses to merge across machines.** A per-seed value conflict is
  caught anyway, but two machines can disagree *without* colliding: one
  contributes seeds 42-46 and the other 47-51, the merge succeeds, and the
  resulting distribution exists on no machine at all.

The results land in `docs/author_code_runs_mosi.json`, deliberately separate
from MMSA's file. Two reference sources with different provenance do not belong
in one table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "author_runs"          # git-ignored, like mmsa_runs
RUNS_PATH = PROJECT_ROOT / "docs" / "author_code_runs_mosi.json"
DATASETS = PROJECT_ROOT / "datasets" / "CMU-MOSI" / "Processed"

#: One entry per method. `repo` is where its release is checked out, `command`
#: is how that release trains one seed, and `metrics` maps its own metric names
#: onto ours. Nothing inside a release is edited; anything that has to be
#: adjusted goes here, in the open.
METHODS = {
    "dpdf_lq": {
        "repo": Path("/workspace/DPDF-LQ"),
        "features": "aligned_50.pkl",
        "config": "configs/mosi.yaml",
        "command": ["python", "train.py", "--config_file", "{config}", "--seed", "{seed}"],
        "paper": "Dual-Path Dynamic Fusion with Learnable Query, EMNLP 2025",
        "licence": "MIT",
    },
}
#: Their metric names -> ours. All are fractions on both sides; nothing is
#: rescaled anywhere in this file.
METRIC_KEYS = {
    "MAE": "mae", "Corr": "corr",
    "Non0_acc_2": "acc2_non0", "Non0_F1_score": "f1_non0",
    "Has0_acc_2": "acc2_has0", "Has0_F1_score": "f1_has0",
    "Mult_acc_5": "acc5", "Mult_acc_7": "acc7",
}
#: Matches both plain floats and the numpy wrappers the release prints,
#: e.g. 'MAE': np.float32(0.7226) and 'Corr': np.float64(0.7912).
RESULT_LINE = re.compile(r"'(\w+)':\s*(?:np\.\w+\()?(-?\d+\.?\d*)")


def machine_fingerprint() -> dict[str, object]:
    import torch

    from msa.repro import collect_env

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = collect_env(device)
    return {key: env[key] for key in
            ("device_description", "cpu", "cpu_capability", "torch", "python")}


def run(method: str, seeds: list[int], out_root: Path) -> None:
    spec = METHODS[method]
    repo = spec["repo"]
    if not repo.exists():
        raise SystemExit(f"no checkout at {repo}; clone the release first")

    out = out_root / method
    out.mkdir(parents=True, exist_ok=True)
    # Point the release at our pickles through its own config rather than
    # editing its loader: same tensors every group in this repository reads.
    config = json_safe_config(repo, spec, out)

    environment = dict(os.environ)
    shim = environment.get("MMSA_SHIM", "/workspace/mmsa_env/shim")
    environment["PYTHONPATH"] = os.pathsep.join(
        [shim, str(repo), environment.get("PYTHONPATH", "")])

    for seed in seeds:
        command = [part.format(config=config, seed=seed) for part in spec["command"]]
        command[0] = sys.executable
        print(f"=== {method} seed {seed}", flush=True)
        log = out / f"seed{seed}.log"
        with log.open("w") as handle:
            subprocess.run(command, cwd=repo, env=environment,
                           stdout=handle, stderr=subprocess.STDOUT, check=False)
    (out / "machine.json").write_text(
        json.dumps(machine_fingerprint(), indent=2) + "\n")


def json_safe_config(repo: Path, spec: dict, out: Path) -> str:
    """Copy the release's config with only the data path repointed."""
    import yaml

    original = yaml.safe_load((repo / spec["config"]).read_text())
    original["dataset"]["dataPath"] = str(DATASETS / spec["features"])
    target = out / "config.yaml"
    target.write_text(yaml.safe_dump(original, sort_keys=False))
    return str(target.resolve())


def parse_log(path: Path) -> dict[str, float] | None:
    """The test metrics of the epoch chosen by VALIDATION MAE.

    This cannot be read off the release's own summary lines, and using them
    would silently import a different protocol. DPDF-LQ's `results_recorder`
    keeps two summaries and both select on the test set:

    * `best_results_one_epoch` picks the epoch with the best *test* MAE;
    * `best_results_all_epochs` is worse still -- each metric takes its own best
      epoch, so the Acc-2, MAE and Acc-7 it prints can come from three different
      models.

    Convention 2 here is that selection sees only validation data and the test
    set is touched once, afterwards. So rather than edit the release --
    references are read-only -- the honest number is rebuilt from its own
    per-epoch log: take the epoch with the lowest validation MAE and report that
    epoch's test metrics. Same run, same code, our selection rule.
    """
    epochs: list[tuple[float, dict[str, float]]] = []
    validation: dict[str, float] | None = None
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("Validation Results:"):
            validation = _metrics(line)
        elif line.startswith("Test Results:") and validation is not None:
            test = _metrics(line)
            if "mae" in validation and test:
                epochs.append((validation["mae"], test))
            validation = None
    if not epochs:
        return None
    return min(epochs, key=lambda pair: pair[0])[1]


def _metrics(line: str) -> dict[str, float]:
    found = dict(RESULT_LINE.findall(line))
    return {ours: round(float(found[theirs]), 6)
            for theirs, ours in METRIC_KEYS.items() if theirs in found}


def collect(out_root: Path) -> None:
    payload = json.loads(RUNS_PATH.read_text()) if RUNS_PATH.exists() else {"models": {}}
    models = payload.get("models", {})
    for method in sorted(METHODS):
        directory = out_root / method
        if not directory.exists():
            continue
        machine_file = directory / "machine.json"
        machine = (json.loads(machine_file.read_text()).get("device_description")
                   if machine_file.exists() else None)
        stored = models.get(method, {}).get("machine")
        if stored is not None and machine is not None and stored != machine:
            raise SystemExit(
                f"{method}: stored runs came from {stored!r}, these from {machine!r}.\n"
                f"Rerun this method's whole seed set on one machine rather than "
                f"merging two.")

        runs = dict(models.get(method, {}).get("runs", {}))
        for log in sorted(directory.glob("seed*.log")):
            seed = log.stem.removeprefix("seed")
            metrics = parse_log(log)
            if metrics is None:
                print(f"  skip {log.name}: no metric line found")
                continue
            if seed in runs and runs[seed] != metrics:
                raise SystemExit(
                    f"{method} seed {seed}: {log} disagrees with what is stored.\n"
                    f"  stored: {runs[seed]}\n  log:    {metrics}")
            runs[seed] = metrics
        if not runs:
            continue
        models[method] = {
            "paper": METHODS[method]["paper"],
            "licence": METHODS[method]["licence"],
            "machine": machine,
            "seeds": sorted(int(s) for s in runs),
            "n": len(runs),
            "runs": dict(sorted(runs.items(), key=lambda kv: int(kv[0]))),
        }
        print(f"  {method:10s} {len(runs)} seed(s)")

    RUNS_PATH.write_text(json.dumps({
        "source": "our own runs of each method's unmodified author release",
        "why": ("Methods published after MMSA stopped adding models have no "
                "third-party reference. Running the authors' own code on our "
                "data, our seeds and our machine is the only way to tell an "
                "implementation gap from a protocol gap."),
        "how": ("scripts/author_reference.py run <method> <seeds>, then collect. "
                "Nothing inside a release is edited; the only adjustment is the "
                "dataset path, applied through the release's own config file."),
        "models": dict(sorted(models.items())),
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {RUNS_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("method", choices=sorted(METHODS))
    run_cmd.add_argument("seeds", nargs="+", type=int)
    sub.add_parser("collect")
    args = parser.parse_args()

    if args.command == "run":
        run(args.method, args.seeds, args.out)
    else:
        collect(args.out)


if __name__ == "__main__":
    main()
