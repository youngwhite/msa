"""Everything needed to make a run bit-for-bit repeatable on the same device.

Same seed + same device + same code => identical metrics. Results are *not*
promised to match across device types (CUDA vs MPS vs CPU reduce in different
orders); each device is internally reproducible.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# cuBLAS picks a workspace at CUDA-context creation; without a fixed one, some
# GEMMs (and cuDNN RNN backward) are run-to-run nondeterministic. Must be set
# before the first CUDA call, which is why msa/__init__.py sets it at import.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_cublas_workspace() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)


def set_seed(seed: int, deterministic: bool = True, warn_only: bool = False) -> None:
    """Seed every RNG we touch and pin the backends to deterministic kernels.

    `deterministic=False` restores the fast nondeterministic kernels; use it only
    when an op has no deterministic implementation on the current device (MPS
    coverage is thinner than CUDA's), and record it in the run's metadata.
    """
    configure_cublas_workspace()
    # Only affects interpreters started *after* this point (i.e. spawn-mode
    # dataloader workers); the current process fixed its hash seed at startup.
    # Nothing here depends on hash ordering — it is set for the sake of any
    # subprocess that might.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give each worker a seed derived from the
    parent's generator, so num_workers>0 stays reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


#: Captured once, at import — see `git_revision`.
_GIT_AT_START: dict[str, object] | None = None


def read_git_revision() -> dict[str, object]:
    """Read the repository state *now*.

    `dirty` reflects the **source** only. Two categories are deliberately
    excluded:

    * Untracked files — scratch notes, editor state, the ignored dataset
      directory. They do not change what the committed code does.
    * Anything under `outputs/`. Results are committed in this repo, so a run
      that rewrites its own group's `result.json` makes the tree dirty *by
      running*. Counting that would mark every rerun after its first seed as
      unauditable, which is both false and self-inflicted: the code was clean,
      only the run's own products changed.

    Returns None values outside a git checkout rather than failing.
    """
    repo = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    if head is None:
        return {"commit": None, "dirty": None}
    status = git("status", "--porcelain", "--untracked-files=no")
    if status is None:
        return {"commit": head.strip(), "dirty": None}
    # Porcelain lines are "XY path"; the path starts at column 3.
    changed = [line[3:] for line in status.splitlines() if line.strip()]
    source_changed = [p for p in changed if not p.lstrip('"').startswith("outputs/")]
    return {"commit": head.strip(), "dirty": bool(source_changed)}


def snapshot_git_revision() -> dict[str, object]:
    """The repository state as of process start — the code that actually ran.

    Python imports its modules once, at startup; everything after that executes
    from memory. So a commit made *during* a long run does not change what is
    running, and stamping results with the HEAD at save time attributes them to
    code that never executed. This has already happened here: one Graph-MFN
    sweep recorded three different commits across ten seeds and ran one of them.

    Cached on first call and pinned by `msa/__init__.py` at import.
    """
    global _GIT_AT_START
    if _GIT_AT_START is None:
        _GIT_AT_START = read_git_revision()
    return _GIT_AT_START


def git_revision() -> dict[str, object]:
    """Provenance for a stored result.

    `commit`/`dirty` describe the code that ran. If the repository moved while
    the run was in flight, `at_save` records where it ended up — the results
    stay valid (they came from the start state), but someone was editing during
    a run, which `scripts/verify_runs.py` reports.
    """
    start = dict(snapshot_git_revision())
    end = read_git_revision()
    if end != start:
        start["at_save"] = end
    return start


def collect_env(device: torch.device | None = None) -> dict[str, object]:
    """Provenance to store next to results — what actually produced the numbers."""
    from .device import describe_device

    env: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
        "cpu_threads": torch.get_num_threads(),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "git": git_revision(),
    }
    if device is not None:
        env["device"] = str(device)
        env["device_description"] = describe_device(device)
    return env
