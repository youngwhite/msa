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


def git_revision() -> dict[str, object]:
    """Which commit produced a number, and whether the tree was dirty at the time.

    A result recorded without this is unauditable: the code it came from cannot
    be recovered. Returns None values outside a git checkout rather than failing.
    """
    repo = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if head.returncode != 0:
            return {"commit": None, "dirty": None}
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return {
            "commit": head.stdout.strip(),
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


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
