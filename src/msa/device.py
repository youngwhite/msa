"""Device selection that works the same on CUDA, Apple Silicon (MPS) and CPU."""

from __future__ import annotations

import torch

DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(spec: str = "auto") -> torch.device:
    """Turn a user-facing device string into a torch.device.

    `auto` prefers CUDA, then Apple Silicon MPS, then CPU. An explicit choice
    that is unavailable is an error rather than a silent fallback to CPU — a
    run that quietly drops to CPU is a run whose timings mean nothing.
    """
    spec = spec.lower()
    if spec not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {DEVICE_CHOICES}, got {spec!r}")

    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    if spec == "mps" and not mps_available():
        raise RuntimeError("--device mps requested but torch.backends.mps.is_available() is False")
    return torch.device(spec)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        major, minor = torch.cuda.get_device_capability(idx)
        return f"cuda:{idx} {torch.cuda.get_device_name(idx)} (sm_{major}{minor})"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return f"cpu ({torch.get_num_threads()} threads)"


def configure_threads(num_threads: int | None) -> int:
    """Pin CPU intra-op parallelism and return the effective thread count.

    CPU reductions are split across threads, so the thread count changes the
    summation order and therefore the last bits of every result: 1, 4 and 12
    threads produce three different (each internally reproducible) runs. Pin it
    to a fixed number whenever results must match across machines.
    """
    if num_threads is not None:
        if num_threads < 1:
            raise ValueError("--num-threads must be >= 1")
        torch.set_num_threads(num_threads)
    return torch.get_num_threads()


def supports_pin_memory(device: torch.device) -> bool:
    """Pinned host memory only helps (and only exists) for CUDA transfers."""
    return device.type == "cuda"
