"""Multimodal sentiment analysis research code."""

import os

# Set before torch creates a CUDA context, otherwise deterministic cuBLAS/cuDNN
# kernels are unavailable. Keep this above any torch import in this package.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _pin_git_revision() -> None:
    """Record the repository state now, before any training can start.

    What runs is what Python imported at startup. Capturing this at save time
    instead attributes results to whatever commit happens to be checked out when
    a seed finishes — which has already mislabelled a ten-seed sweep here.
    Failures are ignored: provenance must never stop a run.
    """
    try:
        from .repro import snapshot_git_revision

        snapshot_git_revision()
    except Exception:  # noqa: BLE001 - provenance is not worth crashing over
        pass


_pin_git_revision()

__all__ = ["config", "data", "device", "metrics", "models", "repro"]
