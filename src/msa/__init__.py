"""Multimodal sentiment analysis research code."""

import os

# Set before torch creates a CUDA context, otherwise deterministic cuBLAS/cuDNN
# kernels are unavailable. Keep this above any torch import in this package.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__all__ = ["config", "data", "device", "metrics", "models", "repro"]
