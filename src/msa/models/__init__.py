"""Importing this package registers every model with `msa.registry`."""

from .base import MSAModel
from .lf_lstm import LateFusionLSTM

__all__ = ["MSAModel", "LateFusionLSTM"]
