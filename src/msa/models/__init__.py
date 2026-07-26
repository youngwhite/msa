"""Importing this package registers every model with `msa.registry`."""

from .base import MSAModel
from .bert import TextOnlyBert
from .lf_lstm import LateFusionLSTM
from .lmf import LowRankFusion
from .mfn import MemoryFusionNetwork
from .mult import MultimodalTransformer
from .tfn import TensorFusionNetwork

__all__ = [
    "MSAModel",
    "LateFusionLSTM",
    "LowRankFusion",
    "MemoryFusionNetwork",
    "MultimodalTransformer",
    "TensorFusionNetwork",
    "TextOnlyBert",
]
