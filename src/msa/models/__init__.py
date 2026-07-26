"""Importing this package registers every model with `msa.registry`."""

from .base import MSAModel
from .bert import TextOnlyBert
from .lf_lstm import LateFusionLSTM
from .lmf import LowRankFusion
from .mfn import MemoryFusionNetwork
from .misa import MISA
from .mult import MultimodalTransformer
from .self_mm import SelfMM
from .tfn import TensorFusionNetwork

__all__ = [
    "MSAModel",
    "MISA",
    "LateFusionLSTM",
    "LowRankFusion",
    "MemoryFusionNetwork",
    "MultimodalTransformer",
    "SelfMM",
    "TensorFusionNetwork",
    "TextOnlyBert",
]
