"""Importing this package registers every model with `msa.registry`."""

from .base import MSAModel
from .bert import TextOnlyBert
from .graph_mfn import GraphMemoryFusionNetwork
from .lf_lstm import LateFusionLSTM
from .lmf import LowRankFusion
from .mfn import MemoryFusionNetwork
from .misa import MISA
from .mult import MultimodalTransformer
from .self_mm import SelfMM
from .simple import EarlyFusionLSTM, LateFusionDNN
from .tfn import TensorFusionNetwork

__all__ = [
    "MSAModel",
    "MISA",
    "EarlyFusionLSTM",
    "GraphMemoryFusionNetwork",
    "LateFusionDNN",
    "LateFusionLSTM",
    "LowRankFusion",
    "MemoryFusionNetwork",
    "MultimodalTransformer",
    "SelfMM",
    "TensorFusionNetwork",
    "TextOnlyBert",
]
