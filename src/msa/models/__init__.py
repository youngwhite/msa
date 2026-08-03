"""Importing this package registers every model with `msa.registry`."""

from .almt import ALMT
from .base import MSAModel
from .bert import TextOnlyBert
from .bert_mag import BertMAG
from .cenet import CENet
from .dlf import DisentangledLanguageFocused
from .dpdf_lq import DualPathDynamicFusion
from .graph_mfn import GraphMemoryFusionNetwork
from .lf_lstm import LateFusionLSTM
from .lmf import LowRankFusion
from .mctn import MultimodalCyclicTranslationNetwork
from .mfm import MultimodalFactorizationModel
from .mfn import MemoryFusionNetwork
from .misa import MISA
from .mmim import MMIM
from .mult import MultimodalTransformer
from .self_mm import SelfMM
from .simple import EarlyFusionLSTM, LateFusionDNN
from .tetfn import TETFN
from .tfn import TensorFusionNetwork

__all__ = [
    "MSAModel",
    "DisentangledLanguageFocused",
    "DualPathDynamicFusion",
    "BertMAG",
    "ALMT",
    "CENet",
    "MISA",
    "EarlyFusionLSTM",
    "GraphMemoryFusionNetwork",
    "LateFusionDNN",
    "LateFusionLSTM",
    "LowRankFusion",
    "MultimodalCyclicTranslationNetwork",
    "MultimodalFactorizationModel",
    "MemoryFusionNetwork",
    "MultimodalTransformer",
    "SelfMM",
    "MMIM",
    "TETFN",
    "TensorFusionNetwork",
    "TextOnlyBert",
]
