"""Importing this package registers every contrastive loss with the registry."""

from .contrastive import (
    AlignmentUniformity,
    ArcFace,
    BarlowTwins,
    ContrastiveLoss,
    ContrastivePredictiveCoding,
    DecoupledContrastive,
    HardNegativeContrastive,
    InfoNCE,
    MaxMargin,
    NTXent,
    RankNContrast,
    SimSiam,
    SupervisedContrastive,
    TripletMargin,
    VICReg,
)
from .registry import available_losses, build_loss, get_loss_class, register_loss

__all__ = [
    "ContrastiveLoss",
    "AlignmentUniformity", "ArcFace", "BarlowTwins", "ContrastivePredictiveCoding",
    "DecoupledContrastive", "HardNegativeContrastive", "InfoNCE", "MaxMargin", "NTXent",
    "RankNContrast", "SimSiam", "SupervisedContrastive", "TripletMargin", "VICReg",
    "available_losses", "build_loss", "get_loss_class", "register_loss",
]
