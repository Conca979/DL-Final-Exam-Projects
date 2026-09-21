"""Augmentation policies (Axis C)."""

from .geometric import CenterCrop, GeometricAugmenter
from .policy_factory import AUGMENTATION_CHOICES, AugmentationPolicy, build_augmentation
from .stain_jitter import HEDStainJitter, ruifrok_hed_matrix

__all__ = [
    "GeometricAugmenter",
    "CenterCrop",
    "HEDStainJitter",
    "ruifrok_hed_matrix",
    "AugmentationPolicy",
    "build_augmentation",
    "AUGMENTATION_CHOICES",
]
