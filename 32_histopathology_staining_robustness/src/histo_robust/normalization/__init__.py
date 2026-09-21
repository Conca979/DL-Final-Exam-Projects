"""Stain / colour normalisation (Axis B)."""

from .base import BaseNormalizer, NormalizerStats
from .macenko import (
    MacenkoNormalizer,
    canonical_he_stain_matrix,
    estimate_stain_matrix,
    stain_matrix_separation_deg,
)
from .normalizer_factory import (
    NORMALIZATION_CHOICES,
    build_normalizer,
    load_image_rgb,
    resolve_reference_image,
    synthetic_reference_image,
)
from .reinhard import ReinhardNormalizer

__all__ = [
    "BaseNormalizer",
    "NormalizerStats",
    "ReinhardNormalizer",
    "MacenkoNormalizer",
    "estimate_stain_matrix",
    "canonical_he_stain_matrix",
    "stain_matrix_separation_deg",
    "build_normalizer",
    "resolve_reference_image",
    "load_image_rgb",
    "synthetic_reference_image",
    "NORMALIZATION_CHOICES",
]
