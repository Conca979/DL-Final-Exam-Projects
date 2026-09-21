"""Model backbones and classification head (Axis E)."""

from .backbones import (
    BACKBONE_CHOICES,
    BACKBONE_REGISTRY,
    BackboneSpec,
    available_backbones,
    estimate_backbone_vram_note,
    load_hf_backbone,
    load_timm_backbone,
    resolve_local_weights,
)
from .classifier import StainRobustClassifier, build_classifier

__all__ = [
    "BACKBONE_REGISTRY",
    "BACKBONE_CHOICES",
    "BackboneSpec",
    "available_backbones",
    "load_timm_backbone",
    "load_hf_backbone",
    "resolve_local_weights",
    "estimate_backbone_vram_note",
    "StainRobustClassifier",
    "build_classifier",
]
