"""Augmentation policy factory (Axis C).

Four policies, matching the ablation matrix in ``docs/PLAN.md`` section 3:

===============  ==========================================================
policy key       content
===============  ==========================================================
``none``         deterministic (evaluation transform only)
``aug_geo``      flips + random 90-degree rotations
``aug_stain``    HED stain jitter (+ brightness / hue-saturation)
``aug_combined`` geometric + stain jitter
===============  ==========================================================

Policies are applied *after* stain normalisation and *before* tensor conversion
(see :mod:`histo_robust.data.dataset`), so the augmentation never fights the
normaliser for control of the colour statistics.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from .geometric import GeometricAugmenter
from .stain_jitter import HEDStainJitter

logger = logging.getLogger(__name__)

__all__ = ["AUGMENTATION_CHOICES", "AugmentationPolicy", "build_augmentation"]

AUGMENTATION_CHOICES = ("none", "aug_geo", "aug_stain", "aug_combined")


class AugmentationPolicy:
    """A tiny ordered list of callables with a human-readable name.

    Kept framework-free on purpose: the same object is used by the offline
    preprocessing cache, by the training DataLoader and by the visual-diagnostics
    dump, which removes any chance of train/eval transform drift.
    """

    def __init__(self, name: str, transforms: Optional[List[Any]] = None) -> None:
        self.name = name
        self.transforms = list(transforms or [])

    def __call__(self, image):
        out = image
        for tf in self.transforms:
            out = tf(out)
        return out

    def __len__(self) -> int:
        return len(self.transforms)

    def describe(self) -> Dict[str, Any]:
        detail: List[Dict[str, Any]] = []
        for tf in self.transforms:
            if hasattr(tf, "config"):
                detail.append(tf.config())
            else:
                detail.append({"type": type(tf).__name__})
        return {"name": self.name, "n_transforms": len(self.transforms), "transforms": detail}


def build_augmentation(
    policy: str,
    cfg: Optional[Dict[str, Any]] = None,
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
) -> AugmentationPolicy:
    """Build an :class:`AugmentationPolicy` from its config key.

    ``cfg`` is the ``augmentation`` block of the experiment config; unknown keys
    raise so a typo can never silently disable an ablation cell.
    """
    cfg = dict(cfg or {})
    key = (policy or "none").strip().lower()
    rng = rng or random.Random(seed)

    if key in {"none", "identity", ""}:
        return AugmentationPolicy("none", [])

    if key == "aug_geo":
        geo_cfg = dict(cfg.get("geometric", {}))
        return AugmentationPolicy("aug_geo", [GeometricAugmenter(rng=rng, **geo_cfg)])

    if key == "aug_stain":
        stain_cfg = dict(cfg.get("stain", {}))
        return AugmentationPolicy("aug_stain", [HEDStainJitter(rng=rng, **stain_cfg)])

    if key == "aug_combined":
        geo_cfg = dict(cfg.get("geometric", {}))
        stain_cfg = dict(cfg.get("stain", {}))
        return AugmentationPolicy(
            "aug_combined",
            [GeometricAugmenter(rng=rng, **geo_cfg), HEDStainJitter(rng=rng, **stain_cfg)],
        )

    raise ValueError(
        f"Unknown augmentation policy '{policy}'. Expected one of {AUGMENTATION_CHOICES}"
    )
