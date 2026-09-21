"""Factory / reference-template resolution for stain normalisation.

The reference tile is the single most important shared constant in Axis B: both
Reinhard and Macenko must map every experiment onto the *same* canonical H&E
appearance, otherwise the ablation is not controlled.

Resolution order used by :func:`resolve_reference_image`:

1. explicit ``normalization.reference_path`` from the config (if it exists),
2. ``data/processed/templates/reference_stain.png``,
3. any ``*reference*stain*.{png,tif,tiff,jpg}`` under ``data/`` or the repo root,
4. a deterministic synthetic H&E reference (:func:`synthetic_reference_image`),
   which needs no file at all and keeps Kaggle runs reproducible.

The reference is selected from the **source training cohort only** -- never from
``CRC-VAL-HE-7K`` (see the firewall in ``docs/PLAN.md`` section 4.1).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .base import BaseNormalizer
from .macenko import MacenkoNormalizer
from .reinhard import ReinhardNormalizer

logger = logging.getLogger(__name__)

__all__ = [
    "build_normalizer",
    "resolve_reference_image",
    "synthetic_reference_image",
    "load_image_rgb",
    "NORMALIZATION_CHOICES",
]

NORMALIZATION_CHOICES = ("none", "reinhard", "macenko")

_REFERENCE_PATTERNS: Sequence[str] = (
    "reference_stain.png",
    "reference_stain.tif",
    "reference_stain.tiff",
    "reference_stain.jpg",
)


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load any PIL-readable image as uint8 RGB (TIF/PNG/JPEG all supported)."""
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def synthetic_reference_image(size: int = 224) -> np.ndarray:
    """Deterministic canonical H&E reference built from fixed OD values.

    Two overlapping Gaussian blobs approximate a Hematoxylin-rich nuclear
    region and an Eosin-rich cytoplasmic region.  Because it is generated from
    constants (no file, no download, no RNG), every Kaggle session -- and every
    experiment inside a session -- normalises against exactly the same target.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    #: two blob centres, deterministic
    blobs = (
        (0.35 * size, 0.38 * size, 0.16 * size, 0.9),  # haematoxylin-rich
        (0.62 * size, 0.60 * size, 0.22 * size, 0.8),  # eosin-rich
    )
    h_map = np.zeros((size, size))
    e_map = np.zeros((size, size))
    for cy, cx, sigma, amp in blobs:
        g = amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))
        if amp > 0.85:
            h_map = np.maximum(h_map, g)
        else:
            e_map = np.maximum(e_map, g)
    h_map += 0.25
    e_map += 0.20

    # Canonical H&E OD vectors (Ruifrok & Johnston, 2001), unit-normalised.
    stain = np.array(
        [
            [0.650, 0.072],
            [0.704, 0.990],
            [0.286, 0.105],
        ],
        dtype=np.float64,
    )
    stain = stain / np.linalg.norm(stain, axis=0, keepdims=True)

    concentrations = np.stack([h_map.ravel(), e_map.ravel()], axis=1)
    od = concentrations @ stain.T
    rgb = 256.0 * np.power(10.0, -od) - 1.0
    return np.clip(rgb, 0, 255).astype(np.uint8).reshape(size, size, 3)


def resolve_reference_image(
    reference_path: Optional[str | Path] = None,
    search_roots: Optional[Sequence[str | Path]] = None,
    observed_fraction: float = 0.0,
) -> tuple[np.ndarray, str]:
    """Return ``(reference_uint8_rgb, provenance_string)``.

    ``observed_fraction`` > 0 asks the fallback to emit a loud warning because
    a synthetic reference was substituted for the intended canonical tile.
    """
    if reference_path:
        path = Path(reference_path)
        if path.exists():
            return load_image_rgb(path), f"file:{path}"
        logger.warning(
            "Configured normalization.reference_path '%s' does not exist; "
            "falling back to auto-discovery.",
            path,
        )

    roots = [Path(r) for r in (search_roots or []) if r is not None]
    for root in roots:
        if not root.exists():
            continue
        direct = root / "data" / "processed" / "templates"
        for pattern in _REFERENCE_PATTERNS:
            candidate = direct / pattern
            if candidate.exists():
                return load_image_rgb(candidate), f"file:{candidate}"
        for pattern in ("*reference*stain*.png", "*reference*stain*.tif", "*reference*stain*.tiff"):
            hits = sorted(root.rglob(pattern))
            if hits:
                return load_image_rgb(hits[0]), f"file:{hits[0]}"

    logger.warning(
        "No reference_stain tile found (searched %s). Using the deterministic "
        "synthetic H&E reference; place a source-domain TUM tile at "
        "data/processed/templates/reference_stain.png for the literature-standard setup. "
        "(observed_fraction=%.2f)",
        [str(r) for r in roots],
        observed_fraction,
    )
    return synthetic_reference_image(), "synthetic:canonical_he"


def build_normalizer(
    name: str,
    reference: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[BaseNormalizer]:
    """Instantiate a normaliser by name.

    ``'none'`` (and ``'raw'``) return ``None`` -- the caller treats ``None`` as
    "identity transform", which is also what the cached-preprocessing path uses.
    """
    key = (name or "none").strip().lower()
    params = dict(params or {})

    if key in {"none", "raw", "identity"}:
        return None
    if key == "reinhard":
        return ReinhardNormalizer(reference=reference, **params)
    if key == "macenko":
        return MacenkoNormalizer(reference=reference, **params)
    raise ValueError(
        f"Unknown normalization '{name}'. Expected one of {NORMALIZATION_CHOICES}"
    )
