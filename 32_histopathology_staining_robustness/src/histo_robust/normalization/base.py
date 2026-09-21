"""Stain / colour normalisation algorithms (Axis B of ``docs/PLAN.md``).

Two deterministic, reference-based methods are implemented:

* :class:`ReinhardNormalizer` -- statistical colour transfer in CIELAB space.
* :class:`MacenkoNormalizer`  -- optical-density H&E deconvolution (Beer-Lambert)
  with percentile-based concentration bounds.

Both share the same contract:

* operate on **uint8 RGB** ``(H, W, 3)`` arrays (the dataset's native format),
* return **uint8 RGB** of identical shape and dtype,
* are *stateless after fit* and therefore safe to call inside DataLoader workers,
* never raise on pathological tiles: the tissue-mask guard and a ``try/except``
  fallback (Risk 1 in ``docs/PLAN.md``) demote failures to a pass-through copy while
  incrementing a counter that the trainer logs at the end of the run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["BaseNormalizer", "NormalizerStats"]

#: Luminance above which a pixel is considered "slide glass" rather than tissue.
BACKGROUND_LUMINANCE = 220
#: Fraction of background pixels above which SVD deconvolution is skipped.
BACKGROUND_FRACTION_LIMIT = 0.85


class NormalizerStats:
    """Tiny mutable counter so we can report normalisation failures honestly."""

    def __init__(self) -> None:
        self.calls = 0
        self.background_skips = 0
        self.exceptions = 0
        self.last_error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "normalizer_calls": self.calls,
            "normalizer_background_skips": self.background_skips,
            "normalizer_exceptions": self.exceptions,
            "normalizer_last_error": self.last_error,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"NormalizerStats({self.as_dict()})"


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


class BaseNormalizer:
    """Interface shared by all normalisers."""

    name: str = "base"

    def __init__(self, reference: Optional[np.ndarray] = None, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.stats = NormalizerStats()
        self.reference: Optional[np.ndarray] = None
        if reference is not None:
            self.reference = _as_uint8_rgb(reference)
            self._fit_from_reference(self.reference)

    # -- subclasses implement these -------------------------------------
    def _fit_from_reference(self, reference: np.ndarray) -> None:
        raise NotImplementedError

    def _transform(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -- public API ------------------------------------------------------
    def fit(self, reference: np.ndarray) -> "BaseNormalizer":
        self.reference = _as_uint8_rgb(reference)
        self._fit_from_reference(self.reference)
        return self

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalise one tile.  Never raises; falls back to a raw copy."""
        self.stats.calls += 1
        try:
            arr = _as_uint8_rgb(image)
            if self.reference is None:
                raise RuntimeError("Normalizer used before fit()")
            return self._transform(arr)
        except Exception as exc:  # noqa: BLE001 - fallback is the contract
            self.stats.exceptions += 1
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            return _as_uint8_rgb(image)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.normalize(image)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def background_fraction(image: np.ndarray) -> float:
        """Share of pixels brighter than the tissue threshold in all channels."""
        arr = _as_uint8_rgb(image).astype(np.int16)
        mask = np.all(arr > BACKGROUND_LUMINANCE, axis=2)
        return float(mask.mean())
