"""Macenko optical-density stain normalisation (Macenko et al., ISBI 2009).

Pipeline per tile:

1. ``OD = -log10((I + 1) / 256)`` on the RGB intensities (Beer-Lambert).
2. Drop low-density (background) pixels via an OD-norm threshold.
3. SVD -> plane spanned by the two principal stain directions; project onto
   that plane and keep the extreme angular directions -> Hematoxylin/Eosin
   vectors (sign-corrected so H is the darker/more eosinophobic direction).
4. Solve ``C = OD @ pinv(V)`` for stain concentrations.
5. Rescale concentrations to the canonical 99th percentile of the *reference*
   tile and rebuild with the reference stain matrix.

Implementation notes / deviations from torchstain:

* Pure NumPy -- no extra dependency, so the Kaggle image cannot break the run.
* The 99th-percentile is recomputed **per tile** over its own tissue pixels
  (torchstain's ``stain_matrix_target`` path uses a fixed scalar for the whole
  dataset, which does not transfer across scanners).
* ``Risk 1`` guard from ``PLAN.md``: tiles that are >85% slide glass bypass the
  SVD entirely; any residual numerical failure is caught by
  :meth:`BaseNormalizer.normalize` and demoted to a raw copy.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .base import (
    BACKGROUND_FRACTION_LIMIT,
    BaseNormalizer,
    _as_uint8_rgb,
)

logger = logging.getLogger(__name__)

__all__ = ["MacenkoNormalizer", "estimate_stain_matrix"]

#: Below this OD norm a pixel is treated as background / glass.
DEFAULT_OD_THRESHOLD = 0.15
#: Angular percentile used to pick the two extreme stain directions.
DEFAULT_ANGULAR_PERCENTILE = 99.0
#: Concentration percentile mapped onto the reference tile's percentile.
DEFAULT_CONCENTRATION_PERCENTILE = 99.0


def _od_from_rgb(rgb: np.ndarray) -> np.ndarray:
    return -np.log10((rgb.astype(np.float64) + 1.0) / 256.0)


def _rgb_from_od(od: np.ndarray) -> np.ndarray:
    rgb = 256.0 * np.power(10.0, -od) - 1.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def estimate_stain_matrix(
    rgb: np.ndarray,
    od_threshold: float = DEFAULT_OD_THRESHOLD,
    angular_percentile: float = DEFAULT_ANGULAR_PERCENTILE,
) -> np.ndarray:
    """Return a 3x2 unit-norm stain matrix ``[[r, r], [g, g], [b, b]]``.

    Raises ``RuntimeError`` when the tile carries too little optical density to
    resolve two independent stain directions.
    """
    od = _od_from_rgb(_as_uint8_rgb(rgb))
    od_flat = od.reshape(-1, 3)
    od_norm = np.linalg.norm(od_flat, axis=1)
    tissue = od_norm > od_threshold
    if tissue.sum() < 64:
        raise RuntimeError(f"only {int(tissue.sum())} tissue pixels (need >= 64)")

    od_hat = od_flat[tissue] / od_norm[tissue][:, None]

    _, _, vh = np.linalg.svd(od_hat, full_matrices=False)
    plane = vh[:2]  # two principal directions

    projections = od_hat @ plane.T
    phi = np.arctan2(projections[:, 1], projections[:, 0])
    lo = np.percentile(phi, 100.0 - angular_percentile)
    hi = np.percentile(phi, angular_percentile)

    v_lo = plane.T @ np.array([np.cos(lo), np.sin(lo)])
    v_hi = plane.T @ np.array([np.cos(hi), np.sin(hi)])

    # Convention: column 0 = Hematoxylin.  Hue-deconvolution papers order the
    # vectors so that the first has the smaller R/B ratio bound; keeping a
    # deterministic ordering matters because H and E scalars are exchanged
    # otherwise, which would silently corrupt the HED augmentation too.
    if v_lo[0] > v_hi[0]:
        v_lo, v_hi = v_hi, v_lo

    stain = np.stack([v_lo, v_hi], axis=1)
    norms = np.linalg.norm(stain, axis=0, keepdims=True)
    if np.any(norms < 1e-6):
        raise RuntimeError("degenerate stain vector (near-zero norm)")
    stain = stain / norms
    if not np.all(np.isfinite(stain)):
        raise RuntimeError("non-finite stain matrix")
    return stain


class MacenkoNormalizer(BaseNormalizer):
    """Reference-based Macenko normalisation."""

    name = "macenko"

    def __init__(
        self,
        reference: Optional[np.ndarray] = None,
        od_threshold: float = DEFAULT_OD_THRESHOLD,
        angular_percentile: float = DEFAULT_ANGULAR_PERCENTILE,
        concentration_percentile: float = DEFAULT_CONCENTRATION_PERCENTILE,
        beta: float = 0.15,
        alpha: float = 1.0,
        background_fraction_limit: float = BACKGROUND_FRACTION_LIMIT,
        **kwargs,
    ) -> None:
        self.od_threshold = float(od_threshold)
        self.angular_percentile = float(angular_percentile)
        self.concentration_percentile = float(concentration_percentile)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.background_fraction_limit = float(background_fraction_limit)
        self.stain_matrix_target: Optional[np.ndarray] = None
        self.max_concentration_target: Optional[np.ndarray] = None
        super().__init__(reference=reference, **kwargs)

    # ------------------------------------------------------------------
    def _tile_max_concentration(self, image: np.ndarray) -> np.ndarray:
        od = _od_from_rgb(_as_uint8_rgb(image))
        od_flat = od.reshape(-1, 3)
        od_norm = np.linalg.norm(od_flat, axis=1)
        tissue = od_norm > self.od_threshold
        if tissue.sum() < 64:
            raise RuntimeError("insufficient tissue for concentration estimate")
        stain = estimate_stain_matrix(
            image, self.od_threshold, self.angular_percentile
        )
        concentrations = od_flat[tissue] @ np.linalg.pinv(stain)
        return np.percentile(
            concentrations, self.concentration_percentile, axis=0
        )

    def _fit_from_reference(self, reference: np.ndarray) -> None:
        self.stain_matrix_target = estimate_stain_matrix(
            reference, self.od_threshold, self.angular_percentile
        )
        self.max_concentration_target = self._tile_max_concentration(reference)
        # A reference tile with a degenerate concentration range would blow up
        # the rescaling factor; clamp to something physically sane.
        self.max_concentration_target = np.maximum(
            self.max_concentration_target, 1e-3
        )

    def _transform(self, image: np.ndarray) -> np.ndarray:
        assert self.stain_matrix_target is not None
        assert self.max_concentration_target is not None

        arr = _as_uint8_rgb(image)
        if self.background_fraction(arr) > self.background_fraction_limit:
            self.stats.background_skips += 1
            return arr

        od = _od_from_rgb(arr)
        od_flat = od.reshape(-1, 3)
        od_norm = np.linalg.norm(od_flat, axis=1)
        tissue = od_norm > self.od_threshold
        if tissue.sum() < 64:
            self.stats.background_skips += 1
            return arr

        stain_source = estimate_stain_matrix(
            arr, self.od_threshold, self.angular_percentile
        )
        concentrations = od_flat @ np.linalg.pinv(stain_source)
        src_max = np.percentile(
            concentrations[tissue], self.concentration_percentile, axis=0
        )

        # Rescale to the reference dynamic range, clipped so an outlier-free
        # tile cannot be stretched into noise.
        ratio = np.clip(
            self.max_concentration_target / np.maximum(src_max, 1e-3),
            0.05,
            20.0,
        )
        concentrations = (concentrations * ratio) * self.alpha
        concentrations = np.clip(concentrations, 0.0, None)

        od_out = concentrations @ self.stain_matrix_target.T
        return _rgb_from_od(od_out.reshape(arr.shape))
