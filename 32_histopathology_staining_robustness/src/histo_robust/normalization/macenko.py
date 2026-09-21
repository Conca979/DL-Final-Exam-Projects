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
* ``Risk 1`` guard from ``docs/PLAN.md``: tiles that are >85% slide glass bypass the
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

__all__ = [
    "MacenkoNormalizer",
    "estimate_stain_matrix",
    "canonical_he_stain_matrix",
    "stain_matrix_separation_deg",
]

#: Below this OD norm a pixel is treated as background / glass.
DEFAULT_OD_THRESHOLD = 0.15
#: Angular percentile used to pick the two extreme stain directions.
DEFAULT_ANGULAR_PERCENTILE = 99.0
#: Concentration percentile mapped onto the reference tile's percentile.
DEFAULT_CONCENTRATION_PERCENTILE = 99.0
#: Estimate/template blend point: separations at or above this are trusted fully,
#: below it the estimate is progressively blended with the canonical H&E vectors.
DEFAULT_MIN_SEPARATION_DEG = 25.0
#: Fallback stain vectors.  These are the Ruifrok & Johnston (2001) H&E vectors
#: (Haematoxylin, then Eosin); they are used verbatim only when a tile is so
#: monochrome that the optical-density directions cannot be separated at all.
_CANONICAL_HE = np.array(
    [
        [0.650, 0.072],
        [0.704, 0.990],
        [0.286, 0.105],
    ],
    dtype=np.float64,
)


def canonical_he_stain_matrix() -> np.ndarray:
    """Unit-norm canonical H&E stain matrix (column 0 = H, column 1 = E)."""
    matrix = _CANONICAL_HE.copy()
    return matrix / np.linalg.norm(matrix, axis=0, keepdims=True)


def stain_matrix_separation_deg(stain: np.ndarray) -> float:
    """Angle between the two stain columns, in degrees."""
    a = stain[:, 0] / max(np.linalg.norm(stain[:, 0]), 1e-12)
    b = stain[:, 1] / max(np.linalg.norm(stain[:, 1]), 1e-12)
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0))))


def _od_from_rgb(rgb: np.ndarray) -> np.ndarray:
    return -np.log10((rgb.astype(np.float64) + 1.0) / 256.0)


def _rgb_from_od(od: np.ndarray) -> np.ndarray:
    rgb = 256.0 * np.power(10.0, -od) - 1.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def estimate_stain_matrix(
    rgb: np.ndarray,
    od_threshold: float = DEFAULT_OD_THRESHOLD,
    angular_percentile: float = DEFAULT_ANGULAR_PERCENTILE,
    min_angular_separation_deg: float = DEFAULT_MIN_SEPARATION_DEG,
) -> np.ndarray:
    """Return a 3x2 unit-norm stain matrix ``[[r, r], [g, g], [b, b]]``.

    Columns are ordered so that column 0 is Haematoxylin (the darker, more
    red/green-absorbing direction) and column 1 is Eosin.

    Method
    ------
    Let ``m`` be the mean unit optical-density direction (the tile's dominant
    stain mixture).  Every tissue pixel is then mapped into the plane
    perpendicular to ``m`` by

    ``p_i = normalize(u_i - (u_i . m) m)``

    and the two stain directions are the extreme azimuths of ``p_i`` within that
    plane, at ``+/-angular_percentile``.  Both recovered vectors are therefore
    (nearly) orthogonal to ``m``, which keeps ``[H E]`` well conditioned.  This
    is the parametrisation used by the reference Macenko implementations and it
    is what makes the method stable on tiles that are dominated by one stain.

    Three robustness details matter here, and each is a silent failure mode if
    omitted:

    1. **Branch cut.** Azimuths computed with ``arctan2`` straddle the +/-pi
       discontinuity, so a naive percentile returns two nearly identical
       directions.  Angles are rotated by the *in-plane* mean direction first --
       rotating by the circular mean of the azimuths is not enough, because the
       raw azimuth distribution is symmetric and its 1st percentile lands on the
       opposite side of the circle.
    2. **Basis invariance.** The in-plane basis is built deterministically from
       ``m`` with a fixed reference axis (and a canonical sign), so the same tile
       cannot yield different stain vectors on different BLAS builds.
    3. **Degenerate tiles.** If the two directions come out closer than
       ``min_angular_separation_deg`` the tile cannot resolve H from E; the
       estimate is then blended toward the canonical Ruifrok H&E vectors in
       proportion to its confidence, which keeps normalisation continuous
       instead of amplifying rounding noise through a near-singular
       pseudo-inverse.
    """
    od = _od_from_rgb(_as_uint8_rgb(rgb))
    od_flat = od.reshape(-1, 3)
    od_norm = np.linalg.norm(od_flat, axis=1)
    tissue = od_norm > od_threshold
    if tissue.sum() < 64:
        raise RuntimeError(f"only {int(tissue.sum())} tissue pixels (need >= 64)")

    unit = od_flat[tissue] / od_norm[tissue][:, None]

    # Dominant stain-mixture direction for this tile.
    mean_vec = unit.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))
    if mean_norm < 1e-6:
        raise RuntimeError("mean optical-density direction is degenerate")
    mean_dir = mean_vec / mean_norm

    # (2) deterministic in-plane basis: take the residual of a fixed reference
    # axis and complete the right-handed frame with a cross product.
    reference_axis = np.zeros(3, dtype=np.float64)
    reference_axis[int(np.argmin(np.abs(mean_dir)))] = 1.0
    axis_u = reference_axis - np.dot(reference_axis, mean_dir) * mean_dir
    axis_u_norm = float(np.linalg.norm(axis_u))
    if axis_u_norm < 1e-6:  # pragma: no cover - impossible for the axis choice
        raise RuntimeError("could not construct an in-plane basis")
    axis_u = axis_u / axis_u_norm
    axis_v = np.cross(mean_dir, axis_u)
    axis_v = axis_v / max(float(np.linalg.norm(axis_v)), 1e-12)

    residuals = unit - np.outer(unit @ mean_dir, mean_dir)
    residual_norm = np.linalg.norm(residuals, axis=1)
    valid = residual_norm > 1e-6
    if valid.sum() < 32:
        raise RuntimeError("tile optical density is one-dimensional")
    residual_unit = residuals[valid] / residual_norm[valid][:, None]

    phi = np.arctan2(residual_unit @ axis_v, residual_unit @ axis_u)
    # (1) rotate by the in-plane mean direction to remove the branch cut.
    mean_angle = np.arctan2(np.sin(phi).mean(), np.cos(phi).mean())
    phi = phi - mean_angle

    lo = float(np.percentile(phi, 100.0 - angular_percentile))
    hi = float(np.percentile(phi, angular_percentile))

    v_lo = np.cos(lo) * axis_u + np.sin(lo) * axis_v
    v_hi = np.cos(hi) * axis_u + np.sin(hi) * axis_v
    v_lo = v_lo / max(float(np.linalg.norm(v_lo)), 1e-12)
    v_hi = v_hi / max(float(np.linalg.norm(v_hi)), 1e-12)

    # Deterministic H/E ordering (H absorbs more red than E), applied to both the
    # estimate and the canonical template so a blend cannot swap the channels.
    if v_lo[0] > v_hi[0]:
        v_lo, v_hi = v_hi, v_lo
    canonical = canonical_he_stain_matrix()
    if canonical[0, 0] > canonical[0, 1]:
        canonical = canonical[:, ::-1]

    separation = np.degrees(
        np.arccos(np.clip(abs(float(np.dot(v_lo, v_hi))), -1.0, 1.0))
    )
    # (3) confidence-weighted blend toward the canonical vectors.
    confidence = float(
        np.clip(separation / max(1e-6, min_angular_separation_deg), 0.0, 1.0)
    )
    estimated = np.stack([v_lo, v_hi], axis=1)
    stain = confidence * estimated + (1.0 - confidence) * canonical

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
        # Concentrations solve OD = C @ V^T, hence C = OD @ pinv(V^T) = OD @ pinv(V)^T.
        # (V is (3, 2), pinv(V) is (2, 3), so the transpose is required.)
        concentrations = od_flat[tissue] @ np.linalg.pinv(stain).T
        return np.percentile(
            concentrations, self.concentration_percentile, axis=0
        )

    def _fit_from_reference(self, reference: np.ndarray) -> None:
        try:
            self.stain_matrix_target = estimate_stain_matrix(
                reference, self.od_threshold, self.angular_percentile
            )
            self.max_concentration_target = np.maximum(
                self._tile_max_concentration(reference), 1e-3
            )
        except RuntimeError as exc:
            # A reference tile that cannot resolve H/E (a glass-heavy or
            # near-monochrome patch) must not abort the whole run: fall back to
            # the canonical Ruifrok vectors, which are exactly what the
            # confidence blend would have produced at zero confidence.
            logger.warning(
                "Reference tile could not resolve stain vectors (%s); using the "
                "canonical Ruifrok H&E matrix instead. Prefer a tissue-rich "
                "reference tile (prepare_splits.py --reference-out).",
                exc,
            )
            self.stain_matrix_target = canonical_he_stain_matrix()
            if self.stain_matrix_target[0, 0] > self.stain_matrix_target[0, 1]:
                self.stain_matrix_target = self.stain_matrix_target[:, ::-1]
            try:
                self.max_concentration_target = np.maximum(
                    self._tile_max_concentration(reference), 1e-3
                )
            except RuntimeError:
                # Last resort: a canonical reference profile in OD space.
                od = _od_from_rgb(reference).reshape(-1, 3)
                self.max_concentration_target = np.maximum(
                    np.percentile(
                        od @ np.linalg.pinv(self.stain_matrix_target).T,
                        self.concentration_percentile,
                        axis=0,
                    ),
                    1e-3,
                )
        # A degenerate concentration range would blow up the rescaling factor;
        # clamp to something physically sane.
        self.max_concentration_target = np.maximum(self.max_concentration_target, 1e-3)

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
        # See _tile_max_concentration for why pinv(V) is transposed here.
        concentrations = od_flat @ np.linalg.pinv(stain_source).T
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
        target_matrix = self.stain_matrix_target
        if target_matrix.shape[1] != concentrations.shape[1]:
            # Rebuilding with the reference column count is the only part that
            # needs matching shapes; pad with zeros (a stain the target does not
            # have contributes no optical density).
            padded = np.zeros((3, concentrations.shape[1]), dtype=np.float64)
            padded[:, : target_matrix.shape[1]] = target_matrix
            target_matrix = padded

        concentrations = (concentrations * ratio) * self.alpha
        concentrations = np.clip(concentrations, 0.0, None)

        od_out = concentrations @ target_matrix.T
        return _rgb_from_od(od_out.reshape(arr.shape))
