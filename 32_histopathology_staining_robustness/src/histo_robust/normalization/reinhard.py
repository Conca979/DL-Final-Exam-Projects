"""Reinhard colour normalisation in CIELAB space (Reinhard et al., 2001).

Lab statistics (mean, std) of the source tile are linearly mapped onto the
statistics of a canonical reference tile::

    L*_out = (L*_in - mu_in) / sigma_in * sigma_ref + mu_ref

Implemented with ``skimage.color.lab2rgb``/``rgb2lab`` (D65 white point).  A
luminance tissue mask is used when fitting the reference so that slide glass
does not dominate the statistics of mostly-background tiles.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import BaseNormalizer, _as_uint8_rgb

__all__ = ["ReinhardNormalizer"]

#: D65 white point used by the skimage-free CIELAB fallback.
_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)


def _f_forward(t: np.ndarray) -> np.ndarray:
    return np.where(t > _EPS, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)


def _f_inverse(t: np.ndarray) -> np.ndarray:
    t3 = t**3
    return np.where(t3 > _EPS, t3, (116.0 * t - 16.0) / _KAPPA)


def _srgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] -> CIELAB, matching ``skimage.color.rgb2lab`` numerically."""
    linear = _srgb_to_linear(np.clip(rgb01, 0.0, 1.0))
    xyz = linear @ np.array(
        [
            [0.4124564, 0.2126729, 0.0193339],
            [0.3575761, 0.7151522, 0.1191920],
            [0.1804375, 0.0721750, 0.9503041],
        ]
    ).T
    xyz = xyz / _D65
    f = _f_forward(xyz)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = _f_inverse(f) * _D65
    linear = xyz @ np.array(
        [
            [3.2404542, -0.9692660, 0.0556434],
            [-1.5371385, 1.8760108, -0.2040259],
            [-0.4985314, 0.0415560, 1.0572252],
        ]
    ).T
    return np.clip(_linear_to_srgb(linear), 0.0, 1.0)


class ReinhardNormalizer(BaseNormalizer):
    """Standard Lab-space Reinhard normalisation."""

    name = "reinhard"

    def __init__(
        self,
        reference: Optional[np.ndarray] = None,
        use_tissue_mask: bool = True,
        luminance_threshold: int = 220,
        **kwargs,
    ) -> None:
        self.use_tissue_mask = bool(use_tissue_mask)
        self.luminance_threshold = int(luminance_threshold)
        self.ref_mean: Optional[np.ndarray] = None
        self.ref_std: Optional[np.ndarray] = None
        super().__init__(reference=reference, **kwargs)

    # ------------------------------------------------------------------
    @staticmethod
    def _rgb2lab(rgb: np.ndarray) -> np.ndarray:
        """sRGB -> CIELAB (D65).  sklearn-free fallback if skimage is missing."""
        try:
            from skimage.color import rgb2lab

            return rgb2lab(rgb.astype(np.float64) / 255.0)
        except ImportError:
            return _srgb_to_lab(rgb.astype(np.float64) / 255.0)

    @staticmethod
    def _lab2rgb(lab: np.ndarray) -> np.ndarray:
        try:
            from skimage.color import lab2rgb

            out = lab2rgb(lab) * 255.0
        except ImportError:
            out = _lab_to_srgb(lab) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    def _tissue_mask(self, lab: np.ndarray) -> np.ndarray:
        if not self.use_tissue_mask:
            return np.ones(lab.shape[:2], dtype=bool)
        mask = lab[..., 0] < (self.luminance_threshold / 255.0 * 100.0)
        if mask.sum() < max(32, 0.02 * mask.size):
            return np.ones(lab.shape[:2], dtype=bool)
        return mask

    # ------------------------------------------------------------------
    def _fit_from_reference(self, reference: np.ndarray) -> None:
        lab = self._rgb2lab(reference)
        mask = self._tissue_mask(lab)
        pixels = lab[mask]
        self.ref_mean = pixels.mean(axis=0)
        self.ref_std = pixels.std(axis=0)
        # Guard against flat tiles -> zero std (would amplify noise to inf)
        self.ref_std = np.maximum(self.ref_std, 1e-4)

    def _transform(self, image: np.ndarray) -> np.ndarray:
        assert self.ref_mean is not None and self.ref_std is not None
        lab = self._rgb2lab(image)
        mask = self._tissue_mask(lab)
        pixels = lab[mask]
        src_mean = pixels.mean(axis=0)
        src_std = np.maximum(pixels.std(axis=0), 1e-4)

        out_lab = lab.copy()
        out_lab[mask] = (
            (pixels - src_mean) / src_std * self.ref_std + self.ref_mean
        )
        out_lab = np.clip(out_lab, [0.0, -128.0, -128.0], [100.0, 127.0, 127.0])
        return self._lab2rgb(out_lab)

    def state_dict(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.ref_mean is not None and self.ref_std is not None
        return self.ref_mean.copy(), self.ref_std.copy()
