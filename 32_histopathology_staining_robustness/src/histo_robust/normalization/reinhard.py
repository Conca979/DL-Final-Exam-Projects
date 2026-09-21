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
        from skimage.color import rgb2lab

        return rgb2lab(rgb.astype(np.float64) / 255.0)

    @staticmethod
    def _lab2rgb(lab: np.ndarray) -> np.ndarray:
        from skimage.color import lab2rgb

        out = lab2rgb(lab) * 255.0
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
