"""Haematoxylin-Eosin-DAB stain jitter (Axis C, ``Aug-Stain``).

The augmentation deconvolves each tile into stain *concentration* maps with the
Ruifrok & Johnston (2001) HED matrix, applies independent stochastic scaling and
shifting per stain channel, and reconstructs the RGB tile through the
Beer-Lambert law.  This is the biologically grounded counterpart to geometric
augmentation: it perturbs exactly the nuisance factor that changes between
laboratories (dye concentration / incubation time) while leaving morphology
untouched.

Two coarser photometric jitters are layered on top by default -- a global
brightness/contrast gain and a hue/saturation lift implemented in the HSV colour
space -- so the policy also covers scanner white-balance and illumination drift.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["HEDStainJitter", "ruifrok_hed_matrix"]

#: Ruifrok & Johnston (2001) stain vectors in optical-density space (RGB rows).
_RUIFROK_HED = (
    (0.650, 0.072, 0.268),  # Haematoxylin
    (0.704, 0.990, 0.570),  # Eosin
    (0.286, 0.105, 0.000),  # DAB
)
#: Canonical near-white "empty slide" RGB used as the OD reference.
_WHITE_RGB = 240.0
#: Sampling intensity for the canonical concentration baseline C0.
_REFERENCE_SAMPLE = np.array(
    [
        [20.0, 20.0, 20.0],  # pure Haematoxylin-ish
        [20.0, 180.0, 20.0],  # pure Eosin-ish (green channel absoption)
        [120.0, 70.0, 60.0],  # balanced H&E
        [10.0, 60.0, 10.0],
        [180.0, 200.0, 200.0],  # faint
    ],
    dtype=np.float64,
)


def ruifrok_hed_matrix(normalize: bool = True) -> np.ndarray:
    """Return the ``(3, 3)`` HED stain matrix, optionally unit-normalised."""
    matrix = np.asarray(_RUIFROK_HED, dtype=np.float64).T  # -> (3 channels, 3 stains)
    if normalize:
        matrix = matrix / np.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix


def _canonical_concentrations(stain: np.ndarray) -> np.ndarray:
    """95th-percentile concentrations of two canonical stain profiles.

    ``C0`` is the anchor the stochastic scaling is applied *about*; without it,
    ``exp(shift)`` alone would degrade into a plain brightness multiplier and the
    per-stain structure of the augmentation would be lost.
    """
    od = -np.log10((_REFERENCE_SAMPLE + 1.0) / 256.0)
    concentrations = od @ np.linalg.pinv(stain)
    return np.maximum(np.percentile(concentrations, 95.0, axis=0), 1e-3)


class HEDStainJitter:
    """Stochastic stain, brightness and hue jitter for uint8 RGB tiles."""

    def __init__(
        self,
        he_scale: Sequence[float] = (0.85, 1.15),
        he_shift: Sequence[float] = (-0.08, 0.08),
        d_scale: Sequence[float] = (0.95, 1.05),
        brightness: float = 0.10,
        contrast: float = 0.10,
        hue: float = 0.03,
        saturation: float = 0.12,
        apply_prob: float = 0.8,
        use_dab_channel: bool = True,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.he_scale = (float(he_scale[0]), float(he_scale[1]))
        self.he_shift = (float(he_shift[0]), float(he_shift[1]))
        self.d_scale = (float(d_scale[0]), float(d_scale[1]))
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.hue = float(hue)
        self.saturation = float(saturation)
        self.apply_prob = float(apply_prob)
        self.use_dab_channel = bool(use_dab_channel)
        self.rng = rng or random

        self.stain_matrix = ruifrok_hed_matrix(normalize=True)
        self.stain_pinv = np.linalg.pinv(self.stain_matrix)
        self.c0 = _canonical_concentrations(self.stain_matrix)

    # ------------------------------------------------------------------
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)

    def _uniform(self, bounds: Tuple[float, float]) -> float:
        lo, hi = bounds
        return self.rng.uniform(lo, hi)

    def sample_parameters(self) -> Dict[str, Any]:
        scales = [self._uniform(self.he_scale), self._uniform(self.he_scale)]
        shifts = [self._uniform(self.he_shift), self._uniform(self.he_shift)]
        if self.use_dab_channel:
            scales.append(self._uniform(self.d_scale))
            shifts.append(0.0)
        params = {
            "scale": np.asarray(scales, dtype=np.float64),
            "shift": np.asarray(shifts, dtype=np.float64),
            "brightness": self._uniform((-self.brightness, self.brightness)),
            "contrast": self._uniform((1.0 - self.contrast, 1.0 + self.contrast)),
            "hue": self._uniform((-self.hue, self.hue)),
            "saturation": self._uniform((1.0 - self.saturation, 1.0 + self.saturation)),
        }
        return params

    def apply(
        self, image: np.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if self.rng.random() >= self.apply_prob:
            return arr
        params = params or self.sample_parameters()

        # 1) stain concentration perturbation via Beer-Lambert deconvolution
        od = -np.log10((arr.astype(np.float64) + 1.0) / 256.0)
        flat = od.reshape(-1, 3)
        concentrations = flat @ self.stain_pinv
        n_stains = len(params["scale"])
        c0 = self.c0[:n_stains]
        concentrations[:, :n_stains] = (
            concentrations[:, :n_stains] * params["scale"]
            + params["shift"] * c0
        )
        concentrations = np.clip(concentrations, 0.0, None)
        od_out = concentrations @ self.stain_matrix.T
        rgb = 256.0 * np.power(10.0, -od_out) - 1.0
        rgb = np.clip(rgb, 0, 255).reshape(arr.shape)

        # 2) brightness / contrast gain
        rgb = (rgb - 128.0) * params["contrast"] + 128.0 * (1.0 + params["brightness"])
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # 3) hue / saturation lift (scanner white-balance drift)
        if abs(params["hue"]) > 1e-6 or abs(params["saturation"] - 1.0) > 1e-6:
            rgb = self._hsv_jitter(rgb, params["hue"], params["saturation"])
        return rgb

    @staticmethod
    def _hsv_jitter(rgb: np.ndarray, hue_delta: float, saturation_scale: float) -> np.ndarray:
        from skimage.color import hsv2rgb, rgb2hsv

        hsv = rgb2hsv(rgb.astype(np.float64) / 255.0)
        hsv[..., 0] = (hsv[..., 0] + hue_delta) % 1.0
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0.0, 1.0)
        out = hsv2rgb(hsv) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    def config(self) -> Dict[str, Any]:
        return {
            "type": "hed_stain_jitter",
            "he_scale": self.he_scale,
            "he_shift": self.he_shift,
            "d_scale": self.d_scale,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "hue": self.hue,
            "saturation": self.saturation,
            "apply_prob": self.apply_prob,
            "use_dab_channel": self.use_dab_channel,
            "stain_matrix": self.stain_matrix.tolist(),
        }
