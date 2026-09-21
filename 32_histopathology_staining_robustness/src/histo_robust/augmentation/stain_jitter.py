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

__all__ = ["HEDStainJitter", "ruifrok_hed_matrix", "he_stain_matrix"]

#: Ruifrok & Johnston (2001) stain vectors in optical-density space.  Each entry
#: is one stain as an (R, G, B) absorption triple.  Keeping the stains as a
#: dict -- rather than a bare nested tuple -- removes the transposition ambiguity
#: that silently corrupts a pseudo-inverse.
_RUIFROK_HED_TRIPLES = {
    "hematoxylin": (0.650, 0.072, 0.268),
    "eosin": (0.704, 0.990, 0.570),
    "dab": (0.286, 0.105, 0.000),
}
#: Canonical near-white "empty slide" RGB used as the OD reference.
_WHITE_RGB = 240.0
#: Physical upper bound for optical density (256x attenuation, matching the
#: ``(I + 1) / 256`` convention used for the forward transform).
_MAX_OD = float(np.log10(256.0))
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
    """Return the published Ruifrok & Johnston ``(3, 3)`` HED matrix.

    Shape is ``(channels, stains)``, i.e. column 0 is haematoxylin, column 1 is
    eosin and column 2 is DAB, matching the ``OD = C @ V^T`` convention used
    everywhere in this codebase.  Retained for provenance/inspection.
    """
    order = ("hematoxylin", "eosin", "dab")
    matrix = np.asarray([_RUIFROK_HED_TRIPLES[k] for k in order], dtype=np.float64).T
    if normalize:
        matrix = matrix / np.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix


def he_stain_matrix() -> np.ndarray:
    """Unit-norm ``(3, 2)`` H&E matrix (column 0 = Haematoxylin, column 1 = Eosin).

    The jitter deliberately deconvolves with the **two** H&E columns only:

    * the inversion is well conditioned (the pseudo-inverse has no large
      cancelling coefficients, unlike the 3-stain HED inverse whose DAB row is
      full of +-2 entries),
    * the mapping "scale channel 0 -> more haematoxylin, scale channel 1 -> more
      eosin" is exact, so the augmentation has the biological meaning its name
      claims, and an empty channel stays empty instead of leaking into the other
      stain through the DAB row,
    * it matches the two-stain decomposition used by the Macenko normaliser in
      Axis B, which keeps Axes B and C conceptually comparable.
    """
    matrix = np.asarray(
        [_RUIFROK_HED_TRIPLES["hematoxylin"], _RUIFROK_HED_TRIPLES["eosin"]],
        dtype=np.float64,
    ).T
    return matrix / np.linalg.norm(matrix, axis=0, keepdims=True)


def _canonical_concentrations(stain: np.ndarray) -> np.ndarray:
    """95th-percentile H/E concentrations of canonical stain profiles.

    ``C0`` is the anchor the stochastic *shift* is applied about; without it,
    a shift would degrade into a plain brightness offset instead of a
    stain-specific concentration change.

    ``C = OD @ pinv(V)^T`` (V is ``(3, S)`` so ``pinv(V)^T`` is ``(3, S)``).
    """
    od = -np.log10((_REFERENCE_SAMPLE + 1.0) / 256.0)
    concentrations = od @ np.linalg.pinv(stain).T
    return np.maximum(np.percentile(concentrations, 95.0, axis=0), 1e-3)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB->HSV (all channels in ``[0, 1]``).

    Implemented in NumPy rather than imported from ``skimage.color`` so the
    augmentation has no image-library dependency beyond Pillow.  The formula is
    the standard one and agrees with ``skimage.color.rgb2hsv`` to float
    precision (skimage only adds an epsilon to avoid exact-zero denominators,
    which cannot change an 8-bit result).
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    nonzero = delta > 1e-12

    red_max = nonzero & (maxc == r)
    green_max = nonzero & (maxc == g) & ~red_max
    blue_max = nonzero & ~red_max & ~green_max

    hue[red_max] = ((g - b)[red_max] / delta[red_max]) % 6.0
    hue[green_max] = ((b - r)[green_max] / delta[green_max]) + 2.0
    hue[blue_max] = ((r - g)[blue_max] / delta[blue_max]) + 4.0
    hue = hue / 6.0

    saturation = np.zeros_like(maxc)
    positive = maxc > 1e-12
    saturation[positive] = delta[positive] / maxc[positive]

    return np.stack([hue, saturation, maxc], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Vectorised HSV->RGB inverse of :func:`_rgb_to_hsv`."""
    hue = np.mod(hsv[..., 0], 1.0) * 6.0
    saturation = np.clip(hsv[..., 1], 0.0, 1.0)
    value = np.clip(hsv[..., 2], 0.0, 1.0)

    sector = np.floor(hue).astype(np.int64) % 6
    fraction = hue - np.floor(hue)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))

    out = np.zeros(hsv.shape, dtype=np.float64)
    choices = (
        (value, t, p),
        (q, value, p),
        (p, value, t),
        (p, q, value),
        (t, p, value),
        (value, p, q),
    )
    for idx, (rr, gg, bb) in enumerate(choices):
        mask = sector == idx
        out[..., 0][mask] = rr[mask]
        out[..., 1][mask] = gg[mask]
        out[..., 2][mask] = bb[mask]
    return out


class HEDStainJitter:
    """Stochastic H&E stain, brightness and hue jitter for uint8 RGB tiles.

    The transform is a fixed, well-conditioned two-stain deconvolution followed by
    per-stain concentration rescaling, reconstruction, and two coarser
    photometric perturbations.  Because the H/E basis is fixed (not estimated per
    tile), "channel 0" always means haematoxylin and "channel 1" always means
    eosin -- which is what makes the augmentation biologically interpretable and
    reproducible rather than an arbitrary colour warp.
    """

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
        use_dab_channel: bool = False,
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
        # DAB is an immunohistochemistry counterstain, not an H&E stain; it is
        # off by default and only kept as an optional extra channel.
        self.use_dab_channel = bool(use_dab_channel)
        self.rng = rng or random

        self.stain_matrix = he_stain_matrix()          # (3, 2)
        self.stain_pinv = np.linalg.pinv(self.stain_matrix)  # (2, 3)
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
        if self.use_dab_channel and self.stain_matrix.shape[1] > 2:
            # Only meaningful when a DAB column is actually present in the matrix.
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

        # 1) stain concentration perturbation via Beer-Lambert deconvolution.
        #    OD = C @ V^T  =>  C = OD @ pinv(V)^T  (V is (3, 2), so pinv(V)^T is (2, 3)).
        od = -np.log10((arr.astype(np.float64) + 1.0) / 256.0)
        flat = od.reshape(-1, 3)
        concentrations = flat @ self.stain_pinv.T
        n_stains = min(len(params["scale"]), concentrations.shape[1])
        c0 = self.c0[:n_stains]
        concentrations[:, :n_stains] = (
            concentrations[:, :n_stains] * params["scale"][:n_stains]
            + params["shift"][:n_stains] * c0
        )
        # Clip in OPTICAL DENSITY space, not concentration space: the valid OD
        # range is a known physical bound, whereas concentrating to >= 0 flattens
        # the colour planes and produces an unwanted hue shift.  With identity
        # scales this reproduces the input exactly (a useful property for
        # --save-samples and for reasoning about the transform).
        od_out = concentrations @ self.stain_matrix.T
        od_out = np.clip(od_out, 0.0, _MAX_OD)
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
        rgb_float = rgb.astype(np.float64) / 255.0
        hsv = _rgb_to_hsv(rgb_float)
        hsv[..., 0] = (hsv[..., 0] + hue_delta) % 1.0
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0.0, 1.0)
        out = _hsv_to_rgb(hsv) * 255.0
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
