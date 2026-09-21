"""Geometric augmentation policy (Axis C, ``Aug-Geo``).

Deliberately *stain-blind*: random horizontal/vertical flips plus random 90-degree
rotations only.  Its role in the ablation is to isolate how much robustness comes
from pure spatial invariance, so nothing here may touch colour.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["GeometricAugmenter", "CenterCrop"]


class GeometricAugmenter:
    """Flip + 90-degree rotation augmentation on uint8 RGB ``(H, W, 3)`` tiles.

    Uses ``np.rot90`` / ``np.flip`` instead of an image library so the cost is a
    pure memory view operation (negligible next to the JPEG decode) and so the
    transform is trivially identical in notation to the evaluation pipeline.
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rot90_prob: float = 0.5,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rot90_prob = float(rot90_prob)
        self.rng = rng or random

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)

    def apply(self, image: np.ndarray) -> np.ndarray:
        out = np.asarray(image)
        if self.rng.random() < self.hflip_prob:
            out = np.ascontiguousarray(np.flip(out, axis=1))
        if self.rng.random() < self.vflip_prob:
            out = np.ascontiguousarray(np.flip(out, axis=0))
        if self.rng.random() < self.rot90_prob:
            k = self.rng.choice([1, 2, 3])
            out = np.ascontiguousarray(np.rot90(out, k))
        return out

    def config(self) -> Dict[str, Any]:
        return {
            "hflip_prob": self.hflip_prob,
            "vflip_prob": self.vflip_prob,
            "rot90_prob": self.rot90_prob,
        }


class CenterCrop:
    """Deterministic resize-then-center-crop used by the evaluation transform.

    Kept here (rather than in :mod:`histo_robust.data`) because it is part of the
    geometric policy definition and must stay byte-identical between the
    preprocessing cache and the evaluation path.
    """

    def __init__(self, size: int = 224, resize: Optional[int] = None) -> None:
        self.size = int(size)
        self.resize = int(resize) if resize else None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if self.resize and arr.shape[0] != self.resize:
            arr = self._resize(arr, self.resize)
        h, w = arr.shape[:2]
        if h == self.size and w == self.size:
            return arr
        if h < self.size or w < self.size:  # pad by edge replication
            pad_h = max(0, self.size - h)
            pad_w = max(0, self.size - w)
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            h, w = arr.shape[:2]
        top = (h - self.size) // 2
        left = (w - self.size) // 2
        return np.ascontiguousarray(arr[top : top + self.size, left : left + self.size])

    @staticmethod
    def _resize(arr: np.ndarray, size: int) -> np.ndarray:
        from PIL import Image

        img = Image.fromarray(arr).resize((size, size), resample=Image.BICUBIC)
        return np.asarray(img, dtype=np.uint8)
