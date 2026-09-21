"""Small plotting helpers (confusion matrices, sample tile sheets).

All figures are written to disk only -- nothing is shown interactively, which
keeps the code path identical inside a Kaggle *background* run where no display
exists.  ``matplotlib.use("Agg")`` is forced at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .metrics import CLASS_NAMES

__all__ = ["plot_confusion_matrix", "plot_sample_grid"]


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion_matrix(
    cm: np.ndarray,
    out_path: str | Path,
    class_names: Sequence[str] = CLASS_NAMES,
    title: Optional[str] = None,
) -> Path:
    """Save a row-normalised confusion-matrix heatmap (rows = true class)."""
    plt = _mpl()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(cm, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=140)
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if value <= 0:
                continue
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.55 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_sample_grid(
    images_uint8: np.ndarray,
    out_path: str | Path,
    titles: Optional[Sequence[str]] = None,
    max_images: int = 8,
    ncols: int = 4,
    suptitle: Optional[str] = None,
) -> Path:
    """Save a grid of uint8 RGB tiles (used for visual stain diagnostics)."""
    plt = _mpl()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images_uint8 = np.asarray(images_uint8)
    images_uint8 = images_uint8[:max_images]
    n = images_uint8.shape[0]
    if n == 0:
        raise ValueError("No images supplied to plot_sample_grid")
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.1 * nrows), dpi=130)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for idx in range(n):
        axes[idx].imshow(images_uint8[idx])
        if titles is not None and idx < len(titles):
            axes[idx].set_title(str(titles[idx]), fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
