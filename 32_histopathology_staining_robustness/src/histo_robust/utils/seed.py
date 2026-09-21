"""Deterministic seeding for reproducible Kaggle runs.

``set_seed`` pins Python, NumPy and (when available) PyTorch CPU/CUDA RNGs and
flips cuDNN into deterministic mode.  ``get_rng_state``/``set_rng_state`` are
used by the trainer so a resumed run continues the exact same random stream.
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import numpy as np

__all__ = [
    "set_seed",
    "seed_worker",
    "get_rng_state",
    "set_rng_state",
    "temporary_seed",
    "seed_everything_strict_note",
]


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed every RNG we might touch and return the seed."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is optional at import time (e.g. docs builds)
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - older torch
            pass
    except ImportError:  # pragma: no cover
        pass
    return seed


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - torch API signature
    """``DataLoader(worker_init_fn=...)`` hook: give each worker a stable seed."""
    import torch

    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed)
    random.seed(base_seed)


def get_rng_state() -> Dict[str, Any]:
    """Snapshot RNG state so training can be resumed bit-for-bit."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:  # pragma: no cover
        pass
    return state


def set_rng_state(state: Optional[Dict[str, Any]]) -> None:
    """Restore a snapshot produced by :func:`get_rng_state` (no-op if empty)."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch

        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except ImportError:  # pragma: no cover
        pass


@contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Run a block under a local seed, restoring the previous RNG state after."""
    state = get_rng_state()
    try:
        set_seed(seed)
        yield
    finally:
        set_rng_state(state)


def seed_everything_strict_note() -> str:
    """Human-readable note about what determinism we can actually promise."""
    return (
        "Deterministic mode is requested (cudnn.deterministic=True). "
        "GPU kernels may still differ by a few ULP across driver versions; "
        "report metrics to 4 decimals and keep the seed fixed at 42 for comparability."
    )
