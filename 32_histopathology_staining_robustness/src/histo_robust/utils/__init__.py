"""Utility helpers: configuration, reproducibility, metrics, Kaggle runtime."""

from .checkpoint import CheckpointManager, find_latest_checkpoint, load_checkpoint, save_checkpoint
from .config import (
    apply_overrides,
    config_hash,
    config_to_json,
    deep_update,
    get_by_path,
    load_config,
    set_by_path,
)
from .metrics import (
    CLASS_NAMES,
    NUM_CLASSES,
    compute_classification_metrics,
    confusion_matrix_normalized,
    expected_calibration_error,
    per_class_report,
    robustness_metrics,
)
from .seed import get_rng_state, seed_worker, set_rng_state, set_seed, temporary_seed
from .timebudget import BudgetAllocator, SessionClock, TimeBudget

__all__ = [
    "load_config",
    "apply_overrides",
    "get_by_path",
    "set_by_path",
    "config_hash",
    "config_to_json",
    "deep_update",
    "set_seed",
    "seed_worker",
    "get_rng_state",
    "set_rng_state",
    "temporary_seed",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "compute_classification_metrics",
    "expected_calibration_error",
    "per_class_report",
    "confusion_matrix_normalized",
    "robustness_metrics",
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
    "TimeBudget",
    "SessionClock",
    "BudgetAllocator",
]
