"""Console entry points that dispatch into ``scripts/``.

``pip install -e .`` on Kaggle gives short commands (``histo-robust-train`` ...)
without forcing the user to remember relative script paths.  The heavy lifting
stays in ``scripts/`` so the files remain directly runnable --
``python scripts/train.py`` is still the primary interface, exactly as
``kaggle_guide.md`` documents.

Implementation note: the scripts are loaded by file path with ``importlib``
rather than imported as a package, because ``scripts/`` is deliberately not a
package (it may be executed from any working directory, and it needs no
``__init__.py`` to do so).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional

__all__ = [
    "repo_root",
    "load_script",
    "train_main",
    "evaluate_main",
    "prepare_splits_main",
    "smoke_main",
    "run_all_ablations_main",
    "preprocess_normalize_main",
]


def repo_root() -> Path:
    """Repository root, derived from this file's location (``src/histo_robust/``)."""
    return Path(__file__).resolve().parents[2]


def load_script(name: str) -> Any:
    """Import ``scripts/<name>.py`` by path and return the module."""
    path = repo_root() / "scripts" / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"script not found: {path}")
    src = str(repo_root() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    spec = importlib.util.spec_from_file_location(f"histo_robust_scripts.{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dispatch(script: str, argv: Optional[List[str]] = None) -> int:
    module = load_script(script)
    entry: Callable[[Optional[List[str]]], int] = getattr(module, "main")
    return int(entry(argv))


def train_main(argv: Optional[List[str]] = None) -> int:
    """``histo-robust-train`` -> ``scripts/train.py``."""
    return _dispatch("train", argv)


def evaluate_main(argv: Optional[List[str]] = None) -> int:
    """``histo-robust-evaluate`` -> ``scripts/evaluate.py``."""
    return _dispatch("evaluate", argv)


def prepare_splits_main(argv: Optional[List[str]] = None) -> int:
    """``histo-robust-splits`` -> ``scripts/prepare_splits.py``."""
    return _dispatch("prepare_splits", argv)


def smoke_main(argv: Optional[List[str]] = None) -> int:
    """``histo-robust-smoke`` -> ``scripts/smoke_test.py``."""
    return _dispatch("smoke_test", argv)


def run_all_ablations_main(argv: Optional[List[str]] = None) -> int:
    """Not installed as a console script (it is the notebook's long-running cell)."""
    return _dispatch("run_all_ablations", argv)


def preprocess_normalize_main(argv: Optional[List[str]] = None) -> int:
    """Not installed as a console script (optional cache builder)."""
    return _dispatch("preprocess_normalize", argv)
