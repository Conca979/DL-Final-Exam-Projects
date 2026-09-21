"""Dependency checks + install helpers for the Kaggle notebook cells.

The Kaggle PyTorch/GPU image already ships every heavy dependency this project
needs (torch, torchvision, timm, transformers, scikit-learn, pandas, opencv...).
Re-installing them costs 5-10 minutes of a 12-hour budget for no benefit, so the
notebook *checks first* and only installs what is genuinely missing.

``verify_imports`` returns a report the notebook prints before the 10-hour run,
which turns "it failed at epoch 3 with ImportError" into "the setup cell said
transformers was missing".
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "REQUIRED_IMPORTS",
    "OPTIONAL_IMPORTS",
    "check_imports",
    "verify_imports",
    "install_package",
    "install_project",
    "ensure_dependencies",
    "print_report",
]

#: ``(import_name, pip_name, minimum_version)`` for everything the pipeline needs.
REQUIRED_IMPORTS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("numpy", "numpy", "1.24"),
    ("pandas", "pandas", "1.5"),
    ("sklearn", "scikit-learn", "1.2"),
    ("yaml", "PyYAML", "6.0"),
    ("PIL", "Pillow", "9.0"),
    ("torch", "torch", "2.1"),
    ("torchvision", "torchvision", "0.16"),
    ("timm", "timm", "0.9.16"),
    ("matplotlib", "matplotlib", "3.7"),
    ("seaborn", "seaborn", "0.12"),
)

#: Nice to have. Missing entries degrade a feature instead of killing the run.
OPTIONAL_IMPORTS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("transformers", "transformers", "4.38"),
    ("cv2", "opencv-python-headless", "4.8"),
    ("skimage", "scikit-image", "0.22"),
    ("tqdm", "tqdm", "4.64"),
)

#: Fallbacks used when a pip install is not possible (e.g. Internet is off):
#: the package import name -> the internal module that replaces it.
FALLBACKS: Dict[str, str] = {
    "sklearn": "histo_robust.utils.metrics (built-in NumPy metric fallback)",
    "cv2": "not required by any code path",
    "skimage": "histo_robust.normalization.reinhard / augmentation.stain_jitter (NumPy fallbacks)",
    "transformers": "only needed for the Phikon cells (EXP-12, EXP-13)",
    "tqdm": "not required by any code path",
}


def _version_of(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def check_imports(specs: Sequence[Tuple[str, str, Optional[str]]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for import_name, pip_name, minimum in specs:
        entry: Dict[str, Any] = {
            "import_name": import_name,
            "pip_name": pip_name,
            "required_min": minimum,
            "installed": False,
            "version": None,
            "ok": False,
        }
        try:
            module = importlib.import_module(import_name)
            entry["installed"] = True
            entry["version"] = _version_of(module)
            entry["ok"] = True
            if minimum:
                try:
                    from packaging.version import Version

                    entry["ok"] = Version(entry["version"]) >= Version(minimum)
                except Exception:  # noqa: BLE001 - unknown version string
                    entry["ok"] = True
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results.append(entry)
    return results


def install_package(pip_name: str, quiet: bool = True, upgrade: bool = False) -> bool:
    """``pip install`` inside the running kernel so the import appears immediately."""
    args = [sys.executable, "-m", "pip", "install"]
    if quiet:
        args.append("-q")
    if upgrade:
        args.append("--upgrade")
    args.append(pip_name)
    logger.info("Installing %s ...", pip_name)
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("pip install %s failed:\n%s", pip_name, (result.stderr or "")[-2000:])
        return False
    importlib.invalidate_caches()
    return True


def install_project(repo_root: str = "/kaggle/working/histo-robust", editable: bool = True) -> bool:
    """Install the codebase itself (``pip install -e .``) so imports resolve."""
    args = [sys.executable, "-m", "pip", "install", "-q"]
    args += ["-e", repo_root] if editable else [repo_root]
    logger.info("Installing the project from %s ...", repo_root)
    result = subprocess.run(args, capture_output=True, text=True, cwd=repo_root)
    if result.returncode != 0:
        logger.error("pip install -e %s failed:\n%s", repo_root, (result.stderr or "")[-3000:])
        return False
    importlib.invalidate_caches()
    return True


def ensure_dependencies(
    repo_root: str = "/kaggle/working/histo-robust",
    allow_install: bool = True,
    install_optional: bool = True,
) -> Dict[str, Any]:
    """Install whatever is missing, then report the final state.

    Returns ``{"required": [...], "optional": [...], "missing_required": [...],
    "fallbacks": {...}}`` -- the notebook asserts ``missing_required`` is empty
    before starting the long run.
    """
    required = check_imports(REQUIRED_IMPORTS)
    missing = [entry for entry in required if not entry["ok"]]

    if missing and allow_install:
        for entry in missing:
            install_package(entry["pip_name"], upgrade=not entry["installed"])
        required = check_imports(REQUIRED_IMPORTS)
        missing = [entry for entry in required if not entry["ok"]]

    optional = check_imports(OPTIONAL_IMPORTS)
    if install_optional and allow_install:
        optional_missing = [entry for entry in optional if not entry["ok"]]
        for entry in optional_missing:
            install_package(entry["pip_name"])
        if optional_missing:
            optional = check_imports(OPTIONAL_IMPORTS)

    return {
        "required": required,
        "optional": optional,
        "missing_required": [entry["pip_name"] for entry in missing],
        "missing_optional": [entry["pip_name"] for entry in optional if not entry["ok"]],
        "fallbacks": FALLBACKS,
    }


def verify_imports(repo_root: str = "/kaggle/working/histo-robust") -> bool:
    """Re-check that ``histo_robust`` itself imports after installation."""
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
        sys.path.insert(0, f"{repo_root}/src")
    try:
        importlib.invalidate_caches()
        module = importlib.import_module("histo_robust")
        logger.info("histo_robust %s imported from %s", module.__version__, module.__file__)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("histo_robust import failed: %s: %s", type(exc).__name__, exc)
        return False


def print_report(report: Dict[str, Any]) -> None:
    print("DEPENDENCY REPORT")
    print("-" * 70)
    for entry in report["required"]:
        mark = "OK " if entry["ok"] else "!! "
        print(f"  [{mark}] {entry['pip_name']:<24} {entry['version'] or '-'}")
    for entry in report["optional"]:
        mark = "OK " if entry["ok"] else "-- "
        print(f"  [{mark}] {entry['pip_name']:<24} {entry['version'] or 'missing (optional)'}")
    if report["missing_optional"]:
        print("\n  Optional packages that are absent and their in-repo fallback:")
        for name in report["missing_optional"]:
            print(f"    {name:<24} -> {FALLBACKS.get(name, 'no fallback')}")
    print("-" * 70)
