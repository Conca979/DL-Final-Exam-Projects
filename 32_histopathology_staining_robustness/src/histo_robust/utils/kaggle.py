"""Kaggle runtime helpers: logging, GPU assertion, input discovery, extraction.

Everything a Kaggle notebook cell needs to go from "three zipped datasets are
mounted" to "the scripts can run":

* :func:`configure_logging` / :func:`log_environment` -- one log file per run,
  plus a GPU/driver/version banner that makes a failed session diagnosable;
* :func:`assert_gpu` -- hard failure if the accelerator is missing or invisible
  (a CPU-only 10-hour run is worse than no run at all);
* :func:`discover_inputs` -- recursively classify ``/kaggle/input`` contents into
  ``codebase`` / ``dataset`` / ``checkpoints`` by name and content, because the
  mount slug is chosen by the human and cannot be hardcoded;
* :func:`safe_extract` -- forward-slash-safe zip extraction that also flattens
  the single top-level directory Windows' "Send to > Compressed folder" creates;
* :func:`ensure_dataset_layout` -- normalises the mounted dataset into the
  ``data/raw/<DATASET>/<CLASS>/*.png`` layout the split script expects.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "configure_logging",
    "log_environment",
    "gpu_report",
    "assert_gpu",
    "discover_inputs",
    "safe_extract",
    "extract_archives",
    "ensure_dataset_layout",
    "find_checkpoint_dirs",
    "setup_result_dirs",
    "InputInventory",
    "SOURCE_DATASET_NAME",
    "TARGET_DATASET_NAME",
    "CLASS_NAMES",
]

SOURCE_DATASET_NAME = "NCT-CRC-HE-100K-NONORM"
TARGET_DATASET_NAME = "CRC-VAL-HE-7K"
CLASS_NAMES: tuple[str, ...] = (
    "ADI",
    "BACK",
    "DEB",
    "LYM",
    "MUC",
    "MUS",
    "NORM",
    "STR",
    "TUM",
)


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
def configure_logging(
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
    also_stdout: bool = True,
) -> logging.Logger:
    """Idempotent root-logger configuration writing to stdout and a file."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, "_histo_robust", False):
            root.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    if also_stdout:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        stream._histo_robust = True  # type: ignore[attr-defined]
        root.addHandler(stream)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        file_handler._histo_robust = True  # type: ignore[attr-defined]
        root.addHandler(file_handler)
        logger.info("Logging to %s", path)

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return root


# ----------------------------------------------------------------------------
# Environment / GPU
# ----------------------------------------------------------------------------
def gpu_report() -> Dict[str, Any]:
    """Collect a diagnostic snapshot of the compute environment."""
    report: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "kaggle_env": {
            k: v for k, v in sorted(os.environ.items()) if k.startswith("KAGGLE_")
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_version"] = getattr(torch.version, "cuda", None)
        report["device_count"] = int(torch.cuda.device_count())
        devices = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
        report["devices"] = devices
        report["amp_supported"] = bool(
            torch.cuda.is_available() and torch.cuda.device_count() > 0
        )
    except Exception as exc:  # noqa: BLE001
        report["torch_error"] = f"{type(exc).__name__}: {exc}"
    return report


def log_environment(log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    log = log or logger
    report = gpu_report()
    log.info("=" * 78)
    log.info("ENVIRONMENT SNAPSHOT")
    log.info("  python          : %s", report.get("python"))
    log.info("  platform        : %s", report.get("platform"))
    log.info("  torch           : %s (cuda %s)", report.get("torch"), report.get("cuda_version"))
    log.info("  cuda available  : %s", report.get("cuda_available"))
    for device in report.get("devices", []) or []:
        log.info(
            "  gpu[%d]          : %s (%.1f GB, sm_%s)",
            device["index"],
            device["name"],
            device["total_memory_gb"],
            device["capability"],
        )
    if "torch_error" in report:
        log.error("  torch error     : %s", report["torch_error"])
    log.info("=" * 78)
    return report


def assert_gpu(require: bool = True, min_devices: int = 1) -> Dict[str, Any]:
    """Hard-assert that at least ``min_devices`` CUDA devices are visible."""
    report = gpu_report()
    if not require:
        logger.warning("GPU assertion skipped (assert_gpu(require=False))")
        return report

    if not report.get("cuda_available"):
        raise RuntimeError(
            "No CUDA device is available. On Kaggle: (1) open the notebook "
            "Settings panel, (2) set Accelerator to 'GPU P100' or 'GPU T4 x2', "
            "(3) re-run this cell. A CPU-only run of this pipeline is not viable."
        )
    if int(report.get("device_count", 0)) < int(min_devices):
        raise RuntimeError(
            f"Expected at least {min_devices} CUDA device(s), found "
            f"{report.get('device_count', 0)}."
        )
    logger.info(
        "GPU assertion passed: %d device(s), %s",
        report["device_count"],
        [d["name"] for d in report.get("devices", [])],
    )
    return report


# ----------------------------------------------------------------------------
# /kaggle/input discovery
# ----------------------------------------------------------------------------
@dataclass
class InputInventory:
    """What we managed to find under ``/kaggle/input``."""

    codebase: Optional[Path] = None
    source_dataset: Optional[Path] = None
    target_dataset: Optional[Path] = None
    checkpoints: List[Path] = field(default_factory=list)
    archives: List[Path] = field(default_factory=list)
    weights_dirs: List[Path] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "codebase": str(self.codebase) if self.codebase else None,
            "source_dataset": str(self.source_dataset) if self.source_dataset else None,
            "target_dataset": str(self.target_dataset) if self.target_dataset else None,
            "checkpoints": [str(p) for p in self.checkpoints],
            "archives": [str(p) for p in self.archives],
            "weights_dirs": [str(p) for p in self.weights_dirs],
            "notes": list(self.notes),
        }


def _has_class_folders(directory: Path, class_names: Sequence[str] = CLASS_NAMES) -> int:
    """Count how many canonical class folders exist (directly or one level down)."""
    if not directory.is_dir():
        return 0
    wanted = {c.upper() for c in class_names}
    found = 0
    checked = 0
    for child in directory.iterdir():
        if not child.is_dir():
            continue
        checked += 1
        if child.name.upper() in wanted:
            found += 1
        elif checked <= 20:
            for grand in child.iterdir() if child.is_dir() else []:
                if grand.is_dir() and grand.name.upper() in wanted:
                    found += 1
                    break
    return found


def discover_inputs(
    input_root: str | Path = "/kaggle/input",
    class_names: Sequence[str] = CLASS_NAMES,
    codebase_markers: Sequence[str] = ("pyproject.toml", "src/histo_robust", "scripts/train.py"),
) -> InputInventory:
    """Recursively classify mounted datasets into codebase / data / checkpoints.

    Detection rules (deliberately structural, not slug-based):

    * **dataset**     -- directory whose children include >=6 of the 9 class
      folders (works for a raw folder dataset *and* for a zip-not-yet-extracted
      mount, because the archive name is matched too);
    * **codebase**    -- directory containing ``pyproject.toml`` and ``src/``;
    * **checkpoints** -- directory containing ``last.pt``/``best.pt``/``step_*.pt``
      or an experiment subdirectory that does;
    * **weights**     -- directory containing ``config.json`` next to
      ``model.safetensors``/``pytorch_model.bin`` (offline HF/timm weights).
    """
    root = Path(input_root)
    inventory = InputInventory()
    if not root.exists():
        inventory.notes.append(f"{root} does not exist (are you running on Kaggle?)")
        return inventory

    for path in sorted(root.rglob("*")):
        name = path.name
        upper = name.upper()

        if path.is_file() and path.suffix.lower() == ".zip":
            inventory.archives.append(path)
            continue
        if not path.is_dir():
            continue

        # --- codebase -------------------------------------------------
        if inventory.codebase is None and all(
            (path / marker).exists() for marker in codebase_markers
        ):
            inventory.codebase = path
            inventory.notes.append(f"codebase detected at {path}")
            continue

        # --- checkpoints ---------------------------------------------
        if any(path.glob("*.pt")) or any(
            p.is_dir() and any(p.glob("*.pt")) for p in path.iterdir()
        ):
            if path not in inventory.checkpoints:
                inventory.checkpoints.append(path)
                inventory.notes.append(f"checkpoints detected at {path}")
            continue

        # --- HF/timm weights -----------------------------------------
        if (path / "config.json").exists() and (
            (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
        ):
            inventory.weights_dirs.append(path)
            inventory.notes.append(f"offline model weights detected at {path}")
            continue

        # --- datasets -------------------------------------------------
        if SOURCE_DATASET_NAME.lower() in path.name.lower():
            if inventory.source_dataset is None:
                inventory.source_dataset = path
                continue
        if TARGET_DATASET_NAME.lower() in path.name.lower():
            if inventory.target_dataset is None:
                inventory.target_dataset = path
                continue

        n_classes = _has_class_folders(path, class_names)
        if n_classes >= 6:
            if path.name.upper() == TARGET_DATASET_NAME.upper() and inventory.target_dataset is None:
                inventory.target_dataset = path
            elif inventory.source_dataset is None:
                inventory.source_dataset = path
            elif inventory.target_dataset is None:
                inventory.target_dataset = path
            else:
                inventory.notes.append(f"extra class-folder directory ignored: {path}")

    if inventory.source_dataset is None or inventory.target_dataset is None:
        # Also check inside not-yet-extracted archives by name.
        for archive in inventory.archives:
            if SOURCE_DATASET_NAME.lower() in archive.name.lower() and inventory.source_dataset is None:
                inventory.notes.append(
                    f"{archive.name} is still zipped; extract_archives() will unpack it"
                )
            if TARGET_DATASET_NAME.lower() in archive.name.lower() and inventory.target_dataset is None:
                inventory.notes.append(
                    f"{archive.name} is still zipped; extract_archives() will unpack it"
                )
    return inventory


# ----------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------
def safe_extract(archive: str | Path, dest: str | Path) -> Path:
    """Extract a zip with traversal protection, flattening one wrapper folder.

    Handles both the Linux ``zip -r`` layout (correct forward slashes) and the
    Windows "Compressed folder" layout (a redundant top-level directory, and
    backslashes in the stored names).  Returned path is the *content* directory.
    """
    archive = Path(archive)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive, "r") as zf:
        members = zf.namelist()
        for member in members:
            normalised = member.replace("\\", "/")
            target = (dest / normalised).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"Unsafe path in {archive}: {member}")
        zf.extractall(dest)

    # Flatten a single-wrapper-directory archive (Windows default behaviour).
    entries = [p for p in dest.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        wrapper = entries[0]
        inner = list(wrapper.iterdir())
        if inner and not any((dest / p.name).exists() for p in inner if p.name != wrapper.name):
            for item in inner:
                shutil.move(str(item), str(dest / item.name))
            wrapper.rmdir()
            logger.info("Flattened wrapper directory '%s' inside %s", wrapper.name, dest)
    return dest


def extract_archives(
    archives: Iterable[str | Path], dest: str | Path, cleanup: bool = False
) -> List[Path]:
    extracted: List[Path] = []
    for archive in archives:
        archive = Path(archive)
        target = Path(dest) / archive.stem
        logger.info("Extracting %s -> %s", archive, target)
        extracted.append(safe_extract(archive, target))
        if cleanup:
            try:
                archive.unlink()
            except OSError:
                pass
    return extracted


def ensure_dataset_layout(
    source_dir: str | Path,
    target_dir: str | Path,
    dest_root: str | Path,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Dict[str, Path]:
    """Symlink/copy the mounted class folders into ``<dest_root>/<DATASET>/<CLASS>``.

    Kaggle's ``/kaggle/input`` is read-only, but nothing in the pipeline needs to
    write to the images, so the default is a **symlink** (instant, zero extra
    disk).  ``copy=True`` falls back to real copies when the filesystem refuses
    symlinks.
    """
    dest_root = Path(dest_root)
    mapping: Dict[str, Path] = {}

    def _place(name: str, source: Path) -> Path:
        target_root = dest_root / name
        target_root.mkdir(parents=True, exist_ok=True)
        for class_name in class_names:
            found = None
            for candidate in (source, *[p for p in source.iterdir() if p.is_dir()]):
                direct = candidate / class_name
                if direct.is_dir():
                    found = direct
                    break
            if found is None:
                logger.warning("Class folder %s missing under %s", class_name, source)
                continue
            link = target_root / class_name
            if link.exists() or link.is_symlink():
                mapping[f"{name}/{class_name}"] = link
                continue
            try:
                link.symlink_to(found.resolve(), target_is_directory=True)
            except OSError as exc:
                logger.warning(
                    "Symlink failed for %s (%s); falling back to a copy", link, exc
                )
                shutil.copytree(found, link)
            mapping[f"{name}/{class_name}"] = link
        return target_root

    _place(SOURCE_DATASET_NAME, Path(source_dir))
    _place(TARGET_DATASET_NAME, Path(target_dir))
    return mapping


def find_checkpoint_dirs(search_roots: Sequence[str | Path]) -> List[Path]:
    """All directories that look like a run/experiment checkpoint folder."""
    found: List[Path] = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        if any(root.glob("*.pt")):
            found.append(root)
        for child in root.rglob("*"):
            if child.is_dir() and any(child.glob("*.pt")):
                found.append(child)
    # De-duplicate while preserving order, and drop nested duplicates.
    unique: List[Path] = []
    for path in found:
        if not any(path != other and other in path.parents for other in unique):
            if path not in unique:
                unique.append(path)
    return unique


# ----------------------------------------------------------------------------
# Result directories
# ----------------------------------------------------------------------------
def setup_result_dirs(working_root: str | Path = "/kaggle/working") -> Dict[str, Path]:
    """Create the canonical output tree and return it as a dict of paths."""
    root = Path(working_root)
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "results": root / "results",
        "metrics": root / "results" / "metrics",
        "figures": root / "results" / "figures",
        "confusion_matrices": root / "results" / "figures" / "confusion_matrices",
        "samples": root / "results" / "samples",
        "logs": root / "logs",
        "for_upload": root / "for_upload",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def disk_usage_report(paths: Sequence[str | Path] = ("/kaggle/working",)) -> Dict[str, Any]:
    report: Dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for path in paths:
        path = Path(path)
        try:
            usage = shutil.disk_usage(str(path))
            report[str(path)] = {
                "total_gb": round(usage.total / 1024**3, 2),
                "used_gb": round(usage.used / 1024**3, 2),
                "free_gb": round(usage.free / 1024**3, 2),
            }
        except OSError as exc:
            report[str(path)] = {"error": str(exc)}
    try:
        result = subprocess.run(
            ["du", "-sh", "/kaggle/working"], capture_output=True, text=True, timeout=60
        )
        report["du_kaggle_working"] = result.stdout.strip()
    except Exception:  # noqa: BLE001 - `du` is a nicety, never required
        pass
    return report
