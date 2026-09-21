#!/usr/bin/env python
"""Generate the Kaggle notebook JSON files (kept in version control as source).

Writing the notebooks programmatically keeps them readable as Python here and
guarantees valid ``.ipynb`` JSON (nbformat 4.5, no execution counts) without
hand-editing escaped strings.

    python kaggle/build_notebooks.py

Produced:

* ``kaggle/notebook_01_run_all_ablations.ipynb`` -- the single notebook that is
  run with "Save & Run All (Commit)".  It sets up the environment, mounts the
  three datasets, prepares the splits, benchmarks the pipeline, runs the
  ablation matrix under a 10.5-hour time box, evaluates ID vs OOD, and bundles
  the checkpoints for re-upload.
* ``kaggle/notebook_02_results_tally.ipynb`` -- offline results aggregation
  (works on the downloaded ``results/`` folder, no GPU needed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

KAGGLE_DIR = Path(__file__).resolve().parent


def code_cell(source: str, tags: List[str] | None = None) -> Dict[str, Any]:
    cell: Dict[str, Any] = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }
    if tags:
        cell["metadata"]["tags"] = tags
    return cell


def md_cell(source: str) -> Dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def notebook(cells: List[Dict[str, Any]], title: str) -> Dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# Notebook 1 -- the full Kaggle run
# ---------------------------------------------------------------------------
CELL_00_MD = """
# Robust Histopathology Classification under Staining Variations
### `NCT-CRC-HE-100K-NONORM` (in-domain) -> `CRC-VAL-HE-7K` (out-of-domain)

This notebook runs the full 13-cell ablation matrix from `docs/PLAN.md` under Kaggle's
hard limits:

| Kaggle limit | How this notebook handles it |
| :--- | :--- |
| 12 h session kill, no warning | Training stops itself at **630 min (10.5 h)** and also reserves **20 min** before the session deadline; it always writes `last.pt` + `best.pt` before exiting. |
| 20 GB `/kaggle/working` cap | `--save-every` + `--keep-last 3` pruning plus a per-cell GB quota; the setup cell measures free space and refuses to over-commit. |
| Web uploader breaks on folders | Everything arrives as three `.zip` Kaggle datasets (codebase / data / checkpoints), extracted here with forward-slash-safe paths. |
| No disk between sessions | Resume is automatic: the newest `last.pt` in the mounted checkpoint dataset is copied back and training continues from the stored epoch/step. |

**Before running:** attach the datasets and pick the accelerator.
See `docs/kaggle_guide.md` for the exact click-by-click procedure, and start at
`docs/README.md` for the rest of the documentation (`docs/PLAN.md` design,
`docs/RUN_LOG.md` register, `docs/RESULTS.md` tables, `docs/APPENDICES.md`
artifact/config/CLI reference).

* Accelerator: **GPU P100** or **GPU T4 x2**
* Internet: **ON** for the first run (downloads pretrained weights). OFF is fine
  afterwards if you mount the weights as a dataset.
* Datasets: `histo-robust-code`, `nct-crc-he-100k-nonorm`, `crc-val-he-7k`,
  and (from session 2 onwards) `histo-robust-checkpoints`.
"""

CELL_01_SETUP = """
# =============================================================================
# CELL 1 -- SETUP: locate + extract the codebase, install deps, assert the GPU
# =============================================================================
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
REPO = WORKING / "histo-robust"
SESSION_START = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
os.environ["KAGGLE_SESSION_START"] = SESSION_START

print("=" * 78)
print("SETUP")
print("=" * 78)
print(f"session start (UTC): {SESSION_START}")
print(f"cwd                : {os.getcwd()}")
print(f"python             : {sys.version.split()[0]}")

# --- 1a. extract the codebase zip (search /kaggle/input recursively) ---------
def find_codebase_zip(root: Path) -> Path | None:
    if not root.exists():
        return None
    preferred = [
        root / "histo-robust-code" / "histo-robust-code.zip",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate
    hits = sorted(root.rglob("histo-robust-code*.zip"))
    if hits:
        return hits[0]
    for candidate in sorted(root.rglob("*.zip")):
        try:
            with zipfile.ZipFile(candidate) as zf:
                names = zf.namelist()
        except zipfile.BadZipFile:
            continue
        if any(n.replace("\\\\", "/").endswith("pyproject.toml") for n in names):
            return candidate
    return None


def safe_extract(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            normalised = member.replace("\\\\", "/")
            target = (dest / normalised).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"unsafe archive entry: {member}")
        zf.extractall(dest)
    # Flatten a single wrapper directory (the Windows "Compressed folder" shape).
    entries = [p for p in dest.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        wrapper = entries[0]
        if any((wrapper / marker).exists() for marker in ("pyproject.toml", "src", "scripts")):
            for item in list(wrapper.iterdir()):
                shutil.move(str(item), str(dest / item.name))
            wrapper.rmdir()
    return dest


if (REPO / "pyproject.toml").exists():
    print(f"[1a] codebase already present at {REPO}")
else:
    code_zip = find_codebase_zip(INPUT)
    if code_zip is None:
        raise SystemExit(
            "FATAL: could not find the codebase zip under /kaggle/input.\\n"
            "Attach the 'histo-robust-code' dataset (see docs/kaggle_guide.md), then re-run."
        )
    print(f"[1a] extracting {code_zip} -> {REPO}")
    safe_extract(code_zip, REPO)

for marker in ("pyproject.toml", "src/histo_robust", "scripts/run_all_ablations.py"):
    assert (REPO / marker).exists(), f"codebase is incomplete: missing {marker}"
print(f"[1a] codebase OK: {REPO}")

# --- 1b. GPU assertion ------------------------------------------------------
import torch

print(f"[1b] torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "FATAL: no CUDA device. Open the notebook Settings panel, set Accelerator "
        "to 'GPU P100' or 'GPU T4 x2', save, and re-run this cell."
    )
for idx in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(idx)
    print(f"     gpu[{idx}]: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
print("[1b] GPU assertion PASSED")

# --- 1c. install the project (deps are checked, not blindly reinstalled) -----
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "kaggle"))
import notebook_helpers  # noqa: E402

report = notebook_helpers.ensure_dependencies(str(REPO), allow_install=True)
notebook_helpers.print_report(report)
if report["missing_required"]:
    raise SystemExit(f"FATAL: required packages still missing: {report['missing_required']}")

# --- 1d. disk budget --------------------------------------------------------
usage = shutil.disk_usage(str(WORKING))
free_gb = usage.free / 1024**3
print(f"[1d] /kaggle/working free space: {free_gb:.1f} GB (Kaggle caps the output at ~20 GB)")
if free_gb < 6:
    print("     WARNING: less than 6 GB free. Enable the normalisation cache only")
    print("     for a single method, and lower checkpoints.keep_last.")

print("=" * 78)
print("SETUP COMPLETE")
print("=" * 78)
"""

CELL_02_DATA = """
# =============================================================================
# CELL 2 -- DATA + CHECKPOINT MOUNT
#   * locate the source and target datasets under /kaggle/input
#   * extract them into /kaggle/working (only if space allows)
#   * copy any previous session's checkpoints back so training can resume
# =============================================================================
import json
import shutil
import zipfile
from pathlib import Path

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
REPO = WORKING / "histo-robust"
DATA_ROOT = WORKING / "data" / "raw"
SPLITS_DIR = WORKING / "data" / "processed" / "splits"
TEMPLATE_DIR = WORKING / "data" / "processed" / "templates"
CHECKPOINT_DIR = WORKING / "checkpoints"
RESULTS_DIR = WORKING / "results"

CLASS_NAMES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
SOURCE_NAME = "NCT-CRC-HE-100K-NONORM"
TARGET_NAME = "CRC-VAL-HE-7K"

for path in (DATA_ROOT, SPLITS_DIR, TEMPLATE_DIR, CHECKPOINT_DIR, RESULTS_DIR):
    path.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("DATA + CHECKPOINT MOUNT")
print("=" * 78)


def count_class_dirs(root: Path) -> int:
    if not root.is_dir():
        return 0
    checked, found = 0, 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        checked += 1
        if child.name.upper() in CLASS_NAMES:
            found += 1
        elif checked <= 25:
            if any(g.is_dir() and g.name.upper() in CLASS_NAMES for g in child.iterdir()):
                found += 1
    return found


def safe_extract(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            normalised = member.replace("\\\\", "/")
            if not str((dest / normalised).resolve()).startswith(str(dest.resolve())):
                raise RuntimeError(f"unsafe archive entry: {member}")
        zf.extractall(dest)
    entries = [p for p in dest.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        wrapper = entries[0]
        if count_class_dirs(wrapper) >= 6 or wrapper.name in (SOURCE_NAME, TARGET_NAME):
            for item in list(wrapper.iterdir()):
                shutil.move(str(item), str(dest / item.name))
            wrapper.rmdir()
    return dest


# --- 2a. find the datasets --------------------------------------------------
found = {"source": None, "target": None}
archives = {"source": None, "target": None}
for path in sorted(INPUT.rglob("*")):
    name = path.name.lower()
    if SOURCE_NAME.lower() in name:
        archives["source"] = path if path.is_file() else archives["source"]
        if path.is_dir() and found["source"] is None and count_class_dirs(path) >= 6:
            found["source"] = path
    if TARGET_NAME.lower() in name:
        archives["target"] = path if path.is_file() else archives["target"]
        if path.is_dir() and found["target"] is None and count_class_dirs(path) >= 6:
            found["target"] = path

# Fall back to content-based detection when the slugs were renamed.
for path in sorted(INPUT.rglob("*")):
    if not path.is_dir():
        continue
    n_classes = count_class_dirs(path)
    if n_classes < 6:
        continue
    target_like = TARGET_NAME.lower() in path.name.lower() or "7k" in path.name.lower()
    if target_like and found["target"] is None:
        found["target"] = path
    elif found["source"] is None:
        found["source"] = path
    elif found["target"] is None:
        found["target"] = path

print("[2a] located:")
print(f"     source (in-domain) : {found['source']}")
print(f"     target (OOD)       : {found['target']}")

# --- 2b. use /kaggle/input directly (read-only is fine, saves ~12 GB) -------
# Nothing in the pipeline writes to the images, so pointing the split builder at
# /kaggle/input avoids extracting 11 GB of PNGs into the 20 GB output quota.
SOURCE_ROOT = found["source"]
TARGET_ROOT = found["target"]

if SOURCE_ROOT is None and archives["source"] is not None:
    print(f"[2b] source is still zipped ({archives['source'].name}); extracting ...")
    SOURCE_ROOT = safe_extract(archives["source"], DATA_ROOT / SOURCE_NAME)
if TARGET_ROOT is None and archives["target"] is not None:
    print(f"[2b] target is still zipped ({archives['target'].name}); extracting ...")
    TARGET_ROOT = safe_extract(archives["target"], DATA_ROOT / TARGET_NAME)

if SOURCE_ROOT is None or TARGET_ROOT is None:
    raise SystemExit(
        "FATAL: could not locate the datasets. Attach 'nct-crc-he-100k-nonorm' and "
        "'crc-val-he-7k' as Kaggle datasets (see docs/kaggle_guide.md). Found dirs under "
        f"/kaggle/input: {[str(p) for p in sorted(INPUT.glob('*'))]}"
    )

for label, root in (("source", SOURCE_ROOT), ("target", TARGET_ROOT)):
    counts = {c: len(list((root / c).glob("*"))) for c in CLASS_NAMES if (root / c).is_dir()}
    missing = [c for c in CLASS_NAMES if c not in counts]
    print(f"[2b] {label}: {sum(counts.values())} images across {len(counts)} classes"
          + (f" | MISSING {missing}" if missing else ""))
    if missing:
        raise SystemExit(f"FATAL: {label} domain is missing class folders {missing} under {root}")

# --- 2c. previous checkpoints ----------------------------------------------
resume_dirs = []
for path in sorted(INPUT.rglob("*")):
    if path.is_dir() and any(path.glob("*.pt")):
        resume_dirs.append(path)
copied = 0
for directory in resume_dirs:
    relative = directory.relative_to(INPUT)
    destination = CHECKPOINT_DIR / Path(*relative.parts[1:]) if len(relative.parts) > 1 else CHECKPOINT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    for item in directory.rglob("*"):
        if not item.is_file():
            continue
        target_file = destination / item.relative_to(directory)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target_file)
        copied += 1
print(f"[2c] checkpoint files copied into {CHECKPOINT_DIR}: {copied}")
if copied == 0:
    print("     (none found -- this is a fresh start; that is expected on session 1)")

existing_runs = sorted(p.name for p in CHECKPOINT_DIR.iterdir() if p.is_dir()) if CHECKPOINT_DIR.exists() else []
print(f"[2c] experiment folders available to resume: {existing_runs}")

free_gb = shutil.disk_usage(str(WORKING)).free / 1024**3
print(f"[2d] free space in /kaggle/working: {free_gb:.1f} GB")
print("=" * 78)
print("DATA MOUNT COMPLETE")
print("=" * 78)
"""

CELL_03_SPLITS = """
# =============================================================================
# CELL 3 -- SPLITS + CANONICAL STAIN REFERENCE
#   stratified 70/15/15 source split (seed 42) + the full target set as test_ood
#   and one canonical H&E reference tile chosen from the TRAINING split only
# =============================================================================
import json
import subprocess
import sys
from pathlib import Path

WORKING = Path("/kaggle/working")
REPO = WORKING / "histo-robust"
SPLITS_DIR = WORKING / "data" / "processed" / "splits"
TEMPLATE_DIR = WORKING / "data" / "processed" / "templates"
TEMPLATE = TEMPLATE_DIR / "reference_stain.png"

SUBSET = 25000          # 0 + FULL=True trains on all 100,000 patches
FULL = False
SEED = 42

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

command = [
    sys.executable, str(REPO / "scripts" / "prepare_splits.py"),
    "--data-root", str(SOURCE_ROOT),
    "--target-root", str(TARGET_ROOT),
    "--out-dir", str(SPLITS_DIR),
    "--seed", str(SEED),
    "--reference-out", str(TEMPLATE),
]
if FULL:
    command.append("--full")
else:
    command += ["--subset", str(SUBSET)]

print("$ " + " ".join(command))
result = subprocess.run(command, text=True)
if result.returncode != 0:
    raise SystemExit(f"prepare_splits.py failed with exit code {result.returncode}")

summary = json.loads((SPLITS_DIR / "split_summary.json").read_text())
print("\\nsplit summary:")
for name, info in summary["splits"].items():
    print(f"  {name:<9} {info['n_samples']:>7} rows  domains={info['domains']}")
print(f"  canonical reference tile: {summary['reference_source_image']}")
print(f"  subset: {summary['subset']} | seed: {summary['seed']}")

# Hard firewall assertions -- fail loudly rather than produce invalid results.
assert set(summary["splits"]["train"]["domains"]) == {"source"}, "train must be source-only"
assert set(summary["splits"]["val_id"]["domains"]) == {"source"}, "val_id must be source-only"
assert set(summary["splits"]["test_ood"]["domains"]) == {"target"}, "test_ood must be target-only"
print("\\nDOMAIN FIREWALL: PASSED (train/val_id are source-only, test_ood is target-only)")
print("=" * 78)
"""

CELL_04_PREFLIGHT = """
# =============================================================================
# CELL 4 -- PRE-FLIGHT BENCHMARK (run before committing 10 hours)
#   measures DataLoader throughput and a real forward/backward step, then
#   extrapolates minutes-per-epoch and issues a CPU-bound / idle-GPU verdict
# =============================================================================
import json
import subprocess
import sys
from pathlib import Path

WORKING = Path("/kaggle/working")
REPO = WORKING / "histo-robust"
RESULTS_DIR = WORKING / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREFLIGHT_CONFIG = REPO / "configs" / "experiments" / "exp03_norm_macenko_resnet50.yaml"
STEPS = 10

command = [
    sys.executable, str(REPO / "scripts" / "smoke_test.py"),
    "--config", str(PREFLIGHT_CONFIG),
    "--steps", str(STEPS),
    "--output", str(RESULTS_DIR / "metrics" / "preflight_benchmark.json"),
]
print("$ " + " ".join(command))
result = subprocess.run(command, text=True)
if result.returncode != 0:
    raise SystemExit(f"smoke_test.py failed with exit code {result.returncode}")

benchmark = json.loads((RESULTS_DIR / "metrics" / "preflight_benchmark.json").read_text())
print("\\nPRE-FLIGHT VERDICT:", benchmark["extrapolation"]["verdict"])
print("  minutes/epoch   :", benchmark["extrapolation"]["minutes_per_epoch"])
print("  epochs in budget:", benchmark["extrapolation"]["epochs_fitting_in_budget"])
print("  advice          :", benchmark["extrapolation"]["advice"])

# 13 cells at the measured cost: decide up front whether the full matrix can fit.
matrix_hours = benchmark["extrapolation"]["full_matrix_estimate_minutes"] / 60
print(f"  full 13-cell matrix (12 epochs each): ~{matrix_hours:.1f} h")
if matrix_hours > 10:
    print("  -> will need multiple 10.5 h sessions; that is expected and supported.")
print("=" * 78)
print("PRE-FLIGHT COMPLETE -- review the verdict above before running the next cell")
print("=" * 78)
"""

CELL_04B_CACHE = """
# =============================================================================
# CELL 4b (OPTIONAL) -- PRE-NORMALISATION CACHE
#
# Macenko costs ~20-40 ms per 224x224 tile on CPU. Doing that inside the
# DataLoader is fine at num_workers=4 (the pre-flight cell reports the ratio),
# but caching removes the cost entirely and roughly doubles the number of cells
# that fit in one session.
#
# Skip this cell unless the pre-flight verdict says CPU-BOUND, or unless you want
# the extra headroom. When the cache exists, CELL 5 reads it automatically.
# =============================================================================
import subprocess
import sys
from pathlib import Path

WORKING = Path("/kaggle/working")
REPO = WORKING / "histo-robust"
SPLITS_DIR = WORKING / "data" / "processed" / "splits"
TEMPLATE = WORKING / "data" / "processed" / "templates" / "reference_stain.png"
CACHE_ROOT = WORKING / "data" / "processed" / "cache"

BUILD_CACHE = False          # set True to build the cache in this session
NORMALIZATION = "macenko"    # "macenko" or "reinhard"
WORKERS = 4                  # Kaggle gives 4 vCPU

if BUILD_CACHE:
    command = [
        sys.executable, str(REPO / "scripts" / "preprocess_normalize.py"),
        "--normalization", NORMALIZATION,
        "--splits-dir", str(SPLITS_DIR),
        "--out-dir", str(CACHE_ROOT),
        "--reference", str(TEMPLATE),
        "--workers", str(WORKERS),
    ]
    print("$ " + " ".join(command))
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        raise SystemExit(f"preprocess_normalize.py failed with exit code {result.returncode}")
    CACHE_DIR = CACHE_ROOT / NORMALIZATION
    print(f"\\ncache ready: {CACHE_DIR}")
    print("Set NORMALIZATION_CACHE in CELL 5 to this directory:")
    print(f'  NORMALIZATION_CACHE = WORKING / "data" / "processed" / "cache" / "{NORMALIZATION}"')
else:
    print("cache build skipped (BUILD_CACHE=False)")
"""

CELL_05_RUN = """
# =============================================================================
# CELL 5 -- RUN THE ABLATION MATRIX  (this is the 10.5-hour cell)
#
#   * skips cells already completed (so re-running after a session cut-off
#     continues rather than restarts)
#   * splits the remaining wall clock across the remaining cells
#   * stops at 630 min (or 20 min before the 12-hour kill) and writes last.pt
# =============================================================================
import json
import os
import subprocess
import sys
from pathlib import Path

WORKING = Path("/kaggle/working")
REPO = WORKING / "histo-robust"
CHECKPOINT_DIR = WORKING / "checkpoints"
RESULTS_DIR = WORKING / "results"

# --- run configuration ------------------------------------------------------
EXPERIMENTS = None          # e.g. ["EXP-01", "EXP-02"] for a partial run; None = all pending
SESSION_MINUTES = 700       # Kaggle kills at 720; keep headroom
RESERVE_MINUTES = 20        # held back for checkpoint flush + zipping
PER_EXP_MINUTES = 75        # cap per ablation cell
MIN_MINUTES = 6             # do not start a cell with less than this left
EVAL_TEST_OOD = True        # run the CRC-VAL-HE-7K audit after each cell
FORCE = False               # True re-runs cells already marked completed

# Optional speed-up: read pre-normalised tiles if you built them in CELL 4b.
NORMALIZATION_CACHE = None  # e.g. WORKING / "data/processed/cache/macenko"
DRY_RUN = False             # True prints the plan and allocated budget, then exits

command = [
    sys.executable, str(REPO / "scripts" / "run_all_ablations.py"),
    "--checkpoint-dir", str(CHECKPOINT_DIR),
    "--results-dir", str(RESULTS_DIR),
    "--session-start", SESSION_START,
    "--session-minutes", str(SESSION_MINUTES),
    "--reserve-minutes", str(RESERVE_MINUTES),
    "--per-exp-minutes", str(PER_EXP_MINUTES),
    "--min-minutes", str(MIN_MINUTES),
]
if EXPERIMENTS:
    command += ["--experiments", *EXPERIMENTS]
if EVAL_TEST_OOD:
    command.append("--eval-test-ood")
if FORCE:
    command.append("--force")
if NORMALIZATION_CACHE:
    command += ["--normalization-cache", str(NORMALIZATION_CACHE)]
if DRY_RUN:
    command.append("--dry-run")

print("$ " + " ".join(command))
print("-" * 78)
result = subprocess.run(command, text=True)
print("-" * 78)
if result.returncode != 0:
    print(f"run_all_ablations.py exited with code {result.returncode}")
    print("Check results/logs/run_all_ablations.log; completed cells are recorded in")
    print("results/metrics/ablation_manifest.json and survived, so re-running resumes.")
else:
    manifest_path = RESULTS_DIR / "metrics" / "ablation_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        done = [k for k, v in manifest["experiments"].items() if v.get("status") == "completed"]
        print(f"\\ncompleted cells: {len(done)}/{len(manifest['experiments'])} -> {sorted(done)}")
"""

CELL_06_ARTIFACTS = """
# =============================================================================
# CELL 6 -- ARTIFACT BUNDLE + SESSION SUMMARY
#   zips the checkpoints (for re-upload as a Kaggle dataset) and prints exactly
#   what to download before the session is recycled
# =============================================================================
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

WORKING = Path("/kaggle/working")
CHECKPOINT_DIR = WORKING / "checkpoints"
RESULTS_DIR = WORKING / "results"
UPLOAD_DIR = WORKING / "for_upload"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CARRY_OVER = True   # zip the checkpoints so the next session can resume

print("=" * 78)
print("DISK USAGE")
print("=" * 78)
subprocess.run(["du", "-sh", str(CHECKPOINT_DIR), str(RESULTS_DIR)], text=True)
usage = shutil.disk_usage(str(WORKING))
print(f"free: {usage.free / 1024**3:.1f} GB | used by /kaggle/working: {usage.used / 1024**3:.1f} GB")

if CARRY_OVER and CHECKPOINT_DIR.exists():
    archive = UPLOAD_DIR / "histo-robust-checkpoints.zip"
    print(f"\\nzipping {CHECKPOINT_DIR} -> {archive} (forward-slash entries)")
    count = 0
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path in sorted(CHECKPOINT_DIR.rglob("*")):
            if not path.is_file() or path.suffix == ".tmp":
                continue
            zf.write(path, arcname=("checkpoints/" + path.relative_to(CHECKPOINT_DIR).as_posix()))
            count += 1
    print(f"  wrote {archive.name}: {count} files, {archive.stat().st_size / 1024**2:.1f} MB")
    with zipfile.ZipFile(archive) as zf:
        bad = [n for n in zf.namelist() if "\\\\" in n]
    print(f"  separator check: {'OK' if not bad else 'BAD ' + str(bad[:3])}")

report_path = RESULTS_DIR / "metrics" / "ablation_manifest.json"
print("\\n" + "=" * 78)
print("NEXT STEPS")
print("=" * 78)
if report_path.exists():
    manifest = json.loads(report_path.read_text())
    for exp_id, entry in sorted(manifest.get("experiments", {}).items()):
        ev = entry.get("evaluation", {}) or {}
        print(
            f"  {exp_id:<7} {str(entry.get('status')):<12} "
            f"val_f1={entry.get('best_val_macro_f1')} "
            f"delta_f1={ev.get('delta_macro_f1')} rr_f1={ev.get('rr_macro_f1')}"
        )
    remaining = [k for k, v in manifest.get("experiments", {}).items() if v.get("status") != "completed"]
    if remaining:
        print(f"\\n  NOT FINISHED: {remaining}")
print(
    \"\"\"
  1. Download these outputs NOW (Kaggle recycles them when the session ends):
       /kaggle/working/for_upload/histo-robust-checkpoints.zip
       /kaggle/working/results/            (metrics, confusion matrices, logs)
  2. Go to the 'histo-robust-checkpoints' Kaggle dataset -> New Version -> upload
     the zip -> Save. Do NOT create a new dataset each session.
  3. Re-open this notebook, keep the same datasets attached, and run
     Save Version -> Save & Run All (Commit) again. Finished cells are skipped and
     the interrupted one resumes from its last.pt.
\"\"\"
)
print("=" * 78)
"""

NOTEBOOK_1_CELLS = [
    md_cell(CELL_00_MD),
    code_cell(CELL_01_SETUP),
    code_cell(CELL_02_DATA),
    code_cell(CELL_03_SPLITS),
    code_cell(CELL_04_PREFLIGHT),
    code_cell(CELL_04B_CACHE),
    code_cell(CELL_05_RUN),
    code_cell(CELL_06_ARTIFACTS),
]


# ---------------------------------------------------------------------------
# Notebook 2 -- optional results tally (CPU, offline)
# ---------------------------------------------------------------------------
TALLY_MD = """
# Results Tally (offline)

Optional companion notebook. Point `RESULTS_DIR` at the downloaded `results/`
folder (or attach it as a Kaggle dataset) and this cell produces:

* `summary_results.csv` re-ordered by stage, with the ablation matrix columns,
* `RESULTS_TABLE.md` -- a Markdown table ready to paste into `docs/RESULTS.md`,
* a check that every one of the 13 cells has both an in-domain and an
  out-of-domain number (i.e. no half-measured cells).

It computes nothing new: scores come from the `summary_results.csv` written by
the training sessions.
"""

TALLY_CODE = """
import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("/kaggle/working/results")   # or Path("/kaggle/input/<your-results-dataset>")
RESULTS_DIR = RESULTS_DIR if RESULTS_DIR.exists() else Path("results")

summary_csv = RESULTS_DIR / "metrics" / "summary_results.csv"
if not summary_csv.exists():
    raise SystemExit(f"no summary_results.csv under {RESULTS_DIR}; attach the results folder first")

frame = pd.read_csv(summary_csv)
registry = json.loads((Path("configs") / "experiments_registry.json").read_text())
order = [entry["exp_id"] for entry in registry["experiments"]]
frame["__order"] = frame["exp_id"].map({exp: i for i, exp in enumerate(order)})
frame = frame.sort_values("__order").drop(columns="__order")

display_cols = [
    c for c in [
        "exp_id", "stage", "backbone", "normalization", "augmentation", "status",
        "best_val_macro_f1", "test_id_macro_f1", "test_ood_macro_f1",
        "delta_f1", "rr_f1", "test_id_balanced_acc", "test_ood_balanced_acc",
        "delta_balanced_acc", "rr_balanced_acc",
    ] if c in frame.columns
]
print(frame[display_cols].to_string(index=False))

table = frame[display_cols].to_markdown(index=False)
out_md = RESULTS_DIR / "metrics" / "RESULTS_TABLE.md"
out_md.write_text(table + "\\n", encoding="utf-8")
print(f"\\nwrote {out_md}")

missing = frame[frame["test_ood_macro_f1"].isna()]["exp_id"].tolist() if "test_ood_macro_f1" in frame else []
print(f"cells without an OOD number: {missing if missing else 'none'}")
"""

NOTEBOOK_2_CELLS = [md_cell(TALLY_MD), code_cell(TALLY_CODE)]


def main() -> int:
    targets = {
        "notebook_01_run_all_ablations.ipynb": notebook(
            NOTEBOOK_1_CELLS, "Robust histopathology -- full Kaggle run"
        ),
        "notebook_02_results_tally.ipynb": notebook(
            NOTEBOOK_2_CELLS, "Robust histopathology -- results tally"
        ),
    }
    for name, payload in targets.items():
        path = KAGGLE_DIR / name
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(KAGGLE_DIR.parent)} ({len(payload['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
