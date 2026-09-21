#!/usr/bin/env python
"""Build the zip files the Kaggle workflow needs -- with '/' separators.

**Why this script exists.**  Windows' built-in "Send to > Compressed (zipped)
folder" stores entry names with backslashes (``data\\raw\\ADI\\a.png``).  Those
archives unpack into *flat files literally named* ``data\\raw\\ADI\\a.png`` on
Linux, so Kaggle's dataset shows one unusable blob instead of a directory tree.
Python's ``zipfile`` always writes forward slashes and is therefore the safe way
to package anything for Kaggle.

Archives produced (or verified) in ``--out-dir``:

1. ``histo-robust-code.zip``       -- the codebase (``src/``, ``scripts/``,
   ``configs/``, ``kaggle/``, ``docs/``, ``pyproject.toml``, ...).  Small; upload
   as dataset #1, and refresh it with a new dataset version whenever the code or
   docs change.
2. ``NCT-CRC-HE-100K-NONORM.zip``  -- source domain.  Copied from
   ``--data-dir`` when you already have Zenodo's archive (its entry names are
   already correct); only re-packaged if you ask with ``--repack-data``.
3. ``CRC-VAL-HE-7K.zip``           -- target domain, same treatment.
4. ``histo-robust-checkpoints.zip`` -- optional; bundles a previous session's
   ``checkpoints/`` for re-upload as the checkpoint dataset (use the dataset's
   **New Version** button next time, never a new dataset).

Every produced archive is re-opened and checked for backslash, absolute or
drive-letter entries; the script exits non-zero if any is found.

Usage::

    python scripts/make_zips.py --out-dir dist
    python scripts/make_zips.py --out-dir dist --checkpoints checkpoints \\
        --zip-name-version v1
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directories that must never end up inside the codebase archive.
CODE_EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".venv-test",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "data",
    "zipped_dataset",
    "dist",
    "checkpoints",
    "results",
    "logs",
    "for_upload",
    "*.egg-info",
}
#: Individual files that are not part of the codebase.
CODE_EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pth", ".pt", ".npz", ".zip", ".7z", ".rar"}
CODE_EXCLUDE_NAMES = {".DS_Store", "Thumbs.db"}


def _is_excluded(relative: Path) -> bool:
    if relative.name in CODE_EXCLUDE_NAMES:
        return True
    if relative.suffix.lower() in CODE_EXCLUDE_SUFFIXES:
        return True
    for part in relative.parts:
        if part in CODE_EXCLUDE_DIRS:
            return True
        if part.endswith(".egg-info"):
            return True
    return False


def iter_code_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _is_excluded(relative):
            continue
        yield path


def verify_forward_slashes(archive: Path) -> Tuple[int, List[str]]:
    """Return ``(n_entries, offending_entries)`` for a zip.

    An entry is offending when it stores a backslash or an absolute/drive-letter
    path -- both of which break on Linux.
    """
    offenders: List[str] = []
    count = 0
    with zipfile.ZipFile(archive, "r") as zf:
        for name in zf.namelist():
            count += 1
            if "\\" in name or name.startswith("/") or (len(name) > 1 and name[1] == ":"):
                offenders.append(name)
    return count, offenders[:10]


def zip_codebase(root: Path, out_path: Path, verbose: bool = True) -> Path:
    files = list(iter_code_files(root))
    if not files:
        raise RuntimeError(f"No files found to archive under {root}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            # arcname uses POSIX separators by construction on every platform.
            zf.write(path, arcname=path.relative_to(root).as_posix())
    if verbose:
        size_mb = out_path.stat().st_size / 1024**2
        print(f"  wrote {out_path.name}: {len(files)} files, {size_mb:.2f} MB")
    return out_path


def zip_directory(root: Path, out_path: Path, arc_prefix: str = "", verbose: bool = True) -> Path:
    """Archive a directory tree (used for the optional checkpoint bundle)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or _is_excluded(path.relative_to(root)):
                continue
            arcname = f"{arc_prefix}/{path.relative_to(root).as_posix()}" if arc_prefix else path.relative_to(root).as_posix()
            zf.write(path, arcname=arcname)
            count += 1
    if verbose:
        size_mb = out_path.stat().st_size / 1024**2
        print(f"  wrote {out_path.name}: {count} files, {size_mb:.2f} MB")
    return out_path


def stage_dataset_archive(source_zip: Path, out_path: Path) -> Path:
    """Copy a dataset archive into the output folder, then verify its entries."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.resolve() != source_zip.resolve():
        shutil.copy2(source_zip, out_path)
    return out_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--out-dir", default="dist", help="Where the zips are written.")
    parser.add_argument("--data-dir", default="zipped_dataset", help="Directory holding the Zenodo dataset zips.")
    parser.add_argument("--checkpoints", default=None, help="Directory of checkpoints to bundle for re-upload.")
    parser.add_argument("--repack-data", action="store_true", help="Re-create the dataset zips from raw folders (slow, tens of GB).")
    parser.add_argument("--raw-root", default="data/raw", help="Raw dataset folders, used only with --repack-data.")
    parser.add_argument("--include-code", action="store_true", default=True)
    parser.add_argument("--zip-name-version", default=None, help="Optional suffix, e.g. v2 -> code zip named histo-robust-code-v2.zip")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{args.zip_name_version}" if args.zip_name_version else ""

    print("=" * 74)
    print("KAGGLE ZIP BUILDER (forward slashes guaranteed)")
    print("=" * 74)

    produced: List[Path] = []

    if args.include_code:
        print("\n[1/3] codebase")
        code_zip = zip_codebase(root, out_dir / f"histo-robust-code{suffix}.zip")
        produced.append(code_zip)

    print("\n[2/3] datasets")
    data_dir = Path(args.data_dir)
    for name in ("NCT-CRC-HE-100K-NONORM.zip", "CRC-VAL-HE-7K.zip"):
        source = data_dir / name
        target = out_dir / name
        if args.repack_data:
            raw = Path(args.raw_root) / name.replace(".zip", "")
            if not raw.is_dir():
                print(f"  SKIP {name}: {raw} is not a directory")
                continue
            started = time.time()
            print(f"  repacking {raw} (this can take a long time for the 100K set) ...")
            zip_directory(raw, target, arc_prefix=raw.name)
            print(f"  repacked in {(time.time() - started) / 60:.1f} min")
            produced.append(target)
        elif source.exists():
            stage_dataset_archive(source, target)
            print(f"  copied {name} ({(target.stat().st_size / 1024**2):.0f} MB)")
            produced.append(target)
        else:
            print(
                f"  MISSING {source}. Download it from Zenodo record 1214456 "
                f"(see docs/dataset_card.md section 3) or pass --data-dir/--repack-data."
            )

    print("\n[3/3] checkpoints")
    if args.checkpoints:
        ckpt_dir = Path(args.checkpoints)
        if ckpt_dir.is_dir():
            ckpt_zip = zip_directory(
                ckpt_dir, out_dir / f"histo-robust-checkpoints{suffix}.zip", arc_prefix="checkpoints"
            )
            produced.append(ckpt_zip)
        else:
            print(f"  SKIP: {ckpt_dir} is not a directory")
    else:
        print("  skipped (pass --checkpoints <dir> to bundle a previous session)")

    print("\n" + "-" * 74)
    print("VERIFYING entry separators (backslash entries break Kaggle extraction)")
    print("-" * 74)
    exit_code = 0
    for archive in produced:
        count, offenders = verify_forward_slashes(archive)
        status = "OK  " if not offenders else "BAD "
        print(f"  [{status}] {archive.name}: {count} entries")
        if offenders:
            exit_code = 1
            for name in offenders:
                print(f"         offending entry: {name!r}")

    print("\n" + "=" * 74)
    print("NEXT STEPS")
    print("=" * 74)
    print("  1. Upload the codebase zip as Kaggle dataset 'histo-robust-code'.")
    print("  2. Upload NCT-CRC-HE-100K-NONORM.zip as 'nct-crc-he-100k-nonorm'.")
    print("  3. Upload CRC-VAL-HE-7K.zip as 'crc-val-he-7k'.")
    print("  4. After the first 10.5 h session, download the checkpoints output and")
    print("     upload it as 'histo-robust-checkpoints' -- then use 'New Version'")
    print("     on that dataset for every later session (see docs/kaggle_guide.md).")
    print("=" * 74)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
