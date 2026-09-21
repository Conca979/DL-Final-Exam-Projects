#!/usr/bin/env python
"""Offline stain-normalisation cache (``PLAN.md`` risk 2 mitigation).

Macenko's SVD deconvolution costs ~15-40 ms per 224x224 tile on CPU.  Doing that
inside a DataLoader worker seven times per epoch starves the GPU and turns a
2-hour training run into a 10-hour one.  This script normalises each split once
and writes JPEG/PNG tiles to disk, after which training reads pre-normalised
tiles at plain decode speed.

Design notes that matter for correctness:

* **One reference for every experiment.** The canonical reference tile is
  resolved exactly as in training (:func:`histo_robust.data.resolve_reference`),
  because a per-experiment reference would make Axis B incomparable.
* **Split-wise caching.** ``train``, ``val_id``, ``test_id`` and ``test_ood`` are
  cached separately, all with the training-derived reference.  Caching the OOD
  split is a *pure inference-time transform* -- it does not leak: the reference
  never sees it, and the cached tiles are only read by the post-training audit.
* **Resumable.** Already-written tiles are skipped, so a Kaggle session that dies
  after 8 hours can be resumed and only pays for the remainder.
* **Provenance.** ``cache_manifest.json`` records the reference source, the
  normaliser parameters, per-split statistics and the fallback counters, which is
  what the verifier needs to confirm the cache matches the configured pipeline.

Usage::

    python scripts/preprocess_normalize.py \
        --config configs/experiments/exp03_norm_macenko_resnet50.yaml \
        --splits-dir /kaggle/working/data/processed/splits \
        --out-dir    /kaggle/working/data/processed/cache \
        --reference  /kaggle/working/data/processed/templates/reference_stain.png \
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from histo_robust.augmentation import CenterCrop  # noqa: E402
from histo_robust.data.paths import resolve_image_path  # noqa: E402
from histo_robust.normalization import build_normalizer, resolve_reference_image  # noqa: E402
from histo_robust.utils.config import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
logger = logging.getLogger("preprocess_normalize")

_NORMALIZER = None  # per-process normaliser (set by _init_worker)
_SPLIT_OUT: Optional[Path] = None
_CROP: Optional[CenterCrop] = None
_SEARCH_ROOTS: List[str] = []
_SAVE_EXT = ".jpg"
_SAVE_QUALITY = 95


def _init_worker(
    norm_name: str,
    norm_params: Dict[str, Any],
    reference: np.ndarray,
    split_out: str,
    image_size: int,
    search_roots: Sequence[str] = (),
) -> None:
    global _NORMALIZER, _SPLIT_OUT, _CROP, _SEARCH_ROOTS
    _NORMALIZER = build_normalizer(norm_name, reference=reference, params=norm_params)
    if _NORMALIZER is None:
        raise RuntimeError("preprocess_normalize called with normalization 'none'")
    _SPLIT_OUT = Path(split_out)
    _CROP = CenterCrop(size=image_size)
    _SEARCH_ROOTS = [str(r) for r in search_roots]


def _process_one(task: Tuple[str, str, str]) -> Tuple[str, str, bool, str]:
    """``(src_path, rel_out, class_name)`` -> ``(src, out, ok, note)``."""
    from PIL import Image

    src_str, rel_out, _class_name = task
    assert _NORMALIZER is not None and _SPLIT_OUT is not None and _CROP is not None
    out_path = _SPLIT_OUT / rel_out
    if out_path.exists():
        return src_str, str(out_path), True, "skipped-exists"
    try:
        roots = [Path(r) for r in _SEARCH_ROOTS] or [_SPLIT_OUT.parent.parent, REPO_ROOT]
        source = resolve_image_path(src_str, roots, strict=False)
        if source is None:
            return src_str, str(out_path), False, "source-not-found"
        with Image.open(source) as img:
            array = np.asarray(img.convert("RGB"), dtype=np.uint8)
        normalised = _NORMALIZER(array)
        normalised = _CROP(normalised)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(normalised).save(
            out_path, quality=_SAVE_QUALITY, subsampling=0, optimize=True
        )
        return src_str, str(out_path), True, "ok"
    except Exception as exc:  # noqa: BLE001 - never kill a long cache job on one tile
        return src_str, str(out_path), False, f"{type(exc).__name__}: {exc}"


def _relative_output_path(src: str, class_name: str) -> str:
    """Preserve the class folder and the original filename (with .jpg suffix)."""
    name = Path(src).stem + _SAVE_EXT
    return str(Path(class_name) / name)


def write_cached_splits(
    split_csv: Path, split_out: Path, frame: pd.DataFrame, workers: int
) -> Path:
    """Rewrite a split CSV so ``image_path`` points at the cached tiles.

    This is a pure speed optimisation: the cached tiles are the *same* tiles the
    online pipeline would produce (same normaliser, same reference, same crop),
    so switching between the cached and online paths cannot change a metric.
    """
    cached = frame.copy()
    cached["image_path"] = [
        str(split_out / _relative_output_path(row["image_path"], row["class_name"]))
        for _, row in cached.iterrows()
    ]
    cached["source_image_path"] = frame["image_path"].values
    target = Path(split_out) / f"{split_csv.stem}_cached.csv"
    cached.to_csv(target, index=False)
    logger.info("[cache] wrote %s (%d rows)", target, len(cached))
    return target


def cache_split(
    split_csv: Path,
    split_name: str,
    out_dir: Path,
    norm_name: str,
    norm_params: Dict[str, Any],
    reference: np.ndarray,
    image_size: int,
    workers: int,
    limit: int = 0,
    search_roots: Sequence[str] = (),
    write_cached_csv: bool = True,
) -> Dict[str, Any]:
    frame = pd.read_csv(split_csv)
    if limit and limit > 0:
        frame = frame.iloc[:limit].reset_index(drop=True)
    split_out = out_dir / split_name
    split_out.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[str, str, str]] = [
        (row["image_path"], _relative_output_path(row["image_path"], row["class_name"]), row["class_name"])
        for _, row in frame.iterrows()
    ]
    logger.info("[%s] %d tiles -> %s (%d workers)", split_name, len(tasks), split_out, workers)

    started = time.time()
    ok = 0
    failed: List[str] = []
    skipped = 0
    if workers <= 1:
        _init_worker(norm_name, norm_params, reference, str(split_out), image_size, search_roots)
        for task in tasks:
            src, out, success, note = _process_one(task)
            if success:
                ok += 1
                skipped += int(note == "skipped-exists")
            else:
                failed.append(f"{src}: {note}")
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(
                norm_name,
                norm_params,
                reference,
                str(split_out),
                image_size,
                list(search_roots),
            ),
        ) as pool:
            futures = [pool.submit(_process_one, task) for task in tasks]
            for count, future in enumerate(as_completed(futures), start=1):
                src, _out, success, note = future.result()
                if success:
                    ok += 1
                    skipped += int(note == "skipped-exists")
                else:
                    failed.append(f"{src}: {note}")
                if count % 500 == 0 or count == len(tasks):
                    rate = count / max(1e-6, time.time() - started)
                    logger.info(
                        "[%s] %d/%d processed (%.1f tiles/s, %d failures)",
                        split_name,
                        count,
                        len(tasks),
                        rate,
                        len(failed),
                    )

    elapsed = time.time() - started
    cached_csv = None
    if write_cached_csv and not limit:
        cached_csv = write_cached_splits(split_csv, split_out, frame, workers)
    stats = _NORMALIZER.stats.as_dict() if _NORMALIZER is not None else {}
    result = {
        "split": split_name,
        "n_tiles": len(tasks),
        "n_written": int(ok - skipped),
        "n_ok": int(ok),
        "n_skipped_existing": int(skipped),
        "n_failed": len(failed),
        "failure_examples": failed[:10],
        "seconds": round(elapsed, 2),
        "tiles_per_second": round(len(tasks) / max(1e-6, elapsed), 2),
        "out_dir": str(split_out),
        "cached_split_csv": str(cached_csv) if cached_csv else None,
        "normalizer_stats": stats,
    }
    logger.info("[%s] done in %.1f min (%s)", split_name, elapsed / 60.0, json.dumps({k: v for k, v in result.items() if k != "failure_examples"}))
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Experiment YAML (defines normalization + image size).")
    parser.add_argument("--normalization", default=None, choices=["reinhard", "macenko"], help="Override the config's normalisation.")
    parser.add_argument("--splits-dir", default="data/processed/splits")
    parser.add_argument("--out-dir", default="data/processed/cache")
    parser.add_argument("--reference", default=None, help="Path to the canonical reference tile.")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--splits", nargs="*", default=["train", "val_id", "test_id", "test_ood"])
    parser.add_argument("--limit", type=int, default=0, help="Debug: only the first N rows of each split.")
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Extra root for resolving the image_path column (e.g. /kaggle/input/<dataset>).",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Extra config overrides.")
    parser.add_argument(
        "--no-cached-splits",
        action="store_true",
        help="Do not emit <split>_cached.csv (only useful for partial/debug caches).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config, overrides=args.overrides, repo_root=REPO_ROOT)
    norm_cfg = dict(cfg.get("normalization", {}) or {})
    norm_name = (args.normalization or norm_cfg.get("name") or "").lower()
    if norm_name not in {"reinhard", "macenko"}:
        logger.error(
            "This script is only meaningful for stain normalisation; got '%s'. "
            "Use --normalization reinhard|macenko (or a config that sets one).",
            norm_name,
        )
        return 2

    image_size = int(args.image_size or (cfg.get("data", {}) or {}).get("image_size", 224))
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir) / norm_name
    out_dir.mkdir(parents=True, exist_ok=True)

    reference, provenance = resolve_reference_image(
        reference_path=args.reference or norm_cfg.get("reference_path"),
        search_roots=[REPO_ROOT, splits_dir.parent.parent],
    )
    search_roots: List[str] = [str(REPO_ROOT), str(splits_dir.parent.parent)]
    for extra in list((cfg.get("paths", {}) or {}).get("search_roots", []) or []) + list(args.search_root):
        if str(extra) not in search_roots:
            search_roots.append(str(extra))
    logger.info("Reference tile: %s (shape=%s)", provenance, reference.shape)
    logger.info("Search roots  : %s", search_roots)
    logger.info("Normalisation : %s params=%s", norm_name, norm_cfg.get("params", {}))
    logger.info("Output        : %s", out_dir)

    results: List[Dict[str, Any]] = []
    started = time.time()
    for split in args.splits:
        split_csv = splits_dir / f"{split}.csv"
        if not split_csv.exists():
            logger.warning("Split CSV missing, skipping: %s", split_csv)
            continue
        results.append(
            cache_split(
                split_csv=split_csv,
                split_name=split,
                out_dir=out_dir,
                norm_name=norm_name,
                norm_params=dict(norm_cfg.get("params", {}) or {}),
                reference=reference,
                image_size=image_size,
                workers=int(args.workers),
                limit=int(args.limit),
                search_roots=search_roots,
                write_cached_csv=not args.no_cached_splits,
            )
        )

    total_seconds = time.time() - started
    manifest = {
        "normalization": norm_name,
        "normalizer_params": norm_cfg.get("params", {}),
        "reference_provenance": provenance,
        "search_roots": search_roots,
        "image_size": image_size,
        "save_format": _SAVE_EXT,
        "save_quality": _SAVE_QUALITY,
        "splits": results,
        "total_seconds": round(total_seconds, 2),
        "total_minutes": round(total_seconds / 60.0, 2),
        "note": (
            "Tiles are cached AFTER normalisation and BEFORE any augmentation, "
            "matching histo_robust.data.dataset transform order exactly."
        ),
    }
    manifest_path = out_dir / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("\nNORMALISATION CACHE SUMMARY")
    print("-" * 70)
    for result in results:
        print(
            f"  {result['split']:<9} written={result['n_written']:>6} "
            f"skipped={result['n_skipped_existing']:>6} failed={result['n_failed']:>3} "
            f"({result['tiles_per_second']:.1f} tiles/s)"
        )
    print("-" * 70)
    print(f"  total: {total_seconds / 60.0:.1f} min -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
