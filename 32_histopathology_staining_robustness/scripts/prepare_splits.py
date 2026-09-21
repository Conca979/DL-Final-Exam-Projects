#!/usr/bin/env python
"""Build the train / val_id / test_id / test_ood split CSVs.

Implements ``dataset_card.md`` section 4 exactly:

* source domain ``NCT-CRC-HE-100K-NONORM`` -> stratified 70 / 15 / 15 split with
  ``seed=42`` (train / val_id / test_id);
* target domain ``CRC-VAL-HE-7K`` -> 100 % of the patches become ``test_ood``;
* optional standardised 25 000-patch compute-budget subset (17 500 / 3 750 /
  3 750) selected by stratified sampling **before** splitting, so the three parts
  stay class-balanced.

Every output row carries a ``domain`` column (``source`` / ``target``) and the
script asserts the firewall: no target-domain row may appear in train or val_id,
and no image path may appear in two splits.

Usage (Kaggle cell or local shell)::

    python scripts/prepare_splits.py \
        --data-root /kaggle/input/nct-crc-he-100k-nonorm \
        --out-dir   /kaggle/working/data/processed/splits \
        --subset 25000 --seed 42 \
        --reference-out /kaggle/working/data/processed/templates/reference_stain.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from histo_robust.data.paths import IMAGE_SUFFIXES, find_class_dirs, iter_image_files  # noqa: E402
from histo_robust.normalization import load_image_rgb  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
logger = logging.getLogger("prepare_splits")

CLASS_NAMES: Tuple[str, ...] = (
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
SOURCE_NAME = "NCT-CRC-HE-100K-NONORM"
TARGET_NAME = "CRC-VAL-HE-7K"
DEFAULT_SUBSET = 25_000
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


# ---------------------------------------------------------------------------
def build_image_table(root: Path, domain: str, class_names: Sequence[str] = CLASS_NAMES) -> pd.DataFrame:
    """One row per image with canonical class index and domain tag."""
    found = find_class_dirs(root, class_names)
    missing = [c for c in class_names if c not in found]
    if missing:
        raise FileNotFoundError(
            f"Class folders {missing} not found under {root}. Expected nine "
            f"sub-folders named {list(class_names)} (found: {sorted(found)}). "
            f"Check --data-root / the mounted Kaggle dataset."
        )
    rows: List[Dict[str, Any]] = []
    for label_idx, class_name in enumerate(class_names):
        files = iter_image_files(found[class_name])
        logger.info("  %-5s %6d images", class_name, len(files))
        for path in files:
            rows.append(
                {
                    "image_path": str(path),
                    "class_name": class_name,
                    "label_idx": label_idx,
                    "domain": domain,
                }
            )
    return pd.DataFrame(rows)


def _rebalance_subset(frame: pd.DataFrame, subset_size: int, seed: int) -> pd.DataFrame:
    """Stratified subset capped per class, then shuffled (order-independent)."""
    from sklearn.model_selection import train_test_split

    fractions = np.array(SPLIT_FRACTIONS, dtype=np.float64)
    per_class_target = np.floor(fractions * (subset_size / len(CLASS_NAMES))).astype(int)
    per_class_target = np.maximum(per_class_target, 1)

    pieces: List[pd.DataFrame] = []
    for idx, class_name in enumerate(CLASS_NAMES):
        class_rows = frame[frame["class_name"] == class_name]
        want = min(int(per_class_target[idx]), len(class_rows))
        if want == 0:
            logger.warning("Class %s has no rows; skipping in subset", class_name)
            continue
        pieces.append(class_rows.sample(n=want, random_state=seed))
    subset = pd.concat(pieces, ignore_index=True)
    return subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def stratified_split(
    frame: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 stratified split on ``class_name``."""
    from sklearn.model_selection import train_test_split

    train_frac, val_frac, test_frac = SPLIT_FRACTIONS
    train, rest = train_test_split(
        frame,
        test_size=(val_frac + test_frac),
        stratify=frame["class_name"],
        random_state=seed,
    )
    rest_val_frac = val_frac / (val_frac + test_frac)
    val, test = train_test_split(
        rest,
        test_size=(1.0 - rest_val_frac),
        stratify=rest["class_name"],
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def choose_reference_image(train: pd.DataFrame) -> Optional[Path]:
    """Pick a deterministic source-domain TUM tile as the canonical stain reference.

    Heuristic (bounded to the first 200 TUM training tiles, so it costs seconds):
    a *dark* tile (dense nuclei -> strong haematoxylin) with a *low* fraction of
    slide-glass pixels, which correlates with a balanced H&E appearance.  Using
    the training split only keeps the target domain out of the normalisation
    reference, as required by the firewall.
    """
    candidates = train[train["class_name"] == "TUM"]["image_path"].tolist()
    if not candidates:
        candidates = train["image_path"].tolist()
    if not candidates:
        return None

    best: Tuple[float, Optional[Path]] = (float("inf"), None)
    for path_str in candidates[:200]:
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            image = load_image_rgb(path).astype(np.float64)
        except Exception:  # noqa: BLE001
            continue
        background = float(np.mean(np.all(image > 220, axis=2)))
        darkness = float(image.mean())
        score = darkness + 200.0 * background
        if score < best[0]:
            best = (score, path)
    return best[1]


# ---------------------------------------------------------------------------
def assert_no_leakage(splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Verify disjoint paths and an intact domain firewall."""
    report: Dict[str, Any] = {}
    keys = list(splits)
    for i, first in enumerate(keys):
        for second in keys[i + 1 :]:
            overlap = set(splits[first]["image_path"]) & set(splits[second]["image_path"])
            if overlap:
                raise AssertionError(
                    f"DATA LEAKAGE: {len(overlap)} image path(s) appear in both "
                    f"'{first}' and '{second}'. First example: {sorted(overlap)[0]}"
                )
    for name in ("train", "val_id"):
        domains = set(splits[name]["domain"].unique())
        if domains - {"source"}:
            raise AssertionError(
                f"DOMAIN FIREWALL VIOLATION: '{name}' contains domains {domains}; "
                f"only the source domain may be trained on or used for checkpoint "
                f"selection."
            )
    ood_domains = set(splits["test_ood"]["domain"].unique())
    if ood_domains != {"target"}:
        raise AssertionError(
            f"test_ood must contain the target domain only, found {ood_domains}"
        )
    for name, frame in splits.items():
        report[name] = {
            "n_samples": int(len(frame)),
            "domains": sorted(frame["domain"].unique().tolist()),
            "class_counts": {
                str(k): int(v) for k, v in frame["class_name"].value_counts().items()
            },
        }
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_splits(splits: Dict[str, pd.DataFrame], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes: Dict[str, str] = {}
    for name, frame in splits.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        hashes[name] = _sha256(path)
        logger.info("wrote %s (%d rows)", path, len(frame))
    return hashes


# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="Directory holding the class folders (source domain).")
    parser.add_argument("--target-root", default=None, help="Directory holding CRC-VAL-HE-7K class folders.")
    parser.add_argument("--out-dir", default="data/processed/splits", help="Where the CSVs are written.")
    parser.add_argument("--subset", type=int, default=DEFAULT_SUBSET, help="Stratified source subset size (use 0 + --full for all 100k).")
    parser.add_argument("--full", action="store_true", help="Use every source patch (ignores --subset).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-out", default=None, help="Copy the chosen canonical reference tile here.")
    parser.add_argument("--limit-target", type=int, default=0, help="Optional cap on test_ood rows (0 = all).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    source_root = Path(args.data_root)
    target_root = Path(args.target_root) if args.target_root else source_root.parent / TARGET_NAME
    out_dir = Path(args.out_dir)

    logger.info("Source domain : %s", source_root)
    logger.info("Target domain : %s", target_root)
    if not source_root.exists():
        logger.error("Source root does not exist: %s", source_root)
        return 2
    if not target_root.exists():
        logger.error(
            "Target root does not exist: %s (CRC-VAL-HE-7K is mandatory for the "
            "OOD half of the evaluation protocol)",
            target_root,
        )
        return 2

    logger.info("Scanning source domain ...")
    source = build_image_table(source_root, domain="source")
    logger.info("Scanning target domain ...")
    target = build_image_table(target_root, domain="target")
    if args.limit_target and args.limit_target > 0:
        target = target.sample(n=min(args.limit_target, len(target)), random_state=args.seed)
        target = target.reset_index(drop=True)
        logger.warning("test_ood capped to %d rows by --limit-target", len(target))

    if args.full or args.subset <= 0:
        logger.info("Using the FULL source domain (%d patches)", len(source))
        source_pool = source
        subset_label = "full"
    else:
        source_pool = _rebalance_subset(source, args.subset, args.seed)
        subset_label = f"subset{args.subset}"
        logger.info("Using the standardised stratified subset (%d patches)", len(source_pool))

    train, val_id, test_id = stratified_split(source_pool, args.seed)
    splits = {
        "train": train,
        "val_id": val_id,
        "test_id": test_id,
        "test_ood": target,
    }
    report = assert_no_leakage(splits)
    hashes = write_splits(splits, out_dir)

    reference_source = None
    if args.reference_out:
        chosen = choose_reference_image(train)
        if chosen is not None:
            import shutil

            destination = Path(args.reference_out)
            destination.parent.mkdir(parents=True, exist_ok=True)
            image = load_image_rgb(chosen)
            from PIL import Image

            Image.fromarray(image).save(destination)
            reference_source = str(chosen)
            logger.info("Canonical stain reference: %s -> %s", chosen, destination)

    summary = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "subset": subset_label,
        "seed": int(args.seed),
        "split_fractions": list(SPLIT_FRACTIONS),
        "n_source_available": int(len(source)),
        "n_source_used": int(len(source_pool)),
        "splits": report,
        "csv_sha256": hashes,
        "reference_source_image": reference_source,
        "note": (
            "test_ood is the RWTH Aachen cohort (CRC-VAL-HE-7K). It is never used "
            "for training, normalisation-reference fitting, scheduling or "
            "checkpoint selection."
        ),
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSPLIT SUMMARY")
    print("-" * 62)
    for name in ("train", "val_id", "test_id", "test_ood"):
        frame = splits[name]
        domains = ",".join(sorted(frame["domain"].unique().tolist()))
        print(f"  {name:<9} {len(frame):>7} rows   domain={domains}")
    print("-" * 62)
    print(f"  leakage checks   : PASSED (disjoint paths, firewall intact)")
    print(f"  reference tile   : {reference_source or 'not requested'}")
    print(f"  written to       : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
