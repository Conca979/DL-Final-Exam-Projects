"""Post-training evaluation and robustness quantification.

This is the **only** place in the codebase that is allowed to touch
``CRC-VAL-HE-7K`` (``test_ood``).  It runs strictly after ``best.pt`` has been
frozen on ``val_id``, and it produces, per experiment:

* ``metrics_test_id.json`` / ``metrics_test_ood.json`` -- accuracy, balanced
  accuracy, Macro-F1, macro OvR AUROC, ECE (two definitions), per-class table;
* ``predictions_test_id.npz`` / ``predictions_test_ood.npz`` -- raw probabilities
  and labels, so the verification agent can recompute every reported number
  without re-running the model;
* ``confusion_test_{id,ood}.csv`` + ``.png`` -- row-normalised 9x9 matrices;
* ``robustness.json`` -- ``delta_f1 = F1_ID - F1_OOD`` and
  ``rr_f1 = F1_OOD / F1_ID * 100`` (plus the same pair for accuracy and
  balanced accuracy, per ``docs/PLAN.md`` section 4.3);
* sample tile sheets (raw vs normalised) for eyeballing the pipeline.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..utils.metrics import CLASS_NAMES, robustness_metrics
from ..utils.visualization import plot_confusion_matrix
from .trainer import Predictions, run_inference

logger = logging.getLogger(__name__)

__all__ = [
    "evaluate_experiment",
    "write_metrics_csv",
    "ROBUSTNESS_KEYS",
    "SUMMARY_COLUMNS",
]

#: Metrics for which the in-domain vs out-of-domain gap is quantified.
ROBUSTNESS_KEYS = ("macro_f1", "balanced_acc", "accuracy", "macro_auroc")

#: Column order of ``results/metrics/summary_results.csv`` (one row per EXP).
SUMMARY_COLUMNS: tuple[str, ...] = (
    "exp_id",
    "stage",
    "backbone",
    "weights_source",
    "normalization",
    "augmentation",
    "status",
    "stop_reason",
    "epochs_completed",
    "best_val_macro_f1",
    "best_epoch",
    "trained_minutes",
    "test_id_accuracy",
    "test_id_balanced_acc",
    "test_id_macro_f1",
    "test_id_macro_auroc",
    "test_id_ece",
    "test_ood_accuracy",
    "test_ood_balanced_acc",
    "test_ood_macro_f1",
    "test_ood_macro_auroc",
    "test_ood_ece",
    "delta_f1",
    "rr_f1",
    "delta_balanced_acc",
    "rr_balanced_acc",
    "delta_accuracy",
    "rr_accuracy",
    "delta_auroc",
    "rr_auroc",
    "config_hash",
    "checkpoint",
    "notes",
)


def _dump_confusion_csv(path: Path, matrix: np.ndarray, class_names: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred", *class_names])
        for idx, name in enumerate(class_names):
            writer.writerow([name, *[f"{v:.6f}" for v in matrix[idx]]])
    return path


def evaluate_experiment(
    cfg: Dict[str, Any],
    exp_id: str,
    model: Any,
    loaders: Dict[str, Any],
    output_root: str | Path,
    class_names: Sequence[str] = CLASS_NAMES,
    amp: bool = True,
    save_samples: int = 0,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the in-domain and out-of-domain audits and persist every artifact."""
    from ..utils.config import config_hash

    out_root = Path(output_root)
    exp_dir = out_root / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    device = next(iter(model.parameters())).device if hasattr(model, "parameters") else "cpu"

    per_split: Dict[str, Dict[str, Any]] = {}
    for split in ("test_id", "test_ood"):
        loader = loaders.get(split)
        if loader is None:
            logger.warning("%s | split '%s' has no loader; skipping", exp_id, split)
            continue
        logger.info("%s | evaluating %s (%d batches)", exp_id, split, len(loader))
        predictions: Predictions = run_inference(
            model, loader, device, amp=amp, desc=f"{exp_id} {split}"
        )
        metrics = predictions.metrics(class_names)
        per_split[split] = metrics

        (exp_dir / f"metrics_{split}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        predictions.save(exp_dir / f"predictions_{split}.npz")

        cm = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
        _dump_confusion_csv(exp_dir / f"confusion_{split}.csv", cm, class_names)
        try:
            plot_confusion_matrix(
                cm,
                exp_dir / f"confusion_{split}.png",
                class_names=class_names,
                title=f"{exp_id} -- {split} (row-normalised)",
            )
        except Exception as exc:  # noqa: BLE001 - plotting must never break the audit
            logger.warning("%s | confusion plot failed for %s: %s", exp_id, split, exc)

        logger.info(
            "%s | %s -> acc=%.4f bal_acc=%.4f macro_f1=%.4f auroc=%.4f ece=%.4f",
            exp_id,
            split,
            metrics["accuracy"],
            metrics["balanced_acc"],
            metrics["macro_f1"],
            metrics["macro_auroc"],
            metrics["ece"],
        )

    robustness: Dict[str, Any] = {}
    if "test_id" in per_split and "test_ood" in per_split:
        for key in ROBUSTNESS_KEYS:
            robustness.update(robustness_metrics(per_split["test_id"], per_split["test_ood"], key))
        robustness["interpretation"] = {
            "delta": "ID minus OOD; lower is better (0 == perfect stain invariance)",
            "rr": "OOD / ID * 100; higher is better (100 == no diagnostic power lost)",
        }
    else:
        logger.error(
            "%s | cannot compute robustness: need both test_id and test_ood metrics", exp_id
        )
    robustness["exp_id"] = exp_id
    robustness["config_hash"] = config_hash(cfg)
    if extra_meta:
        robustness.update(extra_meta)
    (exp_dir / "robustness.json").write_text(json.dumps(robustness, indent=2), encoding="utf-8")

    payload = {
        "exp_id": exp_id,
        "config_hash": config_hash(cfg),
        "test_id": per_split.get("test_id", {}),
        "test_ood": per_split.get("test_ood", {}),
        "robustness": robustness,
        "meta": extra_meta or {},
    }
    (exp_dir / "evaluation_result.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    _write_evaluation_result_txt(exp_dir / "evaluation_result.txt", payload, class_names)
    return payload


def _write_evaluation_result_txt(
    path: Path, payload: Dict[str, Any], class_names: Sequence[str]
) -> Path:
    """Flat, greppable text dump -- the file the human reads first."""
    lines: List[str] = []
    exp_id = payload.get("exp_id", "?")
    lines.append("=" * 78)
    lines.append(f"EVALUATION RESULT -- {exp_id}")
    lines.append(f"config_hash: {payload.get('config_hash')}")
    lines.append("=" * 78)

    for split in ("test_id", "test_ood"):
        metrics = payload.get(split) or {}
        if not metrics:
            lines.append(f"\n[{split}] not evaluated")
            continue
        lines.append(f"\n[{split}]  n={metrics.get('n_samples')}")
        for key in (
            "accuracy",
            "balanced_acc",
            "macro_f1",
            "weighted_f1",
            "macro_auroc",
            "ece",
            "ece_label",
            "mean_confidence",
        ):
            value = metrics.get(key)
            lines.append(
                f"  {key:<18}: {value:.6f}" if isinstance(value, (int, float)) else f"  {key:<18}: {value}"
            )
        lines.append("  per-class:")
        lines.append(
            f"    {'class':<6} {'prec':>8} {'recall':>8} {'f1':>8} {'auroc':>8} {'n':>6}"
        )
        for row in metrics.get("per_class", []):
            auroc = row.get("auroc")
            auroc_txt = f"{auroc:.4f}" if isinstance(auroc, (int, float)) else "n/a"
            lines.append(
                f"    {row['class_name']:<6} {row['precision']:>8.4f} {row['recall']:>8.4f} "
                f"{row['f1']:>8.4f} {auroc_txt:>8} {row['support']:>6}"
            )

    robustness = payload.get("robustness") or {}
    if robustness:
        lines.append("\n[ROBUSTNESS]  ID -> OOD")
        for key in ROBUSTNESS_KEYS:
            id_val = robustness.get(f"{key}_id")
            ood_val = robustness.get(f"{key}_ood")
            delta = robustness.get(f"delta_{key}")
            rr = robustness.get(f"rr_{key}")
            if id_val is None:
                continue
            lines.append(
                f"  {key:<14}: ID={id_val:.4f} OOD={ood_val:.4f} delta={delta:+.4f} rr={rr:.2f}%"
            )
        lines.append(
            "  delta = ID - OOD (lower is better); rr = OOD/ID*100 (higher is better)"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_metrics_csv(rows: List[Dict[str, Any]], path: str | Path) -> Path:
    """Write the summary table with a stable column order (missing -> '')."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SUMMARY_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in SUMMARY_COLUMNS})
    logger.info("Summary table written to %s (%d rows)", path, len(rows))
    return path


def build_summary_row(
    exp_id: str,
    exp_cfg: Dict[str, Any],
    train_result: Optional[Dict[str, Any]],
    evaluation: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Flatten one experiment's config + training + evaluation into a CSV row."""
    model_cfg = exp_cfg.get("model", {}) or {}
    norm_cfg = exp_cfg.get("normalization", {}) or {}
    aug_cfg = exp_cfg.get("augmentation", {}) or {}
    row: Dict[str, Any] = {
        "exp_id": exp_id,
        "stage": exp_cfg.get("stage", ""),
        "backbone": model_cfg.get("backbone", ""),
        "weights_source": model_cfg.get("weights_source", ""),
        "normalization": norm_cfg.get("name", "none"),
        "augmentation": aug_cfg.get("policy", "none"),
    }
    if train_result:
        row.update(
            {
                "status": train_result.get("status", ""),
                "stop_reason": train_result.get("stop_reason", ""),
                "epochs_completed": train_result.get("epochs_completed", ""),
                "best_val_macro_f1": _fmt(train_result.get("best_val_macro_f1")),
                "best_epoch": train_result.get("best_epoch", ""),
                "trained_minutes": _fmt(
                    (train_result.get("trained_seconds") or 0.0) / 60.0
                ),
                "checkpoint": train_result.get("best_path") or train_result.get("last_path") or "",
            }
        )
    if evaluation:
        for split in ("test_id", "test_ood"):
            metrics = evaluation.get(split) or {}
            for key in ("accuracy", "balanced_acc", "macro_f1", "macro_auroc", "ece"):
                row[f"{split}_{key}"] = _fmt(metrics.get(key)) if metrics else ""
        robustness = evaluation.get("robustness") or {}
        for key in ROBUSTNESS_KEYS:
            short = {"macro_f1": "f1", "balanced_acc": "balanced_acc", "accuracy": "accuracy", "macro_auroc": "auroc"}[key]
            row[f"delta_{short}"] = _fmt(robustness.get(f"delta_{key}"))
            row[f"rr_{short}"] = _fmt(robustness.get(f"rr_{key}"))
        row["config_hash"] = evaluation.get("config_hash", "")
    return row


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return round(value, 6)
    return value
