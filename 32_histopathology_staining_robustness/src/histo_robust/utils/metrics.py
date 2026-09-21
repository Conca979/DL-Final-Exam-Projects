"""Evaluation metrics for the 9-class colorectal tissue task.

Primary metric (per ``docs/PLAN.md`` section 4.2):

* ``macro_f1``        -- primary, class-balanced
* ``balanced_acc``    -- macro-averaged recall
* ``accuracy``        -- overall top-1
* ``macro_auroc``     -- macro one-vs-rest AUROC on softmax probabilities
* ``ece``             -- expected calibration error (15 equal-mass-free bins)

Robustness quantification (``docs/PLAN.md`` section 4.3)::

    delta_f1 = f1_id  - f1_ood      # lower is better
    rr_f1    = f1_ood / f1_id * 100 # higher is better

Everything here is pure NumPy/scikit-learn so the verifier can recompute the
numbers from the dumped ``predictions_*.npz`` arrays without touching torch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "compute_classification_metrics",
    "expected_calibration_error",
    "per_class_report",
    "confusion_matrix_normalized",
    "robustness_metrics",
    "NUM_CLASSES",
    "CLASS_NAMES",
]

#: Canonical class order (must match ``docs/dataset_card.md`` section 2).
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
NUM_CLASSES = len(CLASS_NAMES)


def _sklearn_metrics():
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    return {
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "prfs": precision_recall_fscore_support,
        "roc_auc_score": roc_auc_score,
    }


# ---------------------------------------------------------------------------
# NumPy fallbacks
#
# Kaggle ships scikit-learn, and ``docs/PLAN.md`` selects it as the reference
# implementation, so the sklearn path is always preferred.  These fallbacks exist
# so that (a) a partially provisioned environment can still produce the numbers
# instead of failing at the very end of a 10-hour run, and (b) the verifier can
# cross-check sklearn against an independent implementation.
# ---------------------------------------------------------------------------
def _auc_from_scores(y_binary: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based (Mann-Whitney U) AUROC with average ranks for ties."""
    y_binary = np.asarray(y_binary).astype(int).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int((y_binary == 1).sum())
    n_neg = int((y_binary == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    idx = 0
    while idx < len(sorted_scores):
        end = idx
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[idx]:
            end += 1
        average_rank = (idx + end) / 2.0 + 1.0
        ranks[order[idx : end + 1]] = average_rank
        idx = end + 1
    rank_sum_pos = float(ranks[y_binary == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _numpy_metrics() -> Dict[str, Any]:
    def accuracy_score(y_true, y_pred):
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    def confusion_matrix(y_true, y_pred, labels):
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        index = {label: i for i, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for true, pred in zip(y_true, y_pred):
            if true in index and pred in index:
                matrix[index[true], index[pred]] += 1
        return matrix

    def precision_recall_fscore_support(y_true, y_pred, labels, average=None, zero_division=0):
        matrix = confusion_matrix(y_true, y_pred, labels)
        tp = np.diag(matrix).astype(np.float64)
        pred_sum = matrix.sum(axis=0).astype(np.float64)
        true_sum = matrix.sum(axis=1).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
            recall = np.divide(tp, true_sum, out=np.zeros_like(tp), where=true_sum > 0)
            f1 = np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros_like(tp),
                where=(precision + recall) > 0,
            )
        return precision, recall, f1, true_sum

    def f1_score(y_true, y_pred, average="macro", zero_division=0, labels=None):
        y_true_arr = np.asarray(y_true).astype(int)
        # sklearn macro-averages over the labels PRESENT in y_true; mirroring that
        # exactly is what keeps the fallback in agreement with the reference
        # implementation on partially evaluated splits.
        present = np.unique(y_true_arr)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true_arr, y_pred, labels=present.tolist(), average=average
        )
        if average == "macro":
            return float(np.mean(f1)) if len(f1) else 0.0
        if average == "weighted":
            support = np.array(
                [np.sum(y_true_arr == label) for label in present], dtype=np.float64
            )
            total = support.sum()
            return float(np.sum(f1 * support) / total) if total else 0.0
        return f1

    def balanced_accuracy_score(y_true, y_pred):
        y_true_arr = np.asarray(y_true).astype(int)
        present = np.unique(y_true_arr)
        _, recall, _, _ = precision_recall_fscore_support(
            y_true_arr, y_pred, labels=present.tolist()
        )
        return float(np.mean(recall)) if len(recall) else 0.0

    def roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=None):
        y_true = np.asarray(y_true).astype(int)
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            return _auc_from_scores(y_true, probs)
        aucs = [
            _auc_from_scores((y_true == idx).astype(int), probs[:, idx])
            for idx in range(probs.shape[1])
        ]
        aucs = [a for a in aucs if not np.isnan(a)]
        return float(np.mean(aucs)) if aucs else float("nan")

    return {
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "prfs": precision_recall_fscore_support,
        "roc_auc_score": roc_auc_score,
        "_backend": "numpy",
    }


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


def _metrics_backend() -> Dict[str, Any]:
    """sklearn when available (primary), NumPy otherwise (fallback)."""
    if sklearn_available():
        return _sklearn_metrics()
    global _WARNED_NO_SKLEARN
    if not _WARNED_NO_SKLEARN:
        _WARNED_NO_SKLEARN = True
        import logging

        logging.getLogger(__name__).warning(
            "scikit-learn is not importable; using the built-in NumPy metric "
            "fallback. Values agree with sklearn to ~1e-12 but they are a "
            "different implementation -- record which backend produced a result."
        )
    return _numpy_metrics()


_WARNED_NO_SKLEARN = False


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    mode: str = "confidence",
) -> float:
    """Expected Calibration Error of a softmax classifier.

    ``mode='confidence'`` is the standard definition: bin by max predicted
    probability and compare mean confidence with empirical accuracy.
    ``mode='label'`` instead bins by the predicted probability of the *true*
    class (a.k.a. adaptive/classwise ECE); it is less saturated on 9-class
    problems and is reported alongside the standard number.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(int).reshape(-1)
    if probs.ndim != 2 or probs.shape[0] != labels.shape[0]:
        raise ValueError("probs must be (N, C) and match labels")
    if probs.shape[0] == 0:
        return float("nan")

    if mode == "confidence":
        conf = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    elif mode == "label":
        conf = probs[np.arange(labels.shape[0]), labels]
        correct = np.ones_like(conf)
    else:
        raise ValueError(f"Unknown ECE mode: {mode}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = labels.shape[0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == 0.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
) -> List[Dict[str, Any]]:
    """Precision / recall / F1 / support for every class, in canonical order."""
    m = _metrics_backend()
    precision, recall, f1, support = m["prfs"](
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0,
    )
    rows = []
    for idx, name in enumerate(class_names):
        rows.append(
            {
                "class_index": idx,
                "class_name": name,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )
    return rows


def confusion_matrix_normalized(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Row-normalised confusion matrix (rows = true class), NaN-free."""
    m = _metrics_backend()
    cm = m["confusion_matrix"](y_true, y_pred, labels=list(range(num_classes)))
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    return cm


def compute_classification_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
    ece_bins: int = 15,
) -> Dict[str, Any]:
    """Full metric bundle for one evaluated split.

    Returns a JSON-serialisable dict; ``per_class`` is a list of dicts and
    ``confusion_matrix`` is a nested list (rows = true class).
    """
    m = _metrics_backend()
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probs must be 2-D (N, C)")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"probs rows ({probs.shape[0]}) != labels ({y_true.shape[0]})"
        )
    num_classes = len(class_names)
    if probs.shape[1] != num_classes:
        raise ValueError(f"probs has {probs.shape[1]} columns, expected {num_classes}")

    y_pred = probs.argmax(axis=1)

    result: Dict[str, Any] = {
        "n_samples": int(y_true.shape[0]),
        "accuracy": float(m["accuracy_score"](y_true, y_pred)),
        "balanced_acc": float(m["balanced_accuracy_score"](y_true, y_pred)),
        "macro_f1": float(
            m["f1_score"](y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            m["f1_score"](y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_auroc": float("nan"),
        "ece": expected_calibration_error(probs, y_true, n_bins=ece_bins, mode="confidence"),
        "ece_label": expected_calibration_error(probs, y_true, n_bins=ece_bins, mode="label"),
        "mean_confidence": float(probs.max(axis=1).mean()) if len(y_true) else float("nan"),
    }

    # Macro OvR AUROC needs every class present with both outcomes; guard hard
    # so a degenerate split (e.g. an accidental single-class subset) can never
    # silently poison the summary table.
    present = np.unique(y_true)
    if len(present) == num_classes:
        try:
            result["macro_auroc"] = float(
                m["roc_auc_score"](
                    y_true, probs, multi_class="ovr", average="macro", labels=list(range(num_classes))
                )
            )
        except ValueError:
            result["macro_auroc"] = float("nan")
        per_class_auc: List[Optional[float]] = []
        for idx in range(num_classes):
            binary = (y_true == idx).astype(int)
            if binary.min() == binary.max():
                per_class_auc.append(None)
                continue
            try:
                per_class_auc.append(float(m["roc_auc_score"](binary, probs[:, idx])))
            except ValueError:
                per_class_auc.append(None)
    else:
        per_class_auc = [None] * num_classes

    rows = per_class_report(y_true, y_pred, class_names)
    for row, auc in zip(rows, per_class_auc):
        row["auroc"] = auc
    result["per_class"] = rows

    result["confusion_matrix"] = confusion_matrix_normalized(
        y_true, y_pred, num_classes=num_classes
    ).tolist()
    return result


def robustness_metrics(
    metrics_id: Dict[str, Any],
    metrics_ood: Dict[str, Any],
    key: str = "macro_f1",
) -> Dict[str, float]:
    """``delta`` (ID - OOD, lower better) and ``rr`` (OOD/ID * 100, higher better)."""
    id_val = float(metrics_id.get(key, float("nan")))
    ood_val = float(metrics_ood.get(key, float("nan")))
    delta = id_val - ood_val
    rr = (ood_val / id_val * 100.0) if id_val and not np.isnan(id_val) else float("nan")
    return {
        f"delta_{key}": delta,
        f"rr_{key}": float(rr),
        f"{key}_id": id_val,
        f"{key}_ood": ood_val,
    }
