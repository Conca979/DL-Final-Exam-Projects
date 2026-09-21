#!/usr/bin/env python
"""Post-training audit: in-domain (``test_id``) and out-of-domain (``test_ood``).

Run this **after** training has finished.  It loads the frozen ``best.pt``
(falling back to ``last.pt``) and evaluates it on both held-out splits, then
computes the robustness gap::

    delta_f1 = F1_ID - F1_OOD        # lower is better
    rr_f1    = F1_OOD / F1_ID * 100  # higher is better

Outputs per experiment (under ``results/per_experiment/<EXP>/``):
``evaluation_result.txt``, ``evaluation_result.json``,
``metrics_test_id.json``, ``metrics_test_ood.json``, ``robustness.json``,
``predictions_*.npz``, ``confusion_*.csv/.png``; plus one aggregated
``results/metrics/summary_results.csv``.

Example::

    python scripts/evaluate.py --config configs/experiments/exp03_norm_macenko_resnet50.yaml \
        --exp-id EXP-03 --checkpoint-dir /kaggle/working/checkpoints \
        --results-dir /kaggle/working/results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from histo_robust.data import build_eval_loaders, prefer_cached_splits, resolve_reference  # noqa: E402
from histo_robust.engine import (  # noqa: E402
    Trainer,
    build_summary_row,
    evaluate_experiment,
    write_metrics_csv,
)
from histo_robust.utils.config import config_hash, load_config  # noqa: E402
from histo_robust.utils.kaggle import configure_logging, log_environment  # noqa: E402
from histo_robust.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger("evaluate")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Root directory containing <exp-id>/best.pt")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-bootstrap", type=int, default=0, help="Reserved: bootstrap CI replicates (0 = off).")
    parser.add_argument("--save-samples", type=int, default=0, help="Reserved: sample tiles per split (0 = off).")
    parser.add_argument("--limit-test-id", type=int, default=0)
    parser.add_argument("--limit-test-ood", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument(
        "--normalization-cache",
        default=None,
        help="Pre-normalised tile directory (must match the one used at training time).",
    )
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.set, repo_root=REPO_ROOT)
    exp_id = args.exp_id or str(cfg.get("exp_id") or cfg.get("_config_name") or "EXP")

    results_dir = Path(args.results_dir)
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir / f"evaluate_{exp_id}.log")

    if args.no_amp:
        cfg.setdefault("train", {})["amp"] = False
    if args.weights_dir:
        cfg.setdefault("paths", {})["weights_dir"] = args.weights_dir
    if args.reference:
        cfg.setdefault("normalization", {})["reference_path"] = args.reference
    if args.normalization_cache is not None:
        cfg.setdefault("paths", {})["normalization_cache"] = args.normalization_cache or None
    if args.search_root:
        existing = list((cfg.get("paths", {}) or {}).get("search_roots", []) or [])
        cfg.setdefault("paths", {})["search_roots"] = existing + list(args.search_root)

    reference_path = (cfg.get("normalization", {}) or {}).get("reference_path")
    if reference_path and not Path(str(reference_path)).is_absolute():
        candidate = REPO_ROOT / str(reference_path)
        if candidate.exists():
            cfg.setdefault("normalization", {})["reference_path"] = str(candidate)

    if prefer_cached_splits(cfg):
        logger.warning(
            "%s | evaluating from the pre-normalised cache. The checkpoint must have "
            "been trained with the same cache (or the online equivalent); verify "
            "config_hash in the checkpoint against this run's config hash.",
            exp_id,
        )

    set_seed(int(cfg.get("runtime", {}).get("seed", 42)))
    log_environment(logger)
    logger.info("%s | evaluation config hash=%s", exp_id, config_hash(cfg))

    reference, provenance = resolve_reference(cfg, repo_root=REPO_ROOT)
    logger.info("%s | stain reference: %s", exp_id, provenance)

    trainer = Trainer(
        cfg=cfg,
        exp_id=exp_id,
        output_root=args.checkpoint_dir,
        repo_root=REPO_ROOT,
        reference=reference,
        reference_provenance=provenance,
        resume=False,
    )
    trainer.setup_model()
    if not trainer.load_best_weights():
        logger.error(
            "%s | no checkpoint under %s/%s -- train first (or point --checkpoint-dir "
            "at the mounted checkpoint dataset).",
            exp_id, args.checkpoint_dir, exp_id,
        )
        return 2

    loaders = build_eval_loaders(
        cfg,
        splits=["test_id", "test_ood"],
        reference=reference,
        repo_root=REPO_ROOT,
        limit_test_id=args.limit_test_id or None,
        limit_test_ood=args.limit_test_ood or None,
        post_training=True,
    )
    evaluation = evaluate_experiment(
        cfg=cfg,
        exp_id=exp_id,
        model=trainer.model,
        loaders=loaders,
        output_root=results_dir / "per_experiment",
        class_names=list(cfg.get("data", {}).get("class_names", [])),
        amp=not args.no_amp,
    )

    run_report_path = Path(args.checkpoint_dir) / exp_id / "train_report.json"
    train_result: Dict[str, Any] = {}
    if run_report_path.exists():
        try:
            train_result = json.loads(run_report_path.read_text(encoding="utf-8")).get(
                "train_result", {}
            )
        except json.JSONDecodeError:
            logger.warning("%s | could not parse %s", exp_id, run_report_path)

    row = build_summary_row(exp_id, cfg, train_result, evaluation)
    row["notes"] = f"evaluated {trainer.manager.best_path.name if trainer.manager.best_path.exists() else 'last.pt'}"

    summary_csv = results_dir / "metrics" / "summary_results.csv"
    existing: List[Dict[str, Any]] = []
    if summary_csv.exists():
        import pandas as pd

        existing = pd.read_csv(summary_csv).to_dict("records")
        existing = [r for r in existing if str(r.get("exp_id")) != exp_id]
    write_metrics_csv(existing + [row], summary_csv)

    robustness = evaluation.get("robustness", {})
    print("\n" + "=" * 78)
    print(f"EVALUATION SUMMARY -- {exp_id}")
    print("=" * 78)
    print(f"  in-domain  (test_id) : macro_f1={evaluation.get('test_id', {}).get('macro_f1')}")
    print(f"  out-domain (test_ood): macro_f1={evaluation.get('test_ood', {}).get('macro_f1')}")
    print(f"  delta_f1 (ID - OOD)  : {robustness.get('delta_macro_f1')}")
    print(f"  rr_f1    (OOD/ID %)  : {robustness.get('rr_macro_f1')}")
    print(f"  results              : {results_dir / 'per_experiment' / exp_id}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
