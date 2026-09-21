#!/usr/bin/env python
"""Train one ablation cell (``EXP-01`` ... ``EXP-13``) with a hard time box.

This is the script the Kaggle notebook calls.  It is deliberately explicit about
the Kaggle contract:

* it trains **only** on ``train`` and selects checkpoints **only** on ``val_id``
  (the domain firewall lives in :mod:`histo_robust.data.datamodule`);
* it stops itself at ``--max-minutes`` (default 630 = 10.5 h) and, if
  ``--session-start``/``KAGGLE_SESSION_START`` is available, also ahead of the
  12-hour hard kill, always writing ``last.pt`` + ``best.pt`` before returning;
* it auto-resumes from ``<output-dir>/<exp-id>/last.pt`` when present, so a
  session that was killed at hour 11 continues instead of restarting;
* ``--eval-test-id`` performs the post-training in-domain audit only.  The OOD
  audit is deliberately *not* available here -- it lives in
  ``scripts/evaluate.py``, which is run after training has finished.

Examples::

    # normal Kaggle training run for one cell
    python scripts/train.py --config configs/experiments/exp01_baseline_resnet50.yaml \
        --exp-id EXP-01 --output-dir /kaggle/working/checkpoints --max-minutes 630

    # 6-minute pre-flight check on 40 batches
    python scripts/train.py --config configs/base_config.yaml --exp-id PREFLIGHT \
        --smoke --max-minutes 6
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from histo_robust.data import (  # noqa: E402
    build_eval_loaders,
    build_train_val_loaders,
    describe_splits,
    prefer_cached_splits,
    resolve_reference,
)
from histo_robust.engine import Trainer, build_summary_row, evaluate_experiment, write_metrics_csv  # noqa: E402
from histo_robust.utils.checkpoint import free_space_gb, total_size_gb  # noqa: E402
from histo_robust.utils.config import config_hash, config_to_json, load_config  # noqa: E402
from histo_robust.utils.kaggle import configure_logging, log_environment  # noqa: E402
from histo_robust.utils.seed import set_seed, seed_everything_strict_note  # noqa: E402
from histo_robust.utils.timebudget import SessionClock, TimeBudget  # noqa: E402

logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument("--exp-id", default=None, help="Checkpoint/report folder name (defaults to the config stem).")
    parser.add_argument("--output-dir", default="checkpoints", help="Root for checkpoints/reports.")
    parser.add_argument("--results-dir", default="results", help="Root for metrics output.")
    parser.add_argument("--log-dir", default=None, help="Where to place train_<exp>.log (default: <results>/logs).")

    # --- Kaggle short-session controls -----------------------------------
    parser.add_argument("--max-minutes", type=float, default=None, help="Training budget for this run (default from config: 630).")
    parser.add_argument("--session-start", default=None, help="Session start time (e.g. '2025-01-31 12:00:00', UTC).")
    parser.add_argument("--reserve-minutes", type=float, default=None, help="Minutes held back before the 12-hour kill (default 20).")

    # --- checkpointing ---------------------------------------------------
    parser.add_argument("--save-every", type=int, default=None, help="Write a rolling checkpoint every N optimizer steps.")
    parser.add_argument("--keep-last", type=int, default=None, help="How many rolling step_*.pt checkpoints to retain.")
    parser.add_argument("--keep-best", type=int, default=None, help="How many best_v*.pt snapshots to retain.")
    parser.add_argument("--run-quota-gb", type=float, default=None, help="Per-experiment checkpoint quota enforced by pruning.")

    # --- run shape -------------------------------------------------------
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--limit-train", type=int, default=0, help="Debug: cap the number of training rows.")
    parser.add_argument("--limit-val", type=int, default=0, help="Debug: cap the number of val_id rows.")
    parser.add_argument("--limit-test-id", type=int, default=0, help="Debug: cap the number of test_id rows.")
    parser.add_argument("--smoke", action="store_true", help="Tiny run (limits rows, 1 epoch, small budget) for the pre-flight cell.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints and start from scratch.")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP (debugging only).")
    parser.add_argument("--eval-test-id", action="store_true", help="After training, run the in-domain (test_id) audit.")
    parser.add_argument("--eval-test-ood", action="store_true", help="Also run the OOD audit (post-training only; see scripts/evaluate.py).")
    parser.add_argument("--save-samples", type=int, default=0, help="Save N sample tiles per split for visual inspection.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Config override, e.g. --set model.backbone=phikon")
    parser.add_argument("--weights-dir", default=None, help="Directory with offline pretrained weights (Kaggle model dataset).")
    parser.add_argument("--reference", default=None, help="Canonical stain reference tile (overrides config).")
    parser.add_argument(
        "--normalization-cache",
        default=None,
        help="Directory of pre-normalised tiles (output of scripts/preprocess_normalize.py).",
    )
    parser.add_argument("--search-root", action="append", default=[], help="Extra root for resolving split image paths.")
    return parser.parse_args(argv)


def apply_runtime_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Push CLI flags into the config so the checkpoint records the true settings."""
    from histo_robust.utils.config import set_by_path

    def _set(path: str, value: Any) -> None:
        if value is not None:
            set_by_path(cfg, path, value)

    _set("train.max_train_minutes", args.max_minutes)
    _set("train.epochs", args.epochs)
    _set("train.batch_size", args.batch_size)
    _set("train.lr", args.lr)
    _set("checkpoints.save_every_steps", args.save_every)
    _set("checkpoints.keep_last", args.keep_last)
    _set("checkpoints.keep_best", args.keep_best)
    _set("checkpoints.run_quota_gb", args.run_quota_gb)
    _set("session.session_start", args.session_start)
    _set("session.reserve_minutes", args.reserve_minutes)

    if args.batch_size is not None:
        set_by_path(cfg, "data.batch_size", args.batch_size)
    if args.weights_dir:
        set_by_path(cfg, "paths.weights_dir", args.weights_dir)
    if args.reference:
        set_by_path(cfg, "normalization.reference_path", args.reference)
    if args.normalization_cache is not None:
        set_by_path(
            cfg,
            "paths.normalization_cache",
            args.normalization_cache or None,
        )
    if args.no_amp:
        set_by_path(cfg, "train.amp", False)
    if args.search_root:
        existing = list((cfg.get("paths", {}) or {}).get("search_roots", []) or [])
        set_by_path(cfg, "paths.search_roots", existing + list(args.search_root))
    return cfg


def resolve_reference_path(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Make a repo-relative ``normalization.reference_path`` absolute.

    Kaggle runs the scripts from ``/kaggle/working/histo-robust`` but may be
    invoked from another cwd, and ``reference_stain.png`` ships inside the
    codebase zip -- so resolve it against the repo root before anything else
    looks for it.
    """
    from histo_robust.utils.config import get_by_path

    raw = get_by_path(cfg, "normalization.reference_path")
    if not raw:
        return cfg
    path = Path(str(raw))
    if path.is_absolute():
        return cfg
    candidate = REPO_ROOT / path
    if candidate.exists():
        cfg.setdefault("normalization", {})["reference_path"] = str(candidate)
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    start = time.time()

    cfg = load_config(args.config, overrides=args.set, repo_root=REPO_ROOT)
    cfg = apply_runtime_overrides(cfg, args)
    cfg = resolve_reference_path(cfg)
    # If a complete pre-normalisation cache is present, adopt it BEFORE the
    # reference is resolved: cached tiles are already normalised, so the run's
    # effective normaliser becomes identity and needs no reference at all.
    cached_dir = prefer_cached_splits(cfg)
    if cached_dir:
        logger.info("pre-normalised cache adopted: %s", cached_dir)

    exp_id = args.exp_id or str(cfg.get("exp_id") or cfg.get("_config_name") or "EXP")
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    log_dir = Path(args.log_dir) if args.log_dir else results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir / f"train_{exp_id}.log")

    if args.smoke:
        cfg.setdefault("data", {})["num_workers"] = min(
            int(cfg.get("data", {}).get("num_workers", 4)), 2
        )
        cfg.setdefault("train", {})["epochs"] = 1
        if args.max_minutes is None:
            cfg["train"]["max_train_minutes"] = 6.0

    seed = int(cfg.get("runtime", {}).get("seed", 42))
    set_seed(seed)
    logger.info("=" * 78)
    logger.info("%s | config=%s | hash=%s", exp_id, cfg.get("_config_path"), config_hash(cfg))
    logger.info("%s | seed=%s | %s", exp_id, seed, seed_everything_strict_note())
    logger.info("=" * 78)
    logger.info("effective config:\n%s", config_to_json(cfg))

    environment = log_environment(logger)
    splits = describe_splits(cfg, repo_root=REPO_ROOT)
    logger.info("split inventory: %s", json.dumps(splits, indent=2))

    reference, reference_provenance = resolve_reference(cfg, repo_root=REPO_ROOT)
    if args.reference:
        reference_provenance = f"cli:{args.reference}"

    session_clock = SessionClock(output_dir / "time.json", session_minutes=float(
        (cfg.get("session", {}) or {}).get("session_minutes", 720.0)
    ))
    consumed = session_clock.consumed_seconds(exp_id)
    if consumed > 0:
        logger.info(
            "%s | previous sessions already spent %.1f min on this cell",
            exp_id,
            consumed / 60.0,
        )

    budget = TimeBudget.from_config(
        cfg,
        label=exp_id,
        max_minutes=args.max_minutes,
        reserve_minutes=args.reserve_minutes,
        session_start=args.session_start,
        consumed_before=consumed,
    )
    budget.log(logger)
    logger.info("%s | session clock exit estimate: %s", exp_id, session_clock.estimated_exit_time())

    limit_train = args.limit_train or (200 if args.smoke else None)
    limit_val = args.limit_val or (100 if args.smoke else None)

    train_loader, val_loader = build_train_val_loaders(
        cfg,
        reference=reference,
        repo_root=REPO_ROOT,
        limit_train=limit_train,
        limit_val=limit_val,
    )
    logger.info(
        "%s | train batches=%d val batches=%d train_rows=%d val_rows=%d",
        exp_id,
        len(train_loader),
        len(val_loader),
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    trainer = Trainer(
        cfg=cfg,
        exp_id=exp_id,
        output_root=output_dir,
        repo_root=REPO_ROOT,
        reference=reference,
        reference_provenance=reference_provenance,
        resume=not args.no_resume,
        time_budget=budget,
    )
    max_epochs = 1 if args.smoke else None
    try:
        result = trainer.fit(train_loader, val_loader, max_epochs=max_epochs)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.error(
                "%s | CUDA OOM outside the trainer loop: %s. Re-run with a smaller "
                "--batch-size (grad_accum_steps is scaled automatically by the "
                "orchestrator).", exp_id, exc,
            )
            return 3
        raise

    session_clock.record(exp_id, result.trained_seconds, {"stop_reason": result.stop_reason})

    run_report = {
        "exp_id": exp_id,
        "config_path": str(cfg.get("_config_path")),
        "config_hash": config_hash(cfg),
        "seed": seed,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "train_result": result.as_dict(),
        "train_history": result.train_history,
        "val_history": result.val_history,
        "time_budget": budget.as_dict(),
        "session_clock": session_clock.summary(),
        "environment": environment,
        "split_inventory": splits,
        "reference_provenance": reference_provenance,
        "cached_splits_dir": cfg.get("_cached_splits_dir"),
        "online_normalization_disabled_because_cached": cfg.get("_online_normalization"),
        "effective_config": {k: v for k, v in cfg.items() if not str(k).startswith("_")},
        "checkpoints_size_gb": total_size_gb([output_dir / exp_id]),
        "disk_free_gb": free_space_gb(output_dir if output_dir.exists() else "."),
        "wall_minutes": round((time.time() - start) / 60.0, 3),
    }

    evaluation: Optional[Dict[str, Any]] = None
    if args.eval_test_id or args.eval_test_ood:
        splits_to_eval = ["test_id"] if not args.eval_test_ood else ["test_id", "test_ood"]
        if args.eval_test_ood:
            logger.warning(
                "%s | OOD audit requested from train.py. Running it strictly AFTER "
                "training has finished and checkpoints are frozen (docs/PLAN.md risk 3).",
                exp_id,
            )
        trainer.setup_model()
        if not trainer.load_best_weights():
            logger.error("%s | no checkpoint available; skipping evaluation", exp_id)
        else:
            loaders = build_eval_loaders(
                cfg,
                splits=splits_to_eval,
                reference=reference,
                repo_root=REPO_ROOT,
                limit_test_id=args.limit_test_id or (200 if args.smoke else None),
                limit_test_ood=200 if args.smoke else None,
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
                save_samples=args.save_samples,
                extra_meta={
                    "backbone": (cfg.get("model", {}) or {}).get("backbone"),
                    "normalization": (cfg.get("normalization", {}) or {}).get("name"),
                    "augmentation": (cfg.get("augmentation", {}) or {}).get("policy"),
                    "stop_reason": result.stop_reason,
                    "epochs_completed": result.epochs_completed,
                    "best_val_macro_f1": result.best_val_macro_f1,
                },
            )
        run_report["evaluation"] = {
            "splits_evaluated": ["test_id", "test_ood"]
            if (evaluation or {}).get("test_ood")
            else (["test_id"] if evaluation else []),
            "test_id_macro_f1": (evaluation or {}).get("test_id", {}).get("macro_f1"),
            "test_ood_macro_f1": (evaluation or {}).get("test_ood", {}).get("macro_f1"),
            "delta_f1": (evaluation or {}).get("robustness", {}).get("delta_macro_f1"),
            "rr_f1": (evaluation or {}).get("robustness", {}).get("rr_macro_f1"),
        }

    report_path = output_dir / exp_id / "train_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(run_report, indent=2, default=str), encoding="utf-8")

    row = build_summary_row(exp_id, cfg, result.as_dict(), evaluation)
    row["notes"] = f"stop={result.stop_reason}; {reference_provenance}"
    summary_csv = results_dir / "metrics" / "summary_results.csv"
    existing: List[Dict[str, Any]] = []
    if summary_csv.exists():
        import pandas as pd

        existing = pd.read_csv(summary_csv).to_dict("records")
        existing = [r for r in existing if str(r.get("exp_id")) != exp_id]
    write_metrics_csv(existing + [row], summary_csv)

    print("\n" + "=" * 78)
    print(f"TRAIN SUMMARY -- {exp_id}")
    print("=" * 78)
    print(f"  status            : {result.status} ({result.stop_reason})")
    print(f"  epochs completed  : {result.epochs_completed}  (best epoch {result.best_epoch})")
    print(f"  best val Macro-F1 : {result.best_val_macro_f1:.4f}")
    print(f"  wall time         : {result.trained_seconds / 60.0:.1f} min")
    print(f"  checkpoints       : {result.best_path or '-'}")
    print(f"  checkpoint budget : {json.dumps(result.checkpoint_report.get('current_gb'))} GB in run dir")
    if evaluation:
        print(f"  test_id Macro-F1  : {evaluation.get('test_id', {}).get('macro_f1')}")
        print(f"  test_ood Macro-F1 : {evaluation.get('test_ood', {}).get('macro_f1')}")
    print(f"  report            : {report_path}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
