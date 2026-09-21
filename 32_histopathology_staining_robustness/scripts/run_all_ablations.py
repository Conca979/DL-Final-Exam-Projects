#!/usr/bin/env python
"""Orchestrator: run the 13-cell ablation matrix across (possibly many) sessions.

This is the script the Kaggle notebook calls once.  It is designed so that the
human can hit "Save & Run All (Commit)", walk away, and come back to a cleanly
terminated session whose outputs are safe to download -- even if the matrix does
not fit in one 12-hour window.

Session discipline
------------------
1. Read ``configs/experiments_registry.json`` (the single source of truth for the
   matrix order) and the on-disk manifest.
2. Skip every cell already marked ``completed`` with an evaluated result -- a
   resumed session never re-trains finished work.
3. Split the remaining wall clock across the remaining cells with
   :class:`~histo_robust.utils.timebudget.BudgetAllocator`, capped by
   ``--per-exp-minutes`` and floored at ``--min-minutes``.
4. For each cell: train (auto-resuming from ``last.pt`` if the previous session
   was cut off mid-cell), then evaluate ``test_id`` and (optionally) ``test_ood``.
5. Stop *before* the 12-hour kill, write the manifest, and zip the checkpoints
   for re-upload.  Cells that did not fit are simply picked up next session.

Example (Kaggle)::

    python scripts/run_all_ablations.py \
        --checkpoint-dir /kaggle/working/checkpoints \
        --results-dir    /kaggle/working/results \
        --session-start "$SESSION_START" \
        --session-minutes 700 --reserve-minutes 20 \
        --per-exp-minutes 75 --min-minutes 6 \
        --eval-test-ood
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from histo_robust.utils.config import config_hash, load_config  # noqa: E402
from histo_robust.utils.kaggle import configure_logging, log_environment  # noqa: E402
from histo_robust.utils.seed import set_seed  # noqa: E402
from histo_robust.utils.timebudget import (  # noqa: E402
    STOP_EARLY_STOP,
    BudgetAllocator,
    SessionClock,
    TimeBudget,
    _session_start_epoch,
)

logger = logging.getLogger("run_all_ablations")

DEFAULT_REGISTRY = REPO_ROOT / "configs" / "experiments_registry.json"


# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--experiments", nargs="*", default=None, help="Subset of exp ids (default: every registry entry).")
    parser.add_argument("--per-exp-minutes", type=float, default=75.0, help="Cap per ablation cell.")
    parser.add_argument("--min-minutes", type=float, default=6.0, help="Do not start a cell with less than this left.")
    parser.add_argument("--session-minutes", type=float, default=700.0, help="Budgeted session length (Kaggle kills at 720).")
    parser.add_argument("--reserve-minutes", type=float, default=20.0, help="Time held back for finalisation/zipping.")
    parser.add_argument("--session-start", default=None)
    parser.add_argument("--max-total-minutes", type=float, default=None, help="Hard cap on this orchestrator's own wall clock.")
    parser.add_argument("--eval-test-ood", action="store_true", help="Run the OOD audit after each cell (recommended).")
    parser.add_argument("--force", action="store_true", help="Re-run cells already marked completed.")
    parser.add_argument("--no-resume", action="store_true", help="Never resume mid-cell; restart it.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--keep-last", type=int, default=None)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--reference", default=None)
    parser.add_argument(
        "--normalization-cache",
        default=None,
        help="Pre-normalised tile directory (output of scripts/preprocess_normalize.py).",
    )
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--oom-retries", type=int, default=2, help="Retries per cell with halved batch size on CUDA OOM.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args(argv)


def load_registry(path: str | Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    experiments = payload.get("experiments") if isinstance(payload, dict) else payload
    if not experiments:
        raise ValueError(f"Registry {path} has no 'experiments' list")
    return list(experiments)


# ---------------------------------------------------------------------------
class Manifest:
    """Durable record of which cells are finished, and where their results live.

    Lives in ``results/metrics/ablation_manifest.json`` so it travels with the
    downloaded outputs and can be re-uploaded (or simply inspected) next session.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Any] = {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "sessions": [],
            "experiments": {},
        }
        self.load()
        self.data.setdefault("sessions", []).append(
            {
                "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                "pid": __import__("os").getpid(),
                "host": __import__("socket").gethostname(),
            }
        )

    def load(self) -> None:
        if self.path.exists():
            try:
                self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read manifest %s (%s)", self.path, exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------
    def get(self, exp_id: str) -> Dict[str, Any]:
        return dict((self.data.get("experiments", {}) or {}).get(exp_id, {}))

    def is_complete(self, exp_id: str) -> Tuple[bool, str]:
        """A cell counts as done only when it trained to its planned stopping point.

        ``status`` alone is not enough: a cell that was time-boxed mid-training is
        evaluated so its interim numbers are visible, but it must still be resumed
        next session.
        """
        entry = self.get(exp_id)
        if entry.get("status") != "completed":
            return False, f"status={entry.get('status', 'absent')}"
        best = entry.get("best_path")
        if best and not Path(best).exists():
            return False, "checkpoint is gone (not in this session's /kaggle/working)"
        planned = entry.get("planned_epochs")
        done = entry.get("epochs_completed")
        early = entry.get("stop_reason") == STOP_EARLY_STOP
        if not early and planned and done is not None and int(done) < int(planned):
            return False, f"time-boxed at {done}/{planned} epochs"
        if not entry.get("test_id_macro_f1"):
            return False, "no in-domain (test_id) evaluation recorded"
        return True, "completed"

    def update(self, exp_id: str, **fields: Any) -> None:
        experiments = self.data.setdefault("experiments", {})
        entry = experiments.setdefault(exp_id, {})
        entry.update(fields)
        entry["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        self.save()


# ---------------------------------------------------------------------------
def build_cell_cfg(
    entry: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    cfg = load_config(entry["config"], overrides=args.set, repo_root=REPO_ROOT)

    def _set(path: str, value: Any) -> None:
        if value is None:
            return
        node = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    _set("train.epochs", args.epochs)
    _set("train.batch_size", args.batch_size)
    _set("data.batch_size", args.batch_size)
    _set("checkpoints.save_every_steps", args.save_every)
    _set("checkpoints.keep_last", args.keep_last)
    _set("paths.weights_dir", args.weights_dir)
    _set("normalization.reference_path", args.reference)
    if args.normalization_cache is not None:
        _set("paths.normalization_cache", args.normalization_cache or None)
    if args.search_root:
        existing = list((cfg.get("paths", {}) or {}).get("search_roots", []) or [])
        cfg.setdefault("paths", {})["search_roots"] = existing + list(args.search_root)

    # Repo-relative reference tile -> absolute (it ships inside the codebase zip).
    reference_path = (cfg.get("normalization", {}) or {}).get("reference_path")
    if reference_path and not Path(str(reference_path)).is_absolute():
        candidate = REPO_ROOT / str(reference_path)
        if candidate.exists():
            cfg.setdefault("normalization", {})["reference_path"] = str(candidate)

    # Adopt a complete pre-normalisation cache before resolving the reference:
    # cached tiles are already normalised, so the effective normaliser is identity.
    cached_dir = prefer_cached_splits(cfg)
    if cached_dir:
        logger.info("%s | using pre-normalised cache %s", cfg.get("exp_id", "?"), cached_dir)
    return cfg


def shrink_batch_cfg(cfg: Dict[str, Any], factor: int = 2) -> Dict[str, Any]:
    """Halve the per-device batch and double accumulation (constant effective batch)."""
    import copy

    retry = copy.deepcopy(cfg)
    data = retry.setdefault("data", {})
    train = retry.setdefault("train", {})
    batch = int(data.get("batch_size", 64))
    new_batch = max(4, batch // factor)
    if new_batch == batch:
        return retry
    accum = max(1, int(train.get("grad_accum_steps", 1)))
    data["batch_size"] = new_batch
    train["grad_accum_steps"] = int(round(accum * batch / new_batch))
    logger.warning(
        "OOM retry: batch_size %d -> %d, grad_accum_steps %d -> %d (effective batch held at %d)",
        batch, new_batch, accum, train["grad_accum_steps"], new_batch * train["grad_accum_steps"],
    )
    return retry


def run_one_cell(
    entry: Dict[str, Any],
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    reference: Any,
    reference_provenance: str,
    session_clock: SessionClock,
    minutes: float,
    manifest: Manifest,
    session_deadline_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    """Train + evaluate a single ablation cell, with an OOM retry ladder."""
    exp_id = entry["exp_id"]
    attempt_cfg = cfg
    last_error: Optional[str] = None
    for attempt in range(args.oom_retries + 1):
        if attempt:
            attempt_cfg = shrink_batch_cfg(attempt_cfg)
        logger.info(
            "=" * 78
            + f"\n{exp_id} | {entry.get('research_question', '')}\n"
            + f"{exp_id} | backbone={entry.get('backbone')} norm={entry.get('normalization')} "
            + f"aug={entry.get('augmentation')} budget={minutes:.1f} min (attempt {attempt + 1})"
        )
        try:
            train_loader, val_loader = build_train_val_loaders(
                attempt_cfg, reference=reference, repo_root=REPO_ROOT
            )
            budget = TimeBudget(
                max_minutes=minutes,
                reserve_minutes=args.reserve_minutes,
                session_deadline_epoch=session_deadline_epoch,
                label=exp_id,
                consumed_before=session_clock.consumed_seconds(exp_id),
            )
            trainer = Trainer(
                cfg=attempt_cfg,
                exp_id=exp_id,
                output_root=args.checkpoint_dir,
                repo_root=REPO_ROOT,
                reference=reference,
                reference_provenance=reference_provenance,
                resume=not args.no_resume,
                time_budget=budget,
            )
            result = trainer.fit(train_loader, val_loader)
            session_clock.record(exp_id, result.trained_seconds, {"stop_reason": result.stop_reason})

            evaluation: Optional[Dict[str, Any]] = None
            if result.status != "failed" and result.best_path:
                splits = ["test_id", "test_ood"] if args.eval_test_ood else ["test_id"]
                loaders = build_eval_loaders(
                    attempt_cfg,
                    splits=splits,
                    reference=reference,
                    repo_root=REPO_ROOT,
                    post_training=True,
                )
                if not trainer.load_best_weights():
                    logger.error("%s | checkpoint unusable; skipping evaluation", exp_id)
                else:
                    evaluation = evaluate_experiment(
                        cfg=attempt_cfg,
                        exp_id=exp_id,
                        model=trainer.model,
                        loaders=loaders,
                        output_root=Path(args.results_dir) / "per_experiment",
                        class_names=list(attempt_cfg.get("data", {}).get("class_names", [])),
                        amp=bool((attempt_cfg.get("train", {}) or {}).get("amp", True)),
                        extra_meta={
                            "stage": entry.get("stage"),
                            "backbone": entry.get("backbone"),
                            "normalization": entry.get("normalization"),
                            "augmentation": entry.get("augmentation"),
                            "config_hash": config_hash(attempt_cfg),
                            "attempt": attempt + 1,
                        },
                    )

            # A cell only counts as COMPLETE when its training actually ran to the
            # planned stopping point (all epochs, or early stopping). A time-boxed
            # cell still gets evaluated -- the numbers are real, just interim --
            # but it is recorded as "interrupted" so the next session resumes it
            # instead of treating a partial curve as the final result.
            planned_epochs = int((attempt_cfg.get("train", {}) or {}).get("epochs", 12))
            finished = result.status == "completed" and result.epochs_completed >= planned_epochs
            early_stopped = result.stop_reason == "early_stopping"
            status = "completed" if (finished or early_stopped) else "interrupted"
            if status == "interrupted":
                logger.warning(
                    "%s | time-boxed after %d/%d epochs (stop=%s); results are INTERIM "
                    "and this cell will resume next session.",
                    exp_id,
                    result.epochs_completed,
                    planned_epochs,
                    result.stop_reason,
                )

            if result.status != "failed" and result.best_path:
                splits = ["test_id", "test_ood"] if args.eval_test_ood else ["test_id"]
                loaders = build_eval_loaders(
                    attempt_cfg,
                    splits=splits,
                    reference=reference,
                    repo_root=REPO_ROOT,
                    post_training=True,
                )
                if not trainer.load_best_weights():
                    logger.error("%s | checkpoint unusable; skipping evaluation", exp_id)
                else:
                    evaluation = evaluate_experiment(
                        cfg=attempt_cfg,
                        exp_id=exp_id,
                        model=trainer.model,
                        loaders=loaders,
                        output_root=Path(args.results_dir) / "per_experiment",
                        class_names=list(attempt_cfg.get("data", {}).get("class_names", [])),
                        amp=bool((attempt_cfg.get("train", {}) or {}).get("amp", True)),
                        extra_meta={
                            "stage": entry.get("stage"),
                            "backbone": entry.get("backbone"),
                            "normalization": entry.get("normalization"),
                            "augmentation": entry.get("augmentation"),
                            "config_hash": config_hash(attempt_cfg),
                            "attempt": attempt + 1,
                            "run_status": status,
                            "epochs_completed": result.epochs_completed,
                            "planned_epochs": planned_epochs,
                        },
                    )

            manifest.update(
                exp_id,
                status=status,
                stop_reason=result.stop_reason,
                epochs_completed=result.epochs_completed,
                planned_epochs=planned_epochs,
                best_val_macro_f1=result.best_val_macro_f1,
                best_epoch=result.best_epoch,
                trained_seconds=result.trained_seconds,
                best_path=result.best_path,
                last_path=result.last_path,
                stage=entry.get("stage"),
                config=entry.get("config"),
                config_hash=config_hash(attempt_cfg),
                batch_size=(attempt_cfg.get("data", {}) or {}).get("batch_size"),
                # Flat, greppable evaluation summary for the manifest and the
                # session-level tables (the full tables live in the per-experiment
                # JSON/CSV artefacts).
                evaluation=(evaluation or {}).get("robustness", {}),
                test_id_macro_f1=((evaluation or {}).get("test_id") or {}).get("macro_f1"),
                test_id_accuracy=((evaluation or {}).get("test_id") or {}).get("accuracy"),
                test_ood_macro_f1=((evaluation or {}).get("test_ood") or {}).get("macro_f1"),
                test_ood_accuracy=((evaluation or {}).get("test_ood") or {}).get("accuracy"),
                delta_f1=((evaluation or {}).get("robustness") or {}).get("delta_macro_f1"),
                rr_f1=((evaluation or {}).get("robustness") or {}).get("rr_macro_f1"),
                attempt=attempt + 1,
            )

            del trainer
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            return {
                "exp_id": exp_id,
                "status": status,
                "stop_reason": result.stop_reason,
                "best_val_macro_f1": result.best_val_macro_f1,
                "trained_minutes": result.trained_seconds / 60.0,
                "evaluation": evaluation or {},
                "train_result": result.as_dict(),
                "config": attempt_cfg,
            }
        except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the session
            last_error = f"{type(exc).__name__}: {exc}"
            is_oom = "out of memory" in str(exc).lower()
            logger.error(
                "%s | attempt %d failed (%s)%s",
                exp_id,
                attempt + 1,
                last_error,
                " -- retrying with a smaller batch" if is_oom and attempt < args.oom_retries else "",
            )
            if not is_oom or attempt >= args.oom_retries:
                break
        finally:
            gc.collect()

    manifest.update(
        exp_id,
        status="failed",
        error=last_error,
        stage=entry.get("stage"),
        config=entry.get("config"),
    )
    return {"exp_id": exp_id, "status": "failed", "error": last_error, "evaluation": {}}


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    session_start_epoch = time.time()

    results_dir = Path(args.results_dir)
    log_dir = Path(args.log_dir) if args.log_dir else results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir / "run_all_ablations.log")

    logger.info("=" * 78)
    logger.info("ABLATION ORCHESTRATOR START")
    logger.info("=" * 78)
    log_environment(logger)

    registry = load_registry(args.registry)
    if args.experiments:
        wanted = {e.upper() for e in args.experiments}
        registry = [e for e in registry if str(e["exp_id"]).upper() in wanted]
        missing = wanted - {str(e["exp_id"]).upper() for e in registry}
        if missing:
            logger.error("Unknown experiment ids requested: %s", sorted(missing))
            return 2

    manifest = Manifest(results_dir / "metrics" / "ablation_manifest.json")
    session_clock = SessionClock(Path(args.checkpoint_dir) / "time.json", session_minutes=args.session_minutes)

    deadline = (
        (_session_start_epoch(args.session_start) or session_start_epoch)
        + args.session_minutes * 60.0
    )
    if args.max_total_minutes:
        deadline = min(deadline, session_start_epoch + args.max_total_minutes * 60.0)

    pending: List[str] = []
    plan: List[Dict[str, Any]] = []
    for entry in registry:
        exp_id = entry["exp_id"]
        complete, reason = manifest.is_complete(exp_id)
        if complete and not args.force:
            logger.info("%s | SKIP (%s)", exp_id, reason)
            plan.append({"exp_id": exp_id, "action": "skip", "reason": reason})
            continue
        if not complete:
            logger.info("%s | PENDING (%s)", exp_id, reason)
        else:
            logger.info("%s | FORCED re-run", exp_id)
        pending.append(exp_id)
        plan.append({"exp_id": exp_id, "action": "run", "reason": reason})

    allocator = BudgetAllocator(
        deadline_epoch=deadline,
        reserve_minutes=args.reserve_minutes,
        per_exp_minutes=args.per_exp_minutes,
        min_minutes=args.min_minutes,
    )
    allocation = allocator.allocate(pending)
    logger.info("allocation: %s", json.dumps({k: round(v, 1) for k, v in allocation.items()}))
    logger.info(
        "usable window: %.1f min | reserve: %.1f min | per-cell cap: %.1f min",
        (allocator.usable_seconds() or 0.0) / 60.0,
        args.reserve_minutes,
        args.per_exp_minutes,
    )

    if args.dry_run:
        print(json.dumps({"plan": plan, "allocation": allocation}, indent=2))
        return 0

    splits_info = describe_splits(load_config(registry[0]["config"], repo_root=REPO_ROOT), repo_root=REPO_ROOT)
    logger.info("split inventory: %s", json.dumps(splits_info, indent=2))

    # The canonical stain reference is resolved ONCE per session, from the first
    # registry cell that actually requests normalisation, and reused for every
    # cell.  A per-cell reference would make Axis B incomparable, and the OOD
    # split is never consulted here (firewall).
    reference, reference_provenance = (None, "unused:no-normalizing-cell")
    for entry in registry:
        probe_cfg = load_config(entry["config"], repo_root=REPO_ROOT)
        probe_cfg = build_cell_cfg(entry, args)
        if str((probe_cfg.get("normalization", {}) or {}).get("name", "none")).lower() in {
            "none",
            "raw",
            "identity",
            "",
        }:
            continue
        reference, reference_provenance = resolve_reference(probe_cfg, repo_root=REPO_ROOT)
        logger.info(
            "canonical stain reference resolved from %s -> %s", entry["exp_id"], reference_provenance
        )
        break

    outcomes: List[Dict[str, Any]] = []
    for entry in registry:
        exp_id = entry["exp_id"]
        if exp_id not in allocation:
            continue

        remaining = deadline - args.reserve_minutes * 60.0 - time.time()
        if remaining <= args.min_minutes * 60.0:
            logger.warning(
                "STOPPING before %s: only %.1f min left in the window (need >= %.1f). "
                "Re-run this same command next session -- completed cells are skipped "
                "and this cell resumes from its last.pt.",
                exp_id,
                remaining / 60.0,
                args.min_minutes,
            )
            break

        minutes = min(allocation[exp_id], remaining / 60.0)
        cell_cfg = build_cell_cfg(entry, args)
        set_seed(int((cell_cfg.get("runtime", {}) or {}).get("seed", 42)))

        outcome = run_one_cell(
            entry=entry,
            cfg=cell_cfg,
            args=args,
            reference=reference,
            reference_provenance=reference_provenance,
            session_clock=session_clock,
            minutes=minutes,
            manifest=manifest,
            session_deadline_epoch=deadline,
        )
        outcomes.append(outcome)

        # Keep the aggregated summary table current after every cell.
        rows: List[Dict[str, Any]] = []
        for other in registry:
            record = manifest.get(other["exp_id"])
            if not record:
                continue
            evaluation_path = (
                Path(args.results_dir) / "per_experiment" / other["exp_id"] / "evaluation_result.json"
            )
            evaluation = None
            if evaluation_path.exists():
                try:
                    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    evaluation = None
            other_cfg = load_config(other["config"], repo_root=REPO_ROOT)
            rows.append(build_summary_row(other["exp_id"], other_cfg, record, evaluation))
        write_metrics_csv(rows, results_dir / "metrics" / "summary_results.csv")

        manifest.data["session_summary"] = session_clock.summary()
        manifest.save()

    if not outcomes:
        logger.warning("No cell was executed in this session (everything complete or out of time).")

    summary = {
        "session_started_utc": datetime.fromtimestamp(session_start_epoch, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "session_end_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "session_minutes": args.session_minutes,
        "reserve_minutes": args.reserve_minutes,
        "per_exp_minutes_cap": args.per_exp_minutes,
        "executed": [
            {
                "exp_id": o["exp_id"],
                "status": o["status"],
                "stop_reason": o.get("stop_reason"),
                "best_val_macro_f1": o.get("best_val_macro_f1"),
                "trained_minutes": round(o.get("trained_minutes", 0.0), 2),
                "test_id_macro_f1": (o.get("evaluation", {}).get("test_id") or {}).get("macro_f1"),
                "test_ood_macro_f1": (o.get("evaluation", {}).get("test_ood") or {}).get("macro_f1"),
                "delta_f1": (o.get("evaluation", {}).get("robustness") or {}).get("delta_macro_f1"),
            }
            for o in outcomes
        ],
        "remaining_not_run": [e["exp_id"] for e in registry if e["exp_id"] not in {o["exp_id"] for o in outcomes} and not manifest.is_complete(e["exp_id"])[0]],
        "session_clock": session_clock.summary(),
        "checkpoint_bytes_gb": round(total_size_gb([args.checkpoint_dir]), 3),
        "disk_free_gb": free_space_gb(args.checkpoint_dir),
        "estimated_exit": session_clock.estimated_exit_time(),
    }
    (results_dir / "metrics" / "session_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    manifest.data.setdefault("sessions", [])[-1].update(summary)
    manifest.save()

    print("\n" + "=" * 78)
    print("ABLATION SESSION SUMMARY")
    print("=" * 78)
    for row in summary["executed"]:
        print(
            f"  {row['exp_id']:<7} {row['status']:<12} val_f1={row['best_val_macro_f1']} "
            f"id_f1={row['test_id_macro_f1']} ood_f1={row['test_ood_macro_f1']} "
            f"delta={row['delta_f1']} ({row['trained_minutes']} min)"
        )
    if summary["remaining_not_run"]:
        print(f"  NOT RUN YET: {', '.join(summary['remaining_not_run'])}")
    print(f"  session trained: {summary['session_clock']['total_trained_minutes']} min")
    print(f"  checkpoints on disk: {summary['checkpoint_bytes_gb']} GB")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
