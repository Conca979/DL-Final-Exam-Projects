"""Training / inference engine.

The engine is built around three hard Kaggle constraints:

**1. The 12-hour hard kill (short-session recipe).**
:class:`Trainer` never runs "for as many epochs as it takes".  It holds a
:class:`~histo_robust.utils.timebudget.TimeBudget` and asks it between
*every* micro-batch.  When the budget (default 630 min = 10.5 h) or the session
reserve is exhausted it:

* breaks out of the batch loop immediately (never mid-write),
* writes ``last.pt`` with model + optimizer + scheduler + AMP scaler + RNG state,
* writes ``best.pt`` if this epoch improved ``val_id`` Macro-F1,
* returns normally with ``stop_reason='run_time_budget'``,

so Kaggle still records the version as successful and the next session resumes
from the newest checkpoint instead of restarting from epoch 0.

**2. The 20 GB output cap.**  Every write goes through
:class:`~histo_robust.utils.checkpoint.CheckpointManager`, which keeps
``--keep-last N`` rolling checkpoints plus ``best.pt``/``last.pt`` and prunes the
rest.  ``--save-every K`` controls step granularity.

**3. The domain firewall.**  ``Trainer`` can only be handed ``train_loader`` and
``val_id_loader`` (see :func:`histo_robust.data.datamodule.build_train_val_loaders`).
There is no parameter through which ``CRC-VAL-HE-7K`` could enter -- checkpoint
selection happens exclusively on ``val_id`` Macro-F1.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..models import build_classifier
from ..utils.checkpoint import CheckpointManager, load_checkpoint
from ..utils.metrics import compute_classification_metrics
from ..utils.seed import get_rng_state, seed_worker, set_rng_state, set_seed
from ..utils.timebudget import (
    STOP_EARLY_STOP,
    STOP_MAX_EPOCHS,
    STOP_NONE,
    STOP_OOM,
    STOP_RUN_BUDGET,
    TimeBudget,
)

logger = logging.getLogger(__name__)

__all__ = ["Trainer", "TrainResult", "Predictions", "run_inference", "build_optimizer"]


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


@dataclass
class Predictions:
    """Model outputs for one split, with everything needed to recompute metrics."""

    y_true: np.ndarray
    probs: np.ndarray
    paths: List[str] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)

    @property
    def y_pred(self) -> np.ndarray:
        return self.probs.argmax(axis=1)

    def metrics(self, class_names: Sequence[str]) -> Dict[str, Any]:
        return compute_classification_metrics(self.y_true, self.probs, class_names)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            y_true=self.y_true.astype(np.int16),
            probs=self.probs.astype(np.float32),
            indices=np.asarray(self.indices, dtype=np.int64),
            paths=np.asarray(self.paths, dtype=object),
        )
        return path


@dataclass
class TrainResult:
    """Everything the orchestrator needs from one experiment."""

    exp_id: str
    status: str = "completed"  # completed | interrupted | failed
    stop_reason: str = STOP_NONE
    epochs_completed: int = 0
    global_step: int = 0
    best_val_macro_f1: float = float("nan")
    best_epoch: int = -1
    trained_seconds: float = 0.0
    train_history: List[Dict[str, Any]] = field(default_factory=list)
    val_history: List[Dict[str, Any]] = field(default_factory=list)
    final_val_metrics: Dict[str, Any] = field(default_factory=dict)
    best_path: Optional[str] = None
    last_path: Optional[str] = None
    resume_source: Optional[str] = None
    checkpoint_report: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload = dict(self.__dict__)
        payload.pop("train_history", None)
        payload.pop("val_history", None)
        return payload


def build_optimizer(
    model: Any,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    head_lr_mult: float = 1.0,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> Any:
    torch, _ = _torch()
    groups = model.parameter_groups(
        lr=lr, head_lr_mult=head_lr_mult, weight_decay=weight_decay
    )
    if not any(len(g["params"]) for g in groups):
        raise RuntimeError("Optimizer got no trainable parameters")
    return torch.optim.AdamW(groups, betas=betas)


def build_scheduler(optimizer: Any, cfg: Dict[str, Any], steps_per_epoch: int) -> Optional[Any]:
    """Cosine (step- or epoch-based) or step LR schedule, or ``None``."""
    torch, _ = _torch()
    train_cfg = dict(cfg.get("train", {}) or {})
    name = str(train_cfg.get("scheduler", "cosine")).lower()
    if name in {"none", "constant", ""}:
        return None

    epochs = max(1, int(train_cfg.get("epochs", 12)))
    warmup_epochs = float(train_cfg.get("warmup_epochs", 1.0))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.01))
    total_steps = max(1, epochs * max(1, steps_per_epoch))
    warmup_steps = int(max(0.0, warmup_epochs) * max(1, steps_per_epoch))

    if name == "cosine":
        def lr_lambda(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return max(1e-6, (step + 1) / float(warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if name in {"step", "steplr"}:
        step_size = int(train_cfg.get("step_size_epochs", 4))
        gamma = float(train_cfg.get("gamma", 0.3))
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, step_size * max(1, steps_per_epoch)), gamma=gamma
        )

    raise ValueError(f"Unknown scheduler '{name}' (cosine | step | none)")


def run_inference(
    model: Any,
    loader: Any,
    device: Any,
    amp: bool = True,
    desc: str = "inference",
    max_batches: Optional[int] = None,
) -> Predictions:
    """Forward-only pass that returns probabilities and labels for a loader."""
    torch, _ = _torch()
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_paths: List[str] = []
    all_indices: List[int] = []

    autocast = torch.cuda.amp.autocast(enabled=bool(amp))
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            with autocast:
                logits = model(images)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_indices.extend([int(i) for i in batch["index"].tolist()])
            paths = batch.get("path", [])
            all_paths.extend([str(p) for p in paths])

    if not all_probs:
        return Predictions(
            y_true=np.zeros(0, dtype=int), probs=np.zeros((0, 0), dtype=float)
        )
    return Predictions(
        y_true=np.concatenate(all_labels).astype(int),
        probs=np.concatenate(all_probs, axis=0),
        paths=all_paths,
        indices=all_indices,
    )


class Trainer:
    """Time-boxed trainer for a single ablation cell.

    Parameters
    ----------
    cfg:
        Merged configuration dict.
    exp_id:
        Experiment identifier (``EXP-01`` ...), used for the output folder name.
    output_root:
        Typically ``/kaggle/working/checkpoints``.
    repo_root:
        Root of the extracted codebase, used to resolve split CSVs / reference.
    resume:
        When ``True`` (default) an existing ``last.pt`` in the run directory is
        loaded and training continues from the stored epoch/step.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        exp_id: str,
        output_root: str | Path,
        repo_root: Optional[str | Path] = None,
        reference: Optional[Any] = None,
        reference_provenance: str = "unset",
        resume: bool = True,
        time_budget: Optional[TimeBudget] = None,
        quiet: bool = False,
    ) -> None:
        self.cfg = cfg
        self.exp_id = exp_id
        self.repo_root = Path(repo_root) if repo_root is not None else None
        self.reference = reference
        self.reference_provenance = reference_provenance
        self.resume = bool(resume)
        self.quiet = quiet

        train_root = Path(output_root)
        self.run_dir = train_root / exp_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = dict(cfg.get("train", {}) or {})
        ckpt_cfg = dict(cfg.get("checkpoints", {}) or {})
        self.manager = CheckpointManager(
            run_dir=self.run_dir,
            save_every_steps=int(ckpt_cfg.get("save_every_steps", 500)),
            max_rolling=int(ckpt_cfg.get("keep_last", 3)),
            keep_best=int(ckpt_cfg.get("keep_best", 3)),
            run_quota_gb=float(ckpt_cfg.get("run_quota_gb", 3.0)),
        )
        self.time_budget = time_budget or TimeBudget.from_config(
            cfg,
            label=exp_id,
            max_minutes=float(train_cfg.get("max_train_minutes", 630)),
        )
        self.amp = bool(train_cfg.get("amp", True))
        self.grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
        self.label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
        self.early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))
        self.epochs = max(1, int(train_cfg.get("epochs", 12)))
        self.oom_retry = bool(train_cfg.get("oom_retry", True))
        self.oom_batch_floor = int(train_cfg.get("oom_batch_floor", 8))

        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.class_names: List[str] = list(
            (cfg.get("data", {}) or {}).get("class_names", [])
        )
        self.effective_batch_size = int(
            (cfg.get("data", {}) or {}).get("batch_size", 64)
        ) * self.grad_accum_steps

        self._feature_cache: Optional[Tuple[Any, Any]] = None
        self._epochs_seen = 0
        self._global_step = 0
        self._best_val_f1 = float("-inf")
        self._best_epoch = -1
        self._epochs_without_improvement = 0
        self._resume_source: Optional[str] = None
        self._train_history: List[Dict[str, Any]] = []
        self._val_history: List[Dict[str, Any]] = []
        self._wall_seconds_this_session = 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _resolve_device(self) -> Any:
        torch, _ = _torch()
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning(
            "CUDA unavailable inside Trainer; falling back to CPU. This is only "
            "sane for the pre-flight smoke test, not for a real 10-hour run."
        )
        return torch.device("cpu")

    def setup_model(self) -> Any:
        seed = int((self.cfg.get("runtime", {}) or {}).get("seed", 42))
        set_seed(seed)
        self.device = self._resolve_device()
        weights_dir = (self.cfg.get("paths", {}) or {}).get("weights_dir")
        pretrained = bool((self.cfg.get("model", {}) or {}).get("pretrained", True))
        self.model = build_classifier(self.cfg, pretrained=pretrained, weights_dir=weights_dir)
        self.model.to(self.device)
        self.model.train(True)
        if not self.quiet:
            logger.info(
                "%s | model=%s", self.exp_id, json.dumps(self.model.describe(), default=str)
            )
            logger.info(
                "%s | freeze_backbone=%s effective_batch=%d amp=%s device=%s",
                self.exp_id,
                self.model.freeze_backbone,
                self.effective_batch_size,
                self.amp,
                self.device,
            )
        return self.model

    def _build_train_components(self, steps_per_epoch: int) -> None:
        torch, _ = _torch()
        train_cfg = dict(self.cfg.get("train", {}) or {})
        self.optimizer = build_optimizer(
            self.model,
            lr=float(train_cfg.get("lr", 3e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.05)),
            head_lr_mult=float(train_cfg.get("head_lr_mult", 1.0)),
        )
        self.scheduler = build_scheduler(self.optimizer, self.cfg, steps_per_epoch)
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(self.amp))

    # ------------------------------------------------------------------
    # Feature caching (linear-probe path for frozen backbones)
    # ------------------------------------------------------------------
    def _build_feature_cache(
        self, train_loader: Any, val_loader: Any
    ) -> Tuple[Any, Any, Any, Any]:
        """Precompute frozen-backbone features for train/val (huge speedup)."""
        torch, _ = _torch()
        logger.info(
            "%s | freeze_backbone=True -> caching backbone features once "
            "(linear probe). This removes the backbone backward pass entirely.",
            self.exp_id,
        )
        cache: Dict[str, Tuple[Any, Any]] = {}
        for name, loader in (("train", train_loader), ("val", val_loader)):
            feats: List[Any] = []
            labels: List[Any] = []
            self.model.eval()
            with torch.no_grad():
                for batch in loader:
                    images = batch["image"].to(self.device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=bool(self.amp)):
                        out = self.model.module.backbone(images)
                    if out.ndim > 2:
                        out = out.mean(dim=tuple(range(2, out.ndim)))
                    feats.append(out.float().cpu())
                    labels.append(batch["label"])
            cache[name] = (
                torch.cat(feats, dim=0) if feats else torch.zeros((0, self.model.feature_dim)),
                torch.cat(labels, dim=0).long() if labels else torch.zeros((0,), dtype=torch.long),
            )
            logger.info(
                "%s | cached %s features: %s", self.exp_id, name, tuple(cache[name][0].shape)
            )
        self._feature_cache = (cache["train"][0], cache["train"][1], cache["val"][0], cache["val"][1])
        return self._feature_cache  # type: ignore[return-value]

    def _iterate_feature_batches(
        self, features: Any, labels: Any, batch_size: int, shuffle: bool = True
    ):
        torch, _ = _torch()
        n = features.shape[0]
        order = torch.randperm(n) if shuffle else torch.arange(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            yield (
                features[idx].to(self.device, non_blocking=True),
                labels[idx].to(self.device, non_blocking=True),
            )

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def _build_payload(self, epoch: int, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from ..utils.config import config_hash

        return {
            "format": "histo_robust/v1",
            "exp_id": self.exp_id,
            "epoch": int(epoch),
            "global_step": int(self._global_step),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "best_val_macro_f1": float(self._best_val_f1),
            "best_epoch": int(self._best_epoch),
            "rng": get_rng_state(),
            "config_hash": config_hash(self.cfg),
            "config": {
                k: v for k, v in self.cfg.items() if not str(k).startswith("_")
            },
            "metrics": metrics or {},
            "trained_seconds": float(self.time_budget.elapsed_seconds),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def _try_resume(self) -> bool:
        candidate = self.manager.last_path
        alt = None
        if not candidate.exists():
            steps = sorted(self.run_dir.glob("step_*.pt"))
            if steps:
                alt = steps[-1]
                candidate = alt
        if not candidate.exists():
            return False
        try:
            payload = load_checkpoint(candidate, map_location=self.device)
            self.model.load_state_dict(payload["model"], strict=False)
            if self.optimizer is not None and payload.get("optimizer"):
                self.optimizer.load_state_dict(payload["optimizer"])
            if self.scheduler is not None and payload.get("scheduler"):
                self.scheduler.load_state_dict(payload["scheduler"])
            if self.scaler is not None and payload.get("scaler"):
                self.scaler.load_state_dict(payload["scaler"])
            set_rng_state(payload.get("rng"))
            self._epochs_seen = int(payload.get("epoch", -1)) + 1
            self._global_step = int(payload.get("global_step", 0))
            self._best_val_f1 = float(payload.get("best_val_macro_f1", float("-inf")))
            self._best_epoch = int(payload.get("best_epoch", -1))
            self._resume_source = str(candidate)
            logger.info(
                "%s | RESUMED from %s (epoch=%d next, global_step=%d, best_val_f1=%.4f)",
                self.exp_id,
                candidate.name,
                self._epochs_seen,
                self._global_step,
                self._best_val_f1,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - a corrupt checkpoint must not kill the run
            logger.error(
                "%s | Could not resume from %s (%s: %s); starting fresh",
                self.exp_id,
                candidate,
                type(exc).__name__,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        max_epochs: Optional[int] = None,
        max_train_seconds: Optional[float] = None,
    ) -> TrainResult:
        torch, _ = _torch()
        start_wall = time.time()
        try:
            return self._fit_impl(
                train_loader=train_loader,
                val_loader=val_loader,
                max_epochs=max_epochs,
                max_train_seconds=max_train_seconds,
                start_wall=start_wall,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            logger.error(
                "%s | CUDA OOM (%s). The orchestrator will retry this cell with a "
                "smaller batch size and a matching gradient-accumulation increase, "
                "keeping the effective batch size at %d.",
                self.exp_id,
                exc,
                self.effective_batch_size,
            )
            self.time_budget.reserve(STOP_OOM)
            return TrainResult(
                exp_id=self.exp_id,
                status="failed",
                stop_reason=STOP_OOM,
                epochs_completed=self._epochs_seen,
                global_step=self._global_step,
                best_val_macro_f1=(
                    float(self._best_val_macro_f1) if self._best_val_macro_f1 > -1e8 else float("nan")
                ),
                best_epoch=self._best_epoch,
                trained_seconds=time.time() - start_wall,
                train_history=list(self._train_history),
                val_history=list(self._val_history),
                resume_source=self._resume_source,
                extra={"error": f"{type(exc).__name__}: {exc}", **self._oom_extra()},
            )

    def _oom_extra(self) -> Dict[str, Any]:
        return {
            "suggested_batch_size": max(
                self.oom_batch_floor, int(self.effective_batch_size / max(1, self.grad_accum_steps * 2))
            ),
            "oom_retry_enabled": self.oom_retry,
        }

    def _fit_impl(
        self,
        train_loader: Any,
        val_loader: Any,
        max_epochs: Optional[int],
        max_train_seconds: Optional[float],
        start_wall: float,
    ) -> TrainResult:
        torch, _ = _torch()
        epochs = int(max_epochs if max_epochs is not None else self.epochs)
        self.setup_model()

        steps_per_epoch = max(1, len(train_loader))
        self._build_train_components(steps_per_epoch)

        if max_train_seconds is not None:
            self.time_budget.max_minutes = max(0.05, float(max_train_seconds) / 60.0)
            self.time_budget.label = self.exp_id

        resumed = self._try_resume() if self.resume else False
        if not self.resume:
            logger.info("%s | resume disabled by --no-resume; starting from scratch", self.exp_id)

        use_feature_cache = bool(self.model.freeze_backbone)
        feature_batch = int((self.cfg.get("data", {}) or {}).get("batch_size", 64))
        if use_feature_cache:
            train_feats, train_labels, val_feats, val_labels = self._build_feature_cache(
                train_loader, val_loader
            )

        criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        status = "completed"
        stop_reason = STOP_NONE

        if self.time_budget.should_stop():
            logger.warning(
                "%s | time budget already exhausted before training (%s)",
                self.exp_id,
                self.time_budget.stop_reason,
            )
            stop_reason = self.time_budget.stop_reason
            return self._finalise(
                status="interrupted",
                stop_reason=stop_reason,
                start_wall=start_wall,
                val_loader=val_loader,
            )

        epoch = self._epochs_seen
        while epoch < epochs:
            self.time_budget.log(logger)
            epoch_metrics = self._train_one_epoch(
                epoch=epoch,
                train_loader=train_loader if not use_feature_cache else None,
                feature_data=(train_feats, train_labels) if use_feature_cache else None,
                feature_batch=feature_batch,
                criterion=criterion,
                steps_per_epoch=steps_per_epoch,
            )
            self._train_history.append(epoch_metrics)
            epoch_truncated = bool(epoch_metrics.get("truncated"))

            if epoch_truncated:
                # Partial epoch -> skip validation and the best-checkpoint update;
                # the run is about to stop anyway and every remaining second of
                # the session budget is better spent on the next cell.
                logger.warning(
                    "%s | epoch %d cut short by the time box after %d/%d batches; "
                    "skipping validation and stopping cleanly.",
                    self.exp_id,
                    epoch,
                    epoch_metrics.get("batches_done"),
                    epoch_metrics.get("batches_total"),
                )
                self.manager.save(
                    self._build_payload(epoch), self._global_step, kind="last", epoch=epoch
                )
                self.manager.enforce_quota()
                self._epochs_seen = epoch + 1
                stop_reason = self.time_budget.stop_reason
                status = "interrupted"
                break

            val_preds = run_inference(
                self.model,
                val_loader,
                self.device,
                amp=self.amp,
                desc=f"{self.exp_id} val",
            )
            val_metrics = val_preds.metrics(self.class_names)
            val_metrics["epoch"] = epoch
            val_metrics["global_step"] = self._global_step
            self._val_history.append(
                {k: v for k, v in val_metrics.items() if k not in {"per_class", "confusion_matrix"}}
            )

            improved = val_metrics["macro_f1"] > self._best_val_f1 + 1e-6
            if improved:
                self._best_val_f1 = float(val_metrics["macro_f1"])
                self._best_epoch = epoch
                self._epochs_without_improvement = 0
                self.manager.save(
                    self._build_payload(epoch, val_metrics), self._global_step, kind="best", epoch=epoch
                )
                self.manager.save(
                    self._build_payload(epoch, val_metrics),
                    self._global_step,
                    kind="best_snapshot",
                    epoch=epoch,
                )
            else:
                self._epochs_without_improvement += 1

            logger.info(
                "%s | epoch %d/%d | train_loss=%.4f | val_acc=%.4f val_macro_f1=%.4f "
                "val_auroc=%.4f val_ece=%.4f | best_f1=%.4f (ep %d) | lr=%.2e | %.1f min elapsed",
                self.exp_id,
                epoch,
                epochs,
                epoch_metrics.get("train_loss", float("nan")),
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
                val_metrics["macro_auroc"],
                val_metrics["ece"],
                self._best_val_f1,
                self._best_epoch,
                self._current_lr(),
                self.time_budget.elapsed_minutes,
            )

            self.manager.save(
                self._build_payload(epoch, val_metrics), self._global_step, kind="last", epoch=epoch
            )
            self.manager.enforce_quota()

            epoch += 1
            self._epochs_seen = epoch

            if self.time_budget.should_stop():
                stop_reason = self.time_budget.stop_reason
                status = "interrupted"
                logger.warning(
                    "%s | TIME BOX REACHED (%s) after epoch %d. Saving last.pt and exiting "
                    "cleanly so the next Kaggle session can resume.",
                    self.exp_id,
                    stop_reason,
                    epoch - 1,
                )
                break

            if (
                self.early_stopping_patience > 0
                and self._epochs_without_improvement >= self.early_stopping_patience
            ):
                stop_reason = STOP_EARLY_STOP
                logger.info(
                    "%s | early stopping after %d epochs without val Macro-F1 improvement",
                    self.exp_id,
                    self._epochs_without_improvement,
                )
                break

        if stop_reason == STOP_NONE:
            stop_reason = STOP_MAX_EPOCHS

        return self._finalise(
            status=status, stop_reason=stop_reason, start_wall=start_wall, val_loader=val_loader
        )

    # ------------------------------------------------------------------
    def _current_lr(self) -> float:
        if not self.optimizer:
            return float("nan")
        return float(self.optimizer.param_groups[0]["lr"])

    def _train_one_epoch(
        self,
        epoch: int,
        train_loader: Any,
        feature_data: Optional[Tuple[Any, Any]],
        feature_batch: int,
        criterion: Any,
        steps_per_epoch: int,
    ) -> Dict[str, Any]:
        torch, _ = _torch()
        self.model.train(True)
        running_loss = 0.0
        n_seen = 0
        n_correct = 0
        self.optimizer.zero_grad(set_to_none=True)

        def batches():
            if feature_data is not None:
                return self._iterate_feature_batches(
                    feature_data[0], feature_data[1], feature_batch, shuffle=True
                )
            for batch in train_loader:
                yield batch["image"].to(self.device, non_blocking=True), batch["label"].to(
                    self.device, non_blocking=True
                )

        total_batches = (
            int(math.ceil(feature_data[0].shape[0] / max(1, feature_batch)))
            if feature_data is not None
            else steps_per_epoch
        )
        accum_counter = 0
        truncated = False
        last_step = -1
        for step_in_epoch, (images, labels) in enumerate(batches()):
            if self.time_budget.should_stop():
                # Stop at a batch boundary; the caller skips validation when the
                # epoch was cut short, because a partial epoch's weights are not
                # worth the minutes a validation pass would cost out of the
                # remaining session budget.
                truncated = True
                break
            last_step = step_in_epoch
            with torch.cuda.amp.autocast(enabled=bool(self.amp)):
                logits = self.model(images)
                loss = criterion(logits, labels) / self.grad_accum_steps
            self.scaler.scale(loss).backward()
            accum_counter += 1

            if accum_counter >= self.grad_accum_steps or step_in_epoch == total_batches - 1:
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    try:
                        self.scheduler.step()
                    except TypeError:  # pragma: no cover - epoch-based schedulers
                        pass
                accum_counter = 0
                self._global_step += 1

                if self.manager.should_save_step(self._global_step):
                    self.manager.save(
                        self._build_payload(epoch), self._global_step, kind="rolling", epoch=epoch
                    )
                    self.manager.enforce_quota()
                    if self.time_budget.should_stop():
                        truncated = True
                        break

            batch_size = labels.shape[0]
            running_loss += float(loss.item()) * self.grad_accum_steps * batch_size
            n_seen += batch_size
            n_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())

        metrics = {
            "epoch": int(epoch),
            "train_loss": (running_loss / n_seen) if n_seen else float("nan"),
            "train_accuracy": (n_correct / n_seen) if n_seen else float("nan"),
            "train_samples": int(n_seen),
            "global_step": int(self._global_step),
            "lr": self._current_lr(),
            "elapsed_min": round(self.time_budget.elapsed_minutes, 3),
            "truncated": bool(truncated),
            "batches_done": int(last_step + 1),
            "batches_total": int(total_batches),
        }
        return metrics

    # ------------------------------------------------------------------
    def _finalise(
        self, status: str, stop_reason: str, start_wall: float, val_loader: Any
    ) -> TrainResult:
        """Flush checkpoints, run a closing val pass and build the result record."""
        final_metrics: Dict[str, Any] = {}
        try:
            preds = run_inference(self.model, val_loader, self.device, amp=self.amp)
            final_metrics = preds.metrics(self.class_names)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s | final validation failed: %s", self.exp_id, exc)

        # Always leave a `last.pt` that a future session can pick up, even if the
        # loop never ran a single epoch (e.g. budget already exhausted).
        try:
            payload = self._build_payload(max(self._epochs_seen - 1, 0), final_metrics)
            if not self.manager.best_path.exists() and final_metrics:
                self.manager.save(payload, self._global_step, kind="best", epoch=0)
            self.manager.save(payload, self._global_step, kind="last", epoch=max(self._epochs_seen - 1, 0))
        except Exception as exc:  # noqa: BLE001
            logger.error("%s | could not flush final checkpoint: %s", self.exp_id, exc)

        report = self.manager.write_report()
        self._wall_seconds_this_session = time.time() - start_wall

        result = TrainResult(
            exp_id=self.exp_id,
            status=status,
            stop_reason=stop_reason,
            epochs_completed=int(self._epochs_seen),
            global_step=int(self._global_step),
            best_val_macro_f1=float(self._best_val_f1) if self._best_val_f1 > -1e8 else float("nan"),
            best_epoch=int(self._best_epoch),
            trained_seconds=float(self._wall_seconds_this_session),
            train_history=list(self._train_history),
            val_history=list(self._val_history),
            final_val_metrics=final_metrics,
            best_path=str(self.manager.best_path) if self.manager.best_path.exists() else None,
            last_path=str(self.manager.last_path) if self.manager.last_path.exists() else None,
            resume_source=self._resume_source,
            checkpoint_report=report,
            extra={
                "time_budget": self.time_budget.as_dict(),
                "effective_batch_size": self.effective_batch_size,
                "device": str(self.device),
                "reference_provenance": self.reference_provenance,
            },
        )
        logger.info(
            "%s | FINISHED status=%s stop=%s epochs=%d best_val_f1=%.4f (epoch %d) "
            "wall=%.1f min",
            self.exp_id,
            result.status,
            result.stop_reason,
            result.epochs_completed,
            result.best_val_macro_f1,
            result.best_epoch,
            result.trained_seconds / 60.0,
        )
        return result

    # ------------------------------------------------------------------
    def load_best_weights(self) -> bool:
        """Load ``best.pt`` (falls back to ``last.pt``) for the post-training audit."""
        for path in (self.manager.best_path, self.manager.last_path):
            if path.exists():
                payload = load_checkpoint(path, map_location=self.device)
                self.model.load_state_dict(payload["model"], strict=False)
                logger.info(
                    "%s | loaded %s for evaluation (val_macro_f1=%s, epoch=%s)",
                    self.exp_id,
                    path.name,
                    payload.get("metrics", {}).get("macro_f1"),
                    payload.get("epoch"),
                )
                return True
        logger.error("%s | no checkpoint found to evaluate", self.exp_id)
        return False


def _noop_oom_handler(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
    return None


def build_loader_kwargs_for_retry(cfg: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
    """Copy of the config with a smaller batch size, for the OOM retry path."""
    import copy

    retry_cfg = copy.deepcopy(cfg)
    retry_cfg.setdefault("data", {})["batch_size"] = int(batch_size)
    accum = int((cfg.get("data", {}) or {}).get("batch_size", batch_size))
    original_accum = max(1, int((cfg.get("train", {}) or {}).get("grad_accum_steps", 1)))
    effective = accum * original_accum
    retry_cfg.setdefault("train", {})["grad_accum_steps"] = max(
        1, int(round(effective / max(1, int(batch_size))))
    )
    return retry_cfg
