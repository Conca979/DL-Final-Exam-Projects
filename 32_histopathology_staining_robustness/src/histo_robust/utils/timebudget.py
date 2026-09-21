"""Wall-clock budget guards for the Kaggle 12-hour hard kill.

This is the core of the "Short-Session Recipe".  Kaggle terminates a background
("Save & Run All") session at 12:00:00 with no warning and no cleanup, which
means:

* nothing after the kill point ever runs -- no final checkpoint, no metrics file,
  no pruned output directory;
* a queue-induced start delay means the *session* clock and the *script* clock
  disagree, so the guard must be measured on the script's own monotonic clock
  **and** against the session start when Kaggle exposes it.

:class:`TimeBudget` therefore tracks two independent limits and reports
``should_stop()`` when either is exhausted:

1. ``max_seconds``   -- per-run training budget (default 630 min = 10.5 h),
2. ``session_deadline`` -- absolute epoch seconds derived from
   ``KAGGLE_SESSION_START`` (or an explicit ``--session-start``), used to reserve
   ``reserve_minutes`` for checkpoint flush, OOD evaluation and artifact zipping.

``TimeBudget`` never kills the process; it only ever asks.  The trainer is
responsible for breaking at a batch boundary, saving ``last.pt`` (+ optimizer,
scheduler and RNG state) and returning a normal exit code so Kaggle marks the
version as successful instead of failed.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "TimeBudget",
    "STOP_NONE",
    "STOP_RUN_BUDGET",
    "STOP_SESSION_RESERVE",
    "STOP_ORCHESTRATOR_BUDGET",
    "STOP_EARLY_STOP",
    "STOP_MAX_EPOCHS",
    "STOP_OOM",
]

STOP_NONE = "none"
STOP_RUN_BUDGET = "run_time_budget"
STOP_SESSION_RESERVE = "session_reserve_reached"
STOP_ORCHESTRATOR_BUDGET = "orchestrator_budget_exhausted"
STOP_EARLY_STOP = "early_stopping"
STOP_MAX_EPOCHS = "max_epochs"
STOP_OOM = "cuda_oom"


def _session_start_epoch(session_start: Optional[str] = None) -> Optional[float]:
    """Best-effort session start time.

    Kaggle does not publish a documented env var for this, so we accept several
    spellings and a CLI override.  Returning ``None`` simply disables the
    session-reserve guard (the run budget still protects the session).
    """
    if session_start:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%s"):
            try:
                return datetime.strptime(session_start, fmt).replace(
                    tzinfo=timezone.utc
                ).timestamp()
            except ValueError:
                continue
        logger.warning("Could not parse --session-start '%s'; ignoring it", session_start)

    for key in ("KAGGLE_SESSION_START", "KAGGLE_START_TIME", "SESSION_START_TIME"):
        raw = os.environ.get(key)
        if not raw:
            continue
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%s"):
            try:
                return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
        logger.warning("Unparsable %s='%s'", key, raw)
    return None


@dataclass
class TimeBudget:
    """Tracks a training-time budget and an optional absolute session deadline."""

    max_minutes: float = 630.0
    reserve_minutes: float = 20.0
    session_deadline_epoch: Optional[float] = None
    label: str = "run"
    started_epoch: float = field(default_factory=time.time)
    consumed_before: float = 0.0
    stop_reason: str = STOP_NONE

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        label: str = "run",
        max_minutes: Optional[float] = None,
        reserve_minutes: Optional[float] = None,
        session_start: Optional[str] = None,
        consumed_before: float = 0.0,
    ) -> "TimeBudget":
        time_cfg = dict(cfg.get("time", {}) or {})
        session_cfg = dict(cfg.get("session", {}) or {})
        max_minutes = float(
            max_minutes if max_minutes is not None else time_cfg.get("max_train_minutes", 630)
        )
        reserve_minutes = float(
            reserve_minutes
            if reserve_minutes is not None
            else session_cfg.get("reserve_minutes", 20)
        )
        session_minutes = float(session_cfg.get("session_minutes", 720.0))
        start = _session_start_epoch(session_start or session_cfg.get("session_start"))
        deadline = (start + session_minutes * 60.0) if start else None
        return cls(
            max_minutes=max_minutes,
            reserve_minutes=reserve_minutes,
            session_deadline_epoch=deadline,
            label=label,
            consumed_before=float(consumed_before),
        )

    # ------------------------------------------------------------------
    @property
    def elapsed_seconds(self) -> float:
        return (time.time() - self.started_epoch) + self.consumed_before

    @property
    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds / 60.0

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_minutes * 60.0 - self.elapsed_seconds)

    @property
    def remaining_minutes(self) -> float:
        return self.remaining_seconds / 60.0

    @property
    def session_remaining_seconds(self) -> Optional[float]:
        if self.session_deadline_epoch is None:
            return None
        usable = self.session_deadline_epoch - self.reserve_minutes * 60.0
        return max(0.0, usable - time.time())

    def should_stop(self) -> bool:
        """True when training must break cleanly at the next safe point."""
        if self.stop_reason != STOP_NONE:
            return True
        if self.remaining_seconds <= 0:
            self.stop_reason = STOP_RUN_BUDGET
            return True
        session_left = self.session_remaining_seconds
        if session_left is not None and session_left <= 0:
            self.stop_reason = STOP_SESSION_RESERVE
            return True
        return False

    def reserve(self, reason: str) -> None:
        """Record the FIRST stop reason (later reasons are not overwritten)."""
        if self.stop_reason == STOP_NONE:
            self.stop_reason = reason

    def seconds_for(self, fraction: float) -> float:
        """A slice of the remaining budget, e.g. ``fraction=0.5`` for a half-run."""
        return max(0.0, self.remaining_seconds * float(fraction))

    # ------------------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        session_left = self.session_remaining_seconds
        return {
            "label": self.label,
            "max_minutes": self.max_minutes,
            "reserve_minutes": self.reserve_minutes,
            "elapsed_minutes": round(self.elapsed_minutes, 3),
            "remaining_minutes": round(self.remaining_minutes, 3),
            "session_remaining_minutes": (
                round(session_left / 60.0, 3) if session_left is not None else None
            ),
            "session_deadline_utc": (
                datetime.fromtimestamp(self.session_deadline_epoch, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if self.session_deadline_epoch
                else None
            ),
            "stop_reason": self.stop_reason,
            "stopped": self.should_stop(),
        }

    def log(self, logger_: Optional[logging.Logger] = None) -> None:
        log = logger_ or logger
        info = self.as_dict()
        log.info(
            "[TimeBudget:%s] elapsed=%.1f min | remaining=%.1f min | session_left=%s min | stop=%s",
            info["label"],
            info["elapsed_minutes"],
            info["remaining_minutes"],
            info["session_remaining_minutes"],
            info["stop_reason"],
        )


class SessionClock:
    """Persisted wall-clock accounting shared by every experiment in a session.

    ``time.json`` lives next to the checkpoints, so a resumed session continues
    the accounting instead of pretending the previous 10 hours never happened.
    """

    def __init__(self, path: str | Path, session_minutes: float = 720.0) -> None:
        self.path = Path(path)
        self.session_minutes = float(session_minutes)
        self.data: Dict[str, Any] = {
            "session_minutes": self.session_minutes,
            "experiments": {},
            "session_started": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "total_trained_seconds": 0.0,
        }
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    self.data.update(payload)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read %s (%s); starting fresh", self.path, exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def consumed_seconds(self, exp_id: str) -> float:
        entry = self.data.get("experiments", {}).get(exp_id, {})
        return float(entry.get("trained_seconds", 0.0))

    def record(self, exp_id: str, seconds: float, extra: Optional[Dict[str, Any]] = None) -> None:
        experiments = self.data.setdefault("experiments", {})
        entry = experiments.setdefault(exp_id, {"trained_seconds": 0.0})
        entry["trained_seconds"] = float(entry.get("trained_seconds", 0.0)) + float(seconds)
        entry["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        if extra:
            entry.update(extra)
        self.data["total_trained_seconds"] = float(
            self.data.get("total_trained_seconds", 0.0)
        ) + float(seconds)
        self.save()

    def summary(self) -> Dict[str, Any]:
        total = float(self.data.get("total_trained_seconds", 0.0))
        return {
            "session_minutes": self.session_minutes,
            "total_trained_minutes": round(total / 60.0, 2),
            "total_trained_hours": round(total / 3600.0, 3),
            "n_experiments_touched": len(self.data.get("experiments", {})),
            "per_experiment_minutes": {
                k: round(float(v.get("trained_seconds", 0.0)) / 60.0, 2)
                for k, v in (self.data.get("experiments", {}) or {}).items()
            },
        }

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.data)

    def estimated_exit_time(self) -> str:
        remaining = max(0.0, self.session_minutes * 60.0 - float(
            self.data.get("total_trained_seconds", 0.0)
        ))
        return (
            datetime.now(timezone.utc) + timedelta(seconds=remaining)
        ).strftime("%Y-%m-%d %H:%M:%S UTC")


class BudgetAllocator:
    """Splits the remaining session time across the experiments still to run.

    Two constraints are honoured at once:

    * never plan past ``deadline_epoch - reserve`` (protects the 12-hour kill),
    * never spend more than ``per_exp_minutes`` on one cell (protects the matrix
      from one slow backbone eating the whole night).

    Experiments are executed in ``docs/PLAN.md`` order, so the baseline cells that
    every other comparison depends on are produced first even if the session ends
    early.
    """

    def __init__(
        self,
        deadline_epoch: Optional[float],
        reserve_minutes: float = 20.0,
        per_exp_minutes: float = 90.0,
        min_minutes: float = 5.0,
    ) -> None:
        self.deadline_epoch = deadline_epoch
        self.reserve_minutes = float(reserve_minutes)
        self.per_exp_minutes = float(per_exp_minutes)
        self.min_minutes = float(min_minutes)
        self.allocations: List[Dict[str, Any]] = []

    def usable_seconds(self) -> Optional[float]:
        if self.deadline_epoch is None:
            return None
        return max(0.0, self.deadline_epoch - self.reserve_minutes * 60.0 - time.time())

    def allocate(self, pending: List[str]) -> Dict[str, float]:
        """Return ``{exp_id: minutes}`` for the pending experiments."""
        usable = self.usable_seconds()
        if usable is None:
            return {exp: self.per_exp_minutes for exp in pending}
        if not pending:
            return {}
        fair_share = (usable / 60.0) / len(pending)
        per_exp = min(self.per_exp_minutes, max(self.min_minutes, fair_share))
        allocation = {exp: per_exp for exp in pending}
        self.allocations.append(
            {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                "usable_minutes": round(usable / 60.0, 2),
                "pending": list(pending),
                "per_experiment_minutes": round(per_exp, 2),
            }
        )
        return allocation
