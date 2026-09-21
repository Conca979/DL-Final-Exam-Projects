"""Size-bounded checkpoint management for Kaggle's 20 GB ``/kaggle/working`` cap.

Kaggle kills a session if ``/kaggle/working`` exceeds 20 GB, and that includes
checkpoints.  A 13-cell ablation at ~100-350 MB per checkpoint will breach the
cap long before the matrix is finished, so every write goes through
:class:`CheckpointManager`, which enforces:

* **atomic writes** -- save to ``*.tmp`` then ``os.replace``, so a session killed
  mid-write never leaves a corrupt checkpoint behind;
* **rolling retention** -- at most ``max_rolling`` non-essential checkpoints
  (``step_*.pt``) are kept, oldest pruned first, *unless* a checkpoint is one of
  the ``best_v{epoch}`` snapshots;
* **best snapshots** -- at most ``keep_best`` ``best_v*.pt`` files are retained
  (best-by-val-F1 ordering), plus the evergreen ``best.pt``;
* **``last.pt``** -- always kept, because it carries the optimizer/scheduler/RNG
  state that makes the next Kaggle session resume in seconds instead of redoing
  10 hours;
* **quota guard** -- :meth:`CheckpointManager.enforce_quota` measures the run
  directory and prunes the oldest rolling checkpoints until it fits
  ``run_quota_gb``, and reports the total ever seen for the log.

``--keep-last N`` on the CLI maps to ``max_rolling=N``; ``--save-every K`` maps to
``save_every_steps=K``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "CheckpointManager",
    "CheckpointInfo",
    "find_latest_checkpoint",
    "save_checkpoint",
    "load_checkpoint",
    "gigabytes",
]

_STEP_RE = re.compile(r"step_(\d+)\.pt$")
_BEST_RE = re.compile(r"best_v(\d+)\.pt$")


def gigabytes(num_bytes: float) -> float:
    return float(num_bytes) / (1024.0**3)


def _atomic_torch_save(payload: Any, path: Path) -> Path:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def save_checkpoint(payload: Dict[str, Any], path: str | Path) -> Path:
    """Atomically persist a checkpoint dict (torch.save compatible)."""
    return _atomic_torch_save(payload, Path(path))


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Tolerates both the ``{"model": ...}`` envelope used here and a bare
    ``state_dict`` (older/manual checkpoints), so resuming never hard-fails on a
    format mismatch.
    """
    import torch

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        return payload
    return {"model": payload, "epoch": -1, "global_step": -1, "format": "bare_state_dict"}


@dataclass
class CheckpointInfo:
    path: Path
    kind: str  # "best" | "rolling" | "last"
    step: int = -1
    epoch: int = -1
    metric: float = float("nan")
    mtime: float = 0.0
    size_bytes: int = 0

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024.0**2)


@dataclass
class CheckpointManager:
    """Owns one run directory and guarantees it stays inside its size budget."""

    run_dir: Path
    save_every_steps: int = 500
    max_rolling: int = 3
    keep_best: int = 3
    run_quota_gb: float = 3.0
    total_kept_bytes: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_steps = max(1, int(self.save_every_steps))
        self.max_rolling = max(1, int(self.max_rolling))
        self.keep_best = max(1, int(self.keep_best))

    # ------------------------------------------------------------------
    @property
    def last_path(self) -> Path:
        return self.run_dir / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.run_dir / "best.pt"

    def step_path(self, global_step: int) -> Path:
        return self.run_dir / f"step_{int(global_step):08d}.pt"

    def best_snapshot_path(self, epoch: int) -> Path:
        return self.run_dir / f"best_v{int(epoch):04d}.pt"

    def should_save_step(self, global_step: int) -> bool:
        return global_step > 0 and global_step % self.save_every_steps == 0

    # ------------------------------------------------------------------
    def save(
        self,
        payload: Dict[str, Any],
        global_step: int,
        kind: str = "rolling",
        epoch: int = -1,
    ) -> Path:
        """Write ``payload`` as ``kind`` and immediately prune."""
        if kind == "rolling":
            target = self.step_path(global_step)
        elif kind == "best_snapshot":
            target = self.best_snapshot_path(epoch)
        elif kind == "last":
            target = self.last_path
        elif kind == "best":
            target = self.best_path
        else:
            raise ValueError(f"Unknown checkpoint kind '{kind}'")

        _atomic_torch_save(payload, target)
        size = target.stat().st_size
        self.total_kept_bytes += size
        self.history.append(
            {
                "path": str(target),
                "kind": kind,
                "global_step": int(global_step),
                "epoch": int(epoch),
                "size_mb": round(size / (1024.0**2), 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        logger.info(
            "checkpoint saved: %s (%.1f MB, step=%d, epoch=%d)",
            target.name,
            size / (1024.0**2),
            global_step,
            epoch,
        )
        self.prune()
        return target

    # ------------------------------------------------------------------
    def scan(self) -> List[CheckpointInfo]:
        infos: List[CheckpointInfo] = []
        for path in sorted(self.run_dir.glob("*.pt")):
            name = path.name
            stat = path.stat()
            info = CheckpointInfo(
                path=path,
                kind="rolling",
                mtime=stat.st_mtime,
                size_bytes=stat.st_size,
            )
            if name == "best.pt":
                info.kind = "best"
            elif name == "last.pt":
                info.kind = "last"
            elif (match := _STEP_RE.search(name)):
                info.step = int(match.group(1))
            elif (match := _BEST_RE.search(name)):
                info.kind = "best"
                info.epoch = int(match.group(1))
                info.metric = self._read_metric(path)
            infos.append(info)
        return infos

    @staticmethod
    def _read_metric(path: Path) -> float:
        try:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=False)
            metrics = payload.get("metrics", {}) or {}
            return float(metrics.get("val_macro_f1", float("nan")))
        except Exception:  # noqa: BLE001 - metric read is best-effort only
            return float("nan")

    def prune(self) -> List[Path]:
        """Delete checkpoints beyond the retention policy; return removed paths."""
        removed: List[Path] = []
        infos = self.scan()

        rolling = sorted(
            [i for i in infos if i.kind == "rolling"], key=lambda i: i.step
        )
        if len(rolling) > self.max_rolling:
            for info in rolling[: len(rolling) - self.max_rolling]:
                removed.append(self._remove(info.path))

        bests = sorted(
            [i for i in infos if i.kind == "best" and i.path.name != "best.pt"],
            key=lambda i: (i.metric if i.metric == i.metric else -1.0),  # NaN-safe
            reverse=True,
        )
        if len(bests) > self.keep_best:
            for info in bests[self.keep_best :]:
                removed.append(self._remove(info.path))

        if removed:
            logger.info(
                "checkpoint pruning removed %d file(s): %s",
                len(removed),
                [p.name for p in removed],
            )
        return removed

    def enforce_quota(self, quota_gb: Optional[float] = None) -> Dict[str, Any]:
        """Prune rolling checkpoints until the run directory fits ``quota_gb``."""
        quota = float(quota_gb if quota_gb is not None else self.run_quota_gb)
        removed: List[Path] = []
        while True:
            infos = self.scan()
            total = sum(i.size_bytes for i in infos)
            if gigabytes(total) <= quota:
                break
            rolling = sorted(
                [i for i in infos if i.kind == "rolling"], key=lambda i: i.step
            )
            if not rolling:
                logger.error(
                    "Run directory %.2f GB exceeds the %.2f GB quota but only "
                    "protected checkpoints remain (best/last). Raise Kaggle's "
                    "effective quota or reduce model size.",
                    gigabytes(total),
                    quota,
                )
                break
            oldest = rolling[0]
            removed.append(self._remove(oldest.path))
        infos = self.scan()
        total = sum(i.size_bytes for i in infos)
        report = {
            "run_dir": str(self.run_dir),
            "quota_gb": quota,
            "current_gb": round(gigabytes(total), 3),
            "n_checkpoints": len(infos),
            "files": [
                {
                    "name": i.path.name,
                    "kind": i.kind,
                    "size_mb": round(i.size_bytes / (1024.0**2), 2),
                    "step": i.step,
                    "epoch": i.epoch,
                }
                for i in infos
            ],
            "removed_now": [p.name for p in removed],
        }
        return report

    def _remove(self, path: Path) -> Path:
        try:
            path.unlink()
        except FileNotFoundError:  # pragma: no cover
            pass
        return path

    # ------------------------------------------------------------------
    def write_report(self, filename: str = "checkpoint_report.json") -> Path:
        report = self.enforce_quota()
        report["save_every_steps"] = self.save_every_steps
        report["max_rolling_keep_last"] = self.max_rolling
        report["keep_best"] = self.keep_best
        report["bytes_written_during_session"] = self.total_kept_bytes
        report["gb_written_during_session"] = round(
            gigabytes(self.total_kept_bytes), 3
        )
        target = self.run_dir / filename
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return target


def find_latest_checkpoint(search_dirs: Sequence[str | Path]) -> Optional[Path]:
    """Newest ``last.pt`` (else ``step_*.pt``) across mounted checkpoint datasets.

    Kaggle sessions are stateless: the previous session's ``/kaggle/working`` is
    gone, and whatever survives does so only because the human re-uploaded it as a
    dataset.  This helper is what makes "resume" work without any bookkeeping:
    the newest file wins, and the caller parses ``epoch``/``global_step`` straight
    out of the payload.
    """
    candidates: List[Path] = []
    for directory in search_dirs:
        directory = Path(directory)
        if not directory.exists():
            continue
        candidates.extend(directory.rglob("last.pt"))
        if not candidates:
            candidates.extend(directory.rglob("step_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def total_size_gb(paths: Iterable[str | Path]) -> float:
    total = 0
    for path in paths:
        path = Path(path)
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            total += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return gigabytes(total)


def free_space_gb(path: str | Path = "/kaggle/working") -> Optional[float]:
    try:
        return gigabytes(shutil.disk_usage(str(path)).free)
    except Exception:  # noqa: BLE001
        return None
