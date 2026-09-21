"""Config loading, dotted-path overrides and deterministic config hashing.

A config is a plain dict loaded from YAML.  Experiments may declare
``_base_: <path>`` (relative to the config file or to the repo root) to inherit
from another config; nested dicts are merged recursively, everything else is
replaced.  ``_base_`` keys are stripped from the final result.

CLI overrides use dotted paths with automatic type coercion, e.g.::

    --set train.epochs=6
    --set normalization.name=macenko
    --set augmentation.policy=aug_combined
    --set model.freeze_backbone=true
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

__all__ = [
    "load_config",
    "apply_overrides",
    "get_by_path",
    "set_by_path",
    "config_hash",
    "config_to_json",
    "deep_update",
]


def _resolve_base(base_value: str, config_path: Path, repo_root: Optional[Path]) -> Path:
    base = Path(base_value)
    if base.is_absolute() and base.exists():
        return base
    candidates = [config_path.parent / base]
    if repo_root is not None:
        candidates.append(repo_root / base)
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(
        f"Could not resolve _base_='{base_value}' from {config_path} "
        f"(repo_root={repo_root})"
    )


def deep_update(target: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``src`` into ``target`` (in place) and return it."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def load_config(
    path: str | Path,
    overrides: Optional[Iterable[str]] = None,
    repo_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Load a YAML config, resolve ``_base_`` inheritance and apply overrides."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    root = Path(repo_root).resolve() if repo_root is not None else None

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Config {config_path} must contain a YAML mapping at top level")

    base_value = raw.pop("_base_", None)
    if base_value is not None:
        base_path = _resolve_base(str(base_value), config_path, root)
        base_cfg = load_config(base_path, overrides=None, repo_root=root)
        cfg = deep_update(base_cfg, raw)
    else:
        cfg = raw

    cfg = copy.deepcopy(cfg)
    cfg.setdefault("_config_path", str(config_path))
    cfg.setdefault("_config_name", config_path.stem)

    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply ``key.path=value`` overrides with automatic YAML type coercion."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Malformed override '{item}' (expected key.path=value)")
        key, _, raw_value = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"Malformed override '{item}' (empty key)")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        set_by_path(cfg, key, value)
    return cfg


def _walk_keys(node: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_keys(value, child)
    else:
        yield prefix


def set_by_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``cfg`` at a dotted path; dotted indices into lists are supported."""
    parts = [p for p in dotted.split(".") if p != ""]
    if not parts:
        raise ValueError("Empty override key")
    node: Any = cfg
    for part in parts[:-1]:
        if isinstance(node, list):
            node = node[int(part)]
        else:
            if part not in node or not isinstance(node[part], (dict, list)):
                node[part] = {}
            node = node[part]
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def get_by_path(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read ``cfg`` at a dotted path, returning ``default`` when missing."""
    node: Any = cfg
    for part in dotted.split("."):
        if isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                return default
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


def config_to_json(cfg: Dict[str, Any], indent: int = 2) -> str:
    """JSON dump that tolerates path objects and other non-JSON scalars."""
    return json.dumps(cfg, indent=indent, sort_keys=True, default=str)


def config_hash(cfg: Dict[str, Any], length: int = 12) -> str:
    """Stable short hash of the *semantic* config (underscore keys ignored)."""
    semantic = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    payload = json.dumps(semantic, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]
