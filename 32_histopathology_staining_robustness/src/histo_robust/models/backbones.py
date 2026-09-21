"""Backbone zoo for Axis E: ResNet-50, ConvNeXt-Tiny and Phikon.

| config key        | source                                   | feature dim |
|-------------------|------------------------------------------|-------------|
| ``resnet50``      | ``timm`` ``resnet50.a1_in1k``            | 2048        |
| ``convnext_tiny`` | ``timm`` ``convnext_tiny.fb_in1k``       | 768         |
| ``phikon``        | HF ``transformers`` ``owkin/phikon``     | 768         |

Weight acquisition is Internet-tolerant, which is the main Kaggle failure mode
for this project:

* If Internet is ON, weights download on first use and are cached in
  ``~/.cache`` for the rest of the session.
* If Internet is OFF (or the download is flaky) the loader looks for a locally
  mounted copy first.  ``docs/PLAN.md`` requires an offline path for the foundation
  model, so the search covers the standard Kaggle "model dataset" layouts:
  ``<dataset>/phikon/``, ``<dataset>/models--owkin--phikon/`` and a plain
  ``pytorch_model.bin``/``model.safetensors`` next to a ``config.json``.
  ``docs/kaggle_guide.md`` explains how to build that dataset in five minutes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "BackboneSpec",
    "BACKBONE_REGISTRY",
    "BACKBONE_CHOICES",
    "available_backbones",
    "load_timm_backbone",
    "load_hf_backbone",
    "resolve_local_weights",
    "estimate_backbone_vram_note",
]

#: Registry mirroring the "Backbone" column of the ablation matrix in docs/PLAN.md.
BACKBONE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "resnet50": {
        "family": "timm",
        "model_name": "resnet50.a1_in1k",
        "feature_dim": 2048,
        "params_millions": 25.6,
        "display_name": "ResNet-50 (ImageNet-1k, timm a1_in1k)",
        "default_batch_size": 64,
        "default_accum_steps": 1,
    },
    "convnext_tiny": {
        "family": "timm",
        "model_name": "convnext_tiny.fb_in1k",
        "feature_dim": 768,
        "params_millions": 28.6,
        "display_name": "ConvNeXt-Tiny (ImageNet-1k, timm fb_in1k)",
        "default_batch_size": 64,
        "default_accum_steps": 1,
    },
    "phikon": {
        "family": "transformers",
        "model_name": "owkin/phikon",
        "feature_dim": 768,
        "params_millions": 86.0,
        "display_name": "Phikon ViT-B/16 (histopathology SSL, TCGA iBOT)",
        "default_batch_size": 32,
        "default_accum_steps": 2,
    },
}

BACKBONE_CHOICES: Tuple[str, ...] = tuple(BACKBONE_REGISTRY)


class BackboneSpec(dict):
    """Dict subclass with attribute access, for readable call sites."""

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - trivial
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def available_backbones() -> List[str]:
    return list(BACKBONE_CHOICES)


def resolve_local_weights(
    model_name: str, search_dirs: Optional[Sequence[str | Path]] = None
) -> Optional[Path]:
    """Find an offline copy of ``model_name`` under the given directories.

    Recognised layouts::

        <dir>/phikon/config.json + model.safetensors
        <dir>/models--owkin--phikon/snapshots/<rev>/config.json
        <dir>/<model_name>/config.json
        <dir>/*/config.json                      (single-model mount)
    """
    if not search_dirs:
        return None
    slug = model_name.split("/")[-1]
    hf_slug = "models--" + model_name.replace("/", "--")

    for base in search_dirs:
        base = Path(base)
        if not base.exists():
            continue
        candidates: List[Path] = [base / model_name, base / slug]
        hf_root = base / hf_slug
        if hf_root.exists():
            snapshots = sorted((hf_root / "snapshots").glob("*"))
            candidates.extend(snapshots)
        # single-model dataset mount: any directory holding a config.json
        candidates.extend(sorted(p.parent for p in base.glob("*/config.json")))
        candidates.extend(sorted(p.parent for p in base.glob("*/*/config.json")))
        for candidate in candidates:
            if (candidate / "config.json").exists():
                return candidate
    return None


def load_timm_backbone(
    spec: Dict[str, Any],
    pretrained: bool,
    image_size: int,
    weights_dir: Optional[str | Path] = None,
) -> Tuple[Any, int]:
    """Create a timm model with ``num_classes=0`` (feature extractor)."""
    import timm

    model_name = spec["model_name"]
    local = resolve_local_weights(model_name, [weights_dir] if weights_dir else None)

    kwargs: Dict[str, Any] = {
        "pretrained": bool(pretrained),
        "num_classes": 0,
    }
    if pretrained and local is not None:
        weights_file = local / "pytorch_model.bin"
        if not weights_file.exists():
            weights_file = local / "model.safetensors"
        if weights_file.exists():
            kwargs = {"pretrained": False, "num_classes": 0, "checkpoint_path": str(weights_file)}
            logger.info("Loading timm weights offline for %s from %s", model_name, weights_file)

    try:
        model = timm.create_model(model_name, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to build timm model '{model_name}' (pretrained={pretrained}). "
            f"If Kaggle Internet is OFF, either enable Internet for the first run "
            f"(weights are then cached in ~/.cache/huggingface) or mount a Kaggle "
            f"dataset containing the weights and set paths.weights_dir to it. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    # Force the positional-embedding / crop resolution to the data resolution.
    try:
        model.set_input_size(img_size=(image_size, image_size))
    except Exception:  # noqa: BLE001 - older timm or no positional embeddings
        logger.debug("set_input_size unsupported for %s; using native size", model_name)

    feature_dim = int(getattr(model, "num_features", spec["feature_dim"]))
    return model, feature_dim


def load_hf_backbone(
    spec: Dict[str, Any],
    pretrained: bool,
    image_size: int,
    weights_dir: Optional[str | Path] = None,
) -> Tuple[Any, int]:
    """Create a Hugging Face vision encoder and return ``(module, feature_dim)``.

    The returned callable must map ``(B, 3, H, W)`` float tensors to ``(B, D)``.
    """
    from transformers import AutoConfig, AutoModel

    model_name = spec["model_name"]
    local = resolve_local_weights(model_name, [weights_dir] if weights_dir else None)
    source = str(local) if local is not None else model_name

    try:
        config = AutoConfig.from_pretrained(source, local_files_only=local is not None)
        # Phikon ships as a plain ViT encoder: make sure we ask for hidden states
        # rather than any pooled output the config might not define.
        if hasattr(config, "add_pooling_layer"):
            config.add_pooling_layer = False
        if hasattr(config, "image_size"):
            config.image_size = int(image_size)
        model = AutoModel.from_pretrained(
            source, config=config, local_files_only=local is not None
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load Hugging Face backbone '{model_name}'. Phikon is "
            f"ungated, so a network failure is the usual cause. Enable Kaggle "
            f"Internet, or mount the model as a dataset and point "
            f"paths.weights_dir at it (see docs/kaggle_guide.md, section 'Offline "
            f"model weights'). Original error: {type(exc).__name__}: {exc}"
        ) from exc

    if not pretrained:
        logger.warning("pretrained=False requested for HF backbone; using random init")

    model.train(True)
    feature_dim = int(getattr(model.config, "hidden_size", spec["feature_dim"]))

    class _HFEncoder:
        """Adapter exposing a single ``forward -> (B, D)`` contract."""

        def __init__(self, module: Any) -> None:
            self.module = module
            self.feature_dim = feature_dim

        def __call__(self, pixel_values: Any) -> Any:
            outputs = self.module(pixel_values=pixel_values)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:  # pragma: no cover - defensive
                hidden = outputs[0]
            return hidden[:, 0]  # CLS token

        def parameters(self):
            return self.module.parameters()

        def named_parameters(self, *args: Any, **kwargs: Any):
            return self.module.named_parameters(*args, **kwargs)

        def train(self, mode: bool = True):
            self.module.train(mode)
            return self

        def eval(self):
            self.module.eval()
            return self

        def to(self, *args: Any, **kwargs: Any):
            self.module.to(*args, **kwargs)
            return self

        def state_dict(self, *args: Any, **kwargs: Any):
            return self.module.state_dict(*args, **kwargs)

        def load_state_dict(self, *args: Any, **kwargs: Any):
            return self.module.load_state_dict(*args, **kwargs)

        def zero_grad(self, *args: Any, **kwargs: Any):
            return self.module.zero_grad(*args, **kwargs)

        def __getattr__(self, item: str) -> Any:
            return getattr(self.module, item)

    return _HFEncoder(model), feature_dim


def estimate_backbone_vram_note(spec: Dict[str, Any], batch_size: int) -> str:
    """Human-readable VRAM hint printed by the pre-flight benchmark."""
    return (
        f"{spec['display_name']}: ~{spec['params_millions']:.1f}M params at "
        f"batch_size={batch_size}. If you hit CUDA OOM, lower data.batch_size "
        f"(and raise train.grad_accum_steps to keep the effective batch at 64), "
        f"or set train.freeze_backbone=true to train only the classification head."
    )
