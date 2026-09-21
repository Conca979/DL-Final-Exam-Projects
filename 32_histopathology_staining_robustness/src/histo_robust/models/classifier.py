"""Task head on top of a frozen or fine-tuned backbone (Axis E).

The classifier is intentionally thin -- ``Dropout -> Linear`` on the pooled
feature -- because every experiment in ``docs/PLAN.md`` reuses exactly the same head.
That way a difference in OOD robustness can only come from the representation
(Axis E) or from what the representation was trained to ignore (Axes B/C/D),
never from an architectural confound in the head.

``freeze_backbone=True`` enables the linear-probe fallback required by ``docs/PLAN.md``
risk 5 (Phikon on a 16 GB T4/P100).  In that mode ``scripts/train.py`` switches to
a two-phase recipe: cache the backbone features once, then fit the head on the
cached tensors, which removes the 86M-parameter backward pass entirely.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = ["StainRobustClassifier", "build_classifier"]


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


class StainRobustClassifier:
    """Backbone + linear head, framework-agnostic wrapper around ``nn.Module``.

    Using a plain class (with an inner ``nn.Module``) keeps ``torch`` importable
    lazily, which matters because the preprocessing/eval-only scripts should not
    pay the torch import cost on a CPU-only box.
    """

    def __init__(
        self,
        backbone: Any,
        feature_dim: int,
        num_classes: int = 9,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        backbone_family: str = "timm",
        backbone_name: str = "unknown",
    ) -> None:
        torch, nn = _torch()

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.freeze_backbone = bool(freeze_backbone)
        self.backbone_family = backbone_family
        self.backbone_name = backbone_name

        outer = self

        class _Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = backbone
                self.dropout = nn.Dropout(p=float(dropout))
                self.fc = nn.Linear(outer.feature_dim, outer.num_classes)

            def forward(self, x):
                features = self.backbone(x)
                if features.ndim > 2:
                    features = features.mean(dim=tuple(range(2, features.ndim)))
                return self.fc(self.dropout(features))

            def forward_features(self, x):
                return self.backbone(x)

        self.module = _Head()
        if self.freeze_backbone:
            self.freeze_backbone_parameters()

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------
    def freeze_backbone_parameters(self) -> None:
        for param in self.module.backbone.parameters():
            param.requires_grad = False
        for param in self.module.fc.parameters():
            param.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.module.parameters() if p.requires_grad]

    def parameter_groups(self, lr: float, head_lr_mult: float = 1.0, weight_decay: float = 0.0):
        """Parameter groups with a separate LR for the pretrained backbone."""
        head_params = list(self.module.fc.parameters())
        backbone_params = [p for p in self.module.backbone.parameters() if p.requires_grad]
        groups = [
            {"params": head_params, "lr": float(lr) * float(head_lr_mult), "name": "head"}
        ]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": float(lr), "name": "backbone"})
        for group in groups:
            group["weight_decay"] = float(weight_decay)
        return groups

    # ------------------------------------------------------------------
    # nn.Module passthrough
    # ------------------------------------------------------------------
    def forward(self, x):
        return self.module(x)

    def __call__(self, x):
        return self.module(x)

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self.module.train(mode)
        if self.freeze_backbone:
            self.module.backbone.eval()
        return self

    def eval(self):
        self.module.eval()
        return self

    @property
    def training(self) -> bool:
        return self.module.training

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.module.load_state_dict(state_dict, strict=strict)

    def zero_grad(self, set_to_none: bool = True):
        return self.module.zero_grad(set_to_none=set_to_none)

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)

    def n_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.module.parameters())
        trainable = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        return {"total": int(total), "trainable": int(trainable)}

    def describe(self) -> Dict[str, Any]:
        counts = self.n_parameters()
        return {
            "backbone_family": self.backbone_family,
            "backbone_name": self.backbone_name,
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "freeze_backbone": self.freeze_backbone,
            "parameters_total": counts["total"],
            "parameters_trainable": counts["trainable"],
        }


def build_classifier(
    cfg: Dict[str, Any],
    pretrained: bool = True,
    weights_dir: Optional[str | Path] = None,
) -> StainRobustClassifier:
    """Build the model described by ``cfg['model']``.

    Raises a readable error for unknown backbones instead of silently defaulting,
    so a typo can never produce a plausible-but-wrong ablation row.
    """
    from .backbones import BACKBONE_REGISTRY, BACKBONE_CHOICES, load_hf_backbone, load_timm_backbone

    model_cfg = dict(cfg.get("model", {}) or {})
    data_cfg = dict(cfg.get("data", {}) or {})
    key = str(model_cfg.get("backbone", "resnet50")).lower()
    if key not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{key}'. Expected one of {list(BACKBONE_CHOICES)}"
        )
    spec = BACKBONE_REGISTRY[key]
    image_size = int(data_cfg.get("image_size", 224))

    if spec["family"] == "timm":
        backbone, feature_dim = load_timm_backbone(spec, pretrained, image_size, weights_dir)
    else:
        backbone, feature_dim = load_hf_backbone(spec, pretrained, image_size, weights_dir)

    return StainRobustClassifier(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=int(data_cfg.get("num_classes", 9)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
        backbone_family=spec["family"],
        backbone_name=spec["model_name"],
    )
