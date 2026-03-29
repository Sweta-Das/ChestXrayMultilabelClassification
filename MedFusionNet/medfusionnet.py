"""MedFusionNet model helpers for Grad-CAM loading.

This module mirrors the notebook architecture from
`MedFusionNet/medfusionnet.ipynb` so the checkpoint can be loaded from a
regular Python module instead of a notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torchvision.models as models


MEDFUSION_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


def _normalize_label(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").lower().strip()


def _strip_module_prefix(state_dict: dict) -> dict:
    return {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }


def _looks_like_state_dict(candidate: object) -> bool:
    if not isinstance(candidate, dict) or not candidate:
        return False
    return all(isinstance(value, torch.Tensor) for value in candidate.values())


def _build_permutation(
    source_labels: Sequence[str],
    target_labels: Sequence[str],
) -> Optional[torch.Tensor]:
    source_index = {_normalize_label(label): idx for idx, label in enumerate(source_labels)}
    permutation = []
    for label in target_labels:
        idx = source_index.get(_normalize_label(label))
        if idx is None:
            return None
        permutation.append(idx)
    if permutation == list(range(len(permutation))):
        return None
    return torch.tensor(permutation, dtype=torch.long)


class DenseNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # The checkpoint already contains the learned weights, so avoid a
        # network download here.
        densenet = models.densenet121(weights=None)
        self.features = densenet.features

    def forward(self, x):
        return self.features(x)


class SelfAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5)
        attn = self.softmax(attn)
        return torch.bmm(attn, v)


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, out_channels: int = 128):
        super().__init__()
        self.lateral = nn.Conv2d(1024, out_channels, kernel_size=1)

    def forward(self, x):
        return self.lateral(x)


class MedFusionNet(nn.Module):
    def __init__(self, tabular_dim: int, num_classes: int):
        super().__init__()
        self.backbone = DenseNetBackbone()
        self.fpn = FeaturePyramidNetwork(128)
        self.img_pool = nn.AdaptiveAvgPool2d(1)
        self.attention = SelfAttention(tabular_dim, 64)
        self.fc_fusion = nn.Linear(128 + 64, 128)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, images, tabular):
        img_feat = self.backbone(images)
        img_feat = self.fpn(img_feat)
        img_feat = self.img_pool(img_feat).flatten(1)

        tabular = tabular.unsqueeze(1)
        tab_feat = self.attention(tabular).mean(dim=1)

        fused = torch.cat([img_feat, tab_feat], dim=1)
        fused = self.fc_fusion(fused)
        return self.classifier(fused)


class MedFusionGradCAMWrapper(nn.Module):
    """Make MedFusionNet look like a single-input classifier for Grad-CAM."""

    def __init__(
        self,
        model: nn.Module,
        default_age: float = 48.0,
        target_labels: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.model = model
        self.default_age = float(default_age)
        self.class_labels = list(target_labels) if target_labels else MEDFUSION_LABELS
        self._permutation = (
            _build_permutation(MEDFUSION_LABELS, target_labels)
            if target_labels
            else None
        )

    def forward(self, images):
        return self.forward_with_age(images, None)

    def forward_with_age(self, images, age: Optional[torch.Tensor] = None):
        batch_size = images.shape[0]
        if age is None:
            age = torch.full(
                (batch_size, 1),
                self.default_age,
                device=images.device,
                dtype=images.dtype,
            )
        else:
            age = age.to(device=images.device, dtype=images.dtype)
            if age.ndim == 1:
                age = age.unsqueeze(1)
            if age.shape[-1] != 1:
                age = age.reshape(batch_size, 1)

        logits = self.model(images, age)
        if self._permutation is not None:
            logits = logits.index_select(1, self._permutation.to(logits.device))
        return logits


def load_medfusionnet_from_checkpoint(
    model_path: str | Path,
    *,
    device: Optional[torch.device] = None,
    tabular_dim: int = 1,
    num_classes: int = 14,
    default_age: float = 48.0,
    target_labels: Optional[Sequence[str]] = None,
) -> nn.Module:
    """Load the notebook MedFusionNet checkpoint and wrap it for Grad-CAM."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_path)
    if not model_path.exists():
        raise RuntimeError(f"Missing MedFusionNet checkpoint file: {model_path}")

    checkpoint = torch.load(str(model_path), map_location=device)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state_dict = None
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    state_dict = value
                    break
            if state_dict is None and _looks_like_state_dict(checkpoint):
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise RuntimeError(
                "Could not find a loadable state_dict in the MedFusionNet checkpoint."
            )

        model = MedFusionNet(tabular_dim=tabular_dim, num_classes=num_classes)
        cleaned_state_dict = _strip_module_prefix(state_dict)
        try:
            model.load_state_dict(cleaned_state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "MedFusionNet checkpoint does not exactly match the code architecture. "
                "Please compare the notebook architecture against "
                "`MedFusionNet/medfusionnet.py`."
            ) from exc

    model = model.to(device).eval()
    return MedFusionGradCAMWrapper(
        model,
        default_age=default_age,
        target_labels=target_labels,
    ).to(device).eval()
