"""PyTorch Grad-CAM helpers for the CTransCNN checkpoint.

This module is responsible for the higher-quality explanation path:
load the `.pth` model only when the user asks for an explanation, then
generate Grad-CAM for the selected disease index.
"""

from __future__ import annotations

import base64
import argparse
import json
import io
from functools import lru_cache
from pathlib import Path
import sys
from typing import Callable, Optional, Sequence

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

_CACHED_MODEL_KEYS: set[str] = set()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CTRANS_CNN_ROOT = PROJECT_ROOT / "CTransCNN"
CTRANS_CNN_CONFIG = CTRANS_CNN_ROOT / "configs" / "NIH_ChestX-ray14_CTransCNN.py"


class GradCAMWrapper(nn.Module):
    """Wrap the classifier so Grad-CAM sees logits, not training loss."""

    def __init__(self, backbone_or_classifier: nn.Module, classifier_head: Optional[nn.Module] = None):
        super().__init__()
        self.backbone = backbone_or_classifier
        self.classifier_head = classifier_head

    def forward(self, x):
        if self.classifier_head is None:
            # `return_loss=False` routes through `simple_test`, while disabling
            # post-processing exposes the raw logits needed by Grad-CAM.
            return self.backbone(
                x,
                return_loss=False,
                sigmoid=False,
                post_process=False,
            )

        features = self.backbone(x)

        if isinstance(features, (tuple, list)) and features:
            stage = features[0]
            if isinstance(stage, (tuple, list)) and stage:
                final_features = stage[-1]
                if isinstance(final_features, (tuple, list)) and len(final_features) >= 2:
                    cnn_feat = final_features[0]
                    trans_feat = final_features[1]
                    combined = torch.cat([cnn_feat, trans_feat], dim=1)
                    return self.classifier_head(combined)

        # If the loaded checkpoint is already a full end-to-end model,
        # fall back to its native output.
        return features


def _to_device(model: nn.Module, device: torch.device) -> nn.Module:
    return model.to(device).eval()


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "backbone"):
            value = checkpoint.get(key)
            if isinstance(value, nn.Module):
                return value
            if isinstance(value, dict):
                return value
    return checkpoint


def load_model_from_checkpoint(
    model_path: str | Path,
    *,
    device: Optional[torch.device] = None,
    model_factory: Optional[Callable[[], nn.Module]] = None,
) -> nn.Module:
    """Load a torch model from a checkpoint.

    Supports three common layouts:
    - a full serialized nn.Module
    - a dict containing a full model under `model`
    - a state_dict paired with `model_factory`

    The repo currently ships only the checkpoint, so this stays flexible.
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_factory is None:
        cache_key = f"{str(model_path)}::{str(device)}"
        if cache_key in _CACHED_MODEL_KEYS:
            print(f"Reusing cached Grad-CAM model: {cache_key}", file=sys.stderr)
        else:
            print(f"Loading Grad-CAM model into cache: {cache_key}", file=sys.stderr)
        load_errors = []
        try:
            model = _load_ctranscnn_model_cached(str(model_path), str(device))
        except Exception as exc:
            load_errors.append(f"config-based load: {type(exc).__name__}: {exc}")
            print(
                "CTransCNN config-based load failed, falling back to generic checkpoint load: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            try:
                model = _load_model_from_checkpoint_cached(str(model_path), str(device))
            except Exception as fallback_exc:
                load_errors.append(
                    f"generic checkpoint load: {type(fallback_exc).__name__}: {fallback_exc}"
                )
                raise RuntimeError(
                    "Unable to load the CTransCNN .pth checkpoint. "
                    "Checked the config-based rebuild from "
                    f"{CTRANS_CNN_CONFIG} and the generic torch.load fallback. "
                    "If this is a state_dict checkpoint, ensure the backend has "
                    "the CTransCNN dependencies from CTransCNN/requirements.txt. "
                    f"Details: {' | '.join(load_errors)}"
                ) from fallback_exc
        _CACHED_MODEL_KEYS.add(cache_key)
        return model

    checkpoint = torch.load(str(model_path), map_location=device)

    if isinstance(checkpoint, nn.Module):
        return _to_device(checkpoint, device)

    checkpoint = _extract_state_dict(checkpoint)

    if isinstance(checkpoint, nn.Module):
        return _to_device(checkpoint, device)

    if isinstance(checkpoint, dict):
        backbone = checkpoint.get("backbone")
        classifier_head = checkpoint.get("classifier_head")
        if isinstance(backbone, nn.Module) and isinstance(classifier_head, nn.Module):
            return _to_device(GradCAMWrapper(backbone, classifier_head), device)

    if isinstance(checkpoint, dict) and model_factory is not None:
        model = model_factory()
        model.load_state_dict(checkpoint, strict=False)
        return _to_device(model, device)

    raise RuntimeError(
        "Could not reconstruct the PyTorch model from the checkpoint. "
        "If this .pth contains only weights, pass a `model_factory` that "
        "builds the architecture before loading Grad-CAM."
    )


@lru_cache(maxsize=4)
def _load_ctranscnn_model_cached(model_path: str, device_str: str) -> nn.Module:
    """Build the original CTransCNN classifier and load the checkpoint once."""

    device = torch.device(device_str)

    if str(CTRANS_CNN_ROOT) not in sys.path:
        sys.path.insert(0, str(CTRANS_CNN_ROOT))

    try:
        import mmcv
        from mmcv.runner import load_checkpoint
        from model.models import build_classifier
    except ImportError as exc:
            raise RuntimeError(
                "Missing CTransCNN runtime dependencies. Install the packages "
                "from CTransCNN/requirements.txt, especially `torch` and "
                "`mmcv`, before using the PyTorch Grad-CAM path."
            ) from exc

    if not CTRANS_CNN_CONFIG.exists():
        raise RuntimeError(f"Missing CTransCNN config file: {CTRANS_CNN_CONFIG}")

    if not Path(model_path).exists():
        raise RuntimeError(f"Missing checkpoint file: {model_path}")

    cfg = mmcv.Config.fromfile(str(CTRANS_CNN_CONFIG))
    classifier = build_classifier(cfg.model)
    try:
        load_checkpoint(classifier, model_path, map_location=device)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint {model_path} into the CTransCNN classifier."
        ) from exc

    return _to_device(GradCAMWrapper(classifier), device)


@lru_cache(maxsize=4)
def _load_model_from_checkpoint_cached(model_path: str, device_str: str) -> nn.Module:
    device = torch.device(device_str)
    if not Path(model_path).exists():
        raise RuntimeError(f"Missing checkpoint file: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, nn.Module):
        return _to_device(checkpoint, device)

    checkpoint = _extract_state_dict(checkpoint)

    if isinstance(checkpoint, nn.Module):
        return _to_device(checkpoint, device)

    if isinstance(checkpoint, dict):
        backbone = checkpoint.get("backbone")
        classifier_head = checkpoint.get("classifier_head")
        if isinstance(backbone, nn.Module) and isinstance(classifier_head, nn.Module):
            return _to_device(GradCAMWrapper(backbone, classifier_head), device)

    raise RuntimeError(
        "Could not reconstruct the PyTorch model from the checkpoint. "
        "This fallback expects a serialized nn.Module or a checkpoint "
        "containing `backbone` and `classifier_head` modules."
    )


def find_target_layer(model: nn.Module) -> nn.Module:
    """Find the spatial layer Grad-CAM should hook into."""

    preferred_suffixes = (
        "conv_trans_12.fusion_block.conv3",
        "fusion_block.conv3",
    )
    for name, module in model.named_modules():
        if any(name.endswith(suffix) for suffix in preferred_suffixes):
            return module

    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module

    raise RuntimeError("Could not find a suitable target layer for Grad-CAM.")


def _select_targets(
    predictions: Sequence[float],
    disease_labels: Sequence[str],
    target_index: Optional[int],
    top_k: int,
    threshold: float,
):
    scored = sorted(
        [(label, float(prob), idx) for idx, (label, prob) in enumerate(zip(disease_labels, predictions))],
        key=lambda item: item[1],
        reverse=True,
    )

    if not scored:
        return []

    if target_index is not None:
        if target_index < 0 or target_index >= len(scored):
            raise ValueError(f"target_index {target_index} is out of range")
        label = disease_labels[target_index]
        prob = float(predictions[target_index])
        return [(label, prob, target_index)]

    selected = [item for item in scored[:top_k] if item[1] >= threshold]
    return selected or [scored[0]]


def generate_multi_disease_heatmaps(
    model_path: str | Path,
    image_file,
    predictions: Sequence[float],
    disease_labels: Sequence[str],
    top_k: int = 5,
    threshold: float = 0.1,
    target_index: Optional[int] = None,
    model_factory: Optional[Callable[[], nn.Module]] = None,
) -> dict:
    """Generate Grad-CAM overlays using the PyTorch checkpoint.

    If `target_index` is provided, only that disease is visualized.
    Otherwise we fall back to the highest-confidence diseases.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(
        model_path,
        device=device,
        model_factory=model_factory,
    )

    target_layer = find_target_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer])

    from utils.preprocess import preprocess_image

    input_tensor = torch.from_numpy(preprocess_image(image_file)).to(device)

    original_pil = Image.open(image_file).convert("RGB").resize((224, 224))
    original_np = np.array(original_pil)

    selected = _select_targets(
        predictions=predictions,
        disease_labels=disease_labels,
        target_index=target_index,
        top_k=top_k,
        threshold=threshold,
    )

    if not selected:
        return {}

    results = {}
    for label, prob, class_idx in selected:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(class_idx)],
        )[0, :]

        heatmap_colored = cam_to_colored_heatmap_image(grayscale_cam)

        visualization = show_cam_on_image(
            original_np.astype(np.float32) / 255.0,
            grayscale_cam,
            use_rgb=True,
        )

        results[label] = {
            "probability": float(prob),
            "original": pil_to_base64(original_pil),
            "heatmap": array_to_base64(heatmap_colored),
            "overlay": array_to_base64(visualization),
        }

    return results


def pil_to_base64(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"


def array_to_base64(array: np.ndarray) -> str:
    pil_image = Image.fromarray(array.astype(np.uint8))
    return pil_to_base64(pil_image)


def cam_to_colored_heatmap_image(
    grayscale_cam: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Convert a CAM mask into a smoother, colorized heatmap image."""

    cam = np.asarray(grayscale_cam, dtype=np.float32)
    cam = np.maximum(cam, 0)
    cam_max = float(np.percentile(cam, 99.5))
    if cam_max > 0:
        cam = np.clip(cam / cam_max, 0, 1)

    cam = np.power(cam, 0.88)

    cam = cv2.resize(cam, target_size, interpolation=cv2.INTER_CUBIC)
    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=1.0, sigmaY=1.0)

    cam_max = float(cam.max())
    if cam_max > 0:
        cam = cam / cam_max

    heatmap_uint8 = np.clip(cam * 255.0, 0, 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    return cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)


def _cli():
    parser = argparse.ArgumentParser(description="Generate CTransCNN Grad-CAM heatmaps")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-file", required=True)
    parser.add_argument("--predictions", required=True, help="JSON list of probabilities")
    parser.add_argument("--disease-labels", required=True, help="JSON list of labels")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--target-index", type=int, default=None)
    args = parser.parse_args()

    predictions = json.loads(args.predictions)
    disease_labels = json.loads(args.disease_labels)

    result = generate_multi_disease_heatmaps(
        model_path=args.model_path,
        image_file=args.image_file,
        predictions=predictions,
        disease_labels=disease_labels,
        top_k=args.top_k,
        threshold=args.threshold,
        target_index=args.target_index,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    _cli()
