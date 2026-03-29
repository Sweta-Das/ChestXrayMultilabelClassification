"""Backend inference helper.

This now prefers the MedFusionNet checkpoint so the predictions used by the
app and the Grad-CAM explanations come from the same model.
"""

from __future__ import annotations

from functools import lru_cache

import onnxruntime as ort
import torch

from MedFusionNet.medfusionnet import load_medfusionnet_from_checkpoint
from utils.preprocess import preprocess_image

MEDFUSION_MODEL_PATH = "models/best_medfusionnet.pth"
ONNX_MODEL_PATH = "models/ctranscnn_1.onnx"

# Keep the frontend/API label order stable.
DISEASE_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia",
]


@lru_cache(maxsize=1)
def _load_medfusion_model():
    return load_medfusionnet_from_checkpoint(
        MEDFUSION_MODEL_PATH,
        target_labels=DISEASE_LABELS,
    )


@lru_cache(maxsize=1)
def _load_onnx_session():
    session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name


def _predict_with_medfusion(img_file, age: float | None = None):
    model = _load_medfusion_model()
    device = next(model.parameters()).device
    x = torch.from_numpy(preprocess_image(img_file)).to(device)
    age_tensor = torch.tensor([[float(age) if age is not None else 48.0]], device=device)

    with torch.no_grad():
        logits = model.forward_with_age(x, age_tensor)
        probs = torch.sigmoid(logits)[0].detach().cpu().tolist()

    return probs


def _predict_with_onnx(img_file):
    x = preprocess_image(img_file)
    session, input_name, output_name = _load_onnx_session()
    outputs = session.run([output_name], {input_name: x})
    return outputs[0][0].tolist()


def predict(img_file, age: float | None = None):
    """Return a list of 14 disease probabilities."""

    try:
        return _predict_with_medfusion(img_file, age=age)
    except Exception as exc:
        print(
            "MedFusionNet inference unavailable, falling back to ONNX: "
            f"{type(exc).__name__}: {exc}"
        )
        return _predict_with_onnx(img_file)
