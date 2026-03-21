# utils/gradcam.py
"""
Occlusion-based saliency maps for ONNX models.
Provides visual explainability by highlighting important regions in chest X-rays.
Uses the existing ctranscnn_1.onnx model — no extra model needed.
"""

import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from typing import Tuple
import io
import base64


class OcclusionSaliency:
    """
    Occlusion-based saliency map generator for ONNX models.

    Slides a gray patch across the image and measures drop in prediction
    confidence — regions where the drop is largest are most important.
    """

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def generate_heatmap(
        self,
        image: np.ndarray,
        target_class: int,
        patch_size: int = 28,
        stride: int = 14,
    ) -> np.ndarray:
        """
        Generate saliency heatmap using systematic occlusion.

        Args:
            image: Preprocessed image tensor (1, 3, H, W)
            target_class: Index of disease class to visualize
            patch_size: Size of the gray occluding patch
            stride: Step size for the sliding window

        Returns:
            Heatmap array (H, W) normalized to [0, 1]
        """
        _, _, h, w = image.shape

        # Baseline score for the target class
        baseline = self.session.run(
            [self.output_name], {self.input_name: image}
        )[0][0, target_class]

        # Grid dimensions
        rows = list(range(0, h - patch_size + 1, stride))
        cols = list(range(0, w - patch_size + 1, stride))

        # Fill value: channel-wise mean (gray in normalized space)
        fill = image.mean(axis=(2, 3), keepdims=True)  # (1, 3, 1, 1)

        importance = np.zeros((len(rows), len(cols)), dtype=np.float32)

        for ri, y in enumerate(rows):
            for ci, x in enumerate(cols):
                occluded = image.copy()
                occluded[:, :, y:y + patch_size, x:x + patch_size] = fill
                score = self.session.run(
                    [self.output_name], {self.input_name: occluded}
                )[0][0, target_class]
                # Positive value = occluding this region HURTS the prediction
                importance[ri, ci] = max(baseline - score, 0.0)

        # Up-sample to original spatial size with smooth interpolation
        heatmap = cv2.resize(
            importance, (w, h), interpolation=cv2.INTER_CUBIC
        )
        
        # ReLU to remove negative values from interpolation ringing
        heatmap = np.maximum(heatmap, 0)

        # Normalize to [0, 1]
        hmax = heatmap.max()
        if hmax > 0:
            heatmap /= hmax

        # Light Gaussian blur for smoother visualization
        ksize = patch_size // 2
        if ksize % 2 == 0:
            ksize += 1
        heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), 0)

        # Re-normalize after blur
        hmax = heatmap.max()
        if hmax > 0:
            heatmap /= hmax

        return heatmap


def apply_colormap_and_overlay(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Apply JET colormap to heatmap and blend with original image."""
    heatmap_resized = cv2.resize(
        heatmap, (original_image.shape[1], original_image.shape[0])
    )
    # Clip to [0, 1] to prevent integer wrap-around artifacts
    heatmap_uint8 = (np.clip(heatmap_resized, 0, 1) * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlayed = cv2.addWeighted(
        original_image.astype(np.uint8), 1 - alpha,
        heatmap_colored, alpha,
        0,
    )
    return overlayed


def generate_multi_disease_heatmaps(
    model_path: str,
    image_file,
    predictions: list,
    disease_labels: list,
    top_k: int = 5,
    threshold: float = 0.1,
) -> dict:
    """
    Generate saliency heatmaps for top predicted diseases.

    Uses the *same* ONNX model already used for inference — no extra weights.

    Returns dict keyed by disease name with base64-encoded images.
    """
    from utils.preprocess import preprocess_image

    # Preprocess for model
    preprocessed = preprocess_image(image_file)

    # Original image for overlay
    original_pil = Image.open(image_file).convert("RGB").resize((224, 224))
    original_np = np.array(original_pil)

    # Init saliency generator
    saliency = OcclusionSaliency(model_path)

    # Pick diseases to visualize
    scored = sorted(
        [
            (label, prob, idx)
            for idx, (label, prob) in enumerate(zip(disease_labels, predictions))
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    selected = [(l, p, i) for l, p, i in scored[:top_k] if p >= threshold]
    if not selected:
        selected = [scored[0]]

    results = {}
    for disease_label, prob, class_idx in selected:
        print(f"  -> Generating heatmap for {disease_label} (p={prob:.4f}) ...")
        heatmap = saliency.generate_heatmap(
            preprocessed, class_idx, patch_size=28, stride=14
        )

        original_b64 = pil_to_base64(original_pil)
        heatmap_colored = apply_colormap_to_heatmap(heatmap)
        heatmap_b64 = array_to_base64(heatmap_colored)
        overlay = apply_colormap_and_overlay(heatmap, original_np, alpha=0.5)
        overlay_b64 = array_to_base64(overlay)

        results[disease_label] = {
            "probability": float(prob),
            "original": original_b64,
            "heatmap": heatmap_b64,
            "overlay": overlay_b64,
        }

    return results


# ── helpers ──────────────────────────────────────────────────────────────


def apply_colormap_to_heatmap(
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    heatmap_resized = cv2.resize(heatmap, target_size)
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


def pil_to_base64(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def array_to_base64(array: np.ndarray) -> str:
    pil_image = Image.fromarray(array.astype(np.uint8))
    return pil_to_base64(pil_image)












################################################################











# import torch
# import torch.nn as nn
# import numpy as np
# import cv2
# import io
# import base64
# from PIL import Image
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

# # 1. THE WRAPPER (Corrected for your Tuple[List[Tensor]] output)
# class CTransCNN_GradCAM_Wrapper(nn.Module):
#     def __init__(self, backbone, classifier_head):
#         super().__init__()
#         self.backbone = backbone
#         self.classifier_head = classifier_head

#     def forward(self, x):
#         # backbone(x) returns a tuple: ( [[cnn_feat, trans_feat]], )
#         out_tuple = self.backbone(x)
        
#         # 1. Get the first element of the tuple: [[cnn_feat, trans_feat]]
#         last_stage_list = out_tuple[0] 
        
#         # 2. Get the list of tensors for the final stage: [cnn_feat, trans_feat]
#         # Depending on out_indices, this might be the last item in the list
#         final_features = last_stage_list[-1] 
        
#         cnn_feat = final_features[0]   # Shape: [B, 256]
#         trans_feat = final_features[1] # Shape: [B, 384]
        
#         # 3. Concatenate to 640
#         combined = torch.cat([cnn_feat, trans_feat], dim=1)
        
#         # 4. Final 14-label prediction
#         return self.classifier_head(combined)

# # 2. THE FAIL-SAFE LAYER FINDER
# def get_target_layer(model):
#     """
#     Specifically searches for the last fusion_block.conv3.
#     This is the spatial layer before the CNN branch is pooled.
#     """
#     target = None
#     # We want the highest index 'conv_trans_X'
#     highest_idx = -1
    
#     for name, module in model.named_modules():
#         if "fusion_block.conv3" in name:
#             # Extract number from 'conv_trans_12.fusion_block.conv3'
#             parts = name.split('.')
#             for p in parts:
#                 if 'conv_trans_' in p:
#                     idx = int(p.split('_')[-1])
#                     if idx > highest_idx:
#                         highest_idx = idx
#                         target = module
    
#     if target is None:
#         # Fallback to any last conv layer if the specific path isn't found
#         for module in reversed(list(model.modules())):
#             if isinstance(module, torch.nn.Conv2d):
#                 return module
#     return target

# # 3. MAIN GENERATION FUNCTION
# def generate_multi_disease_heatmaps(
#     model_path,
#     classifier_head,
#     image_file,
#     predictions,
#     disease_labels,
#     top_k=5,
#     threshold=0.1
# ):
#     try:
#         # Move models to device
#         device = next(model_path.parameters()).device
        
#         # Construct wrapper
#         full_model = CTransCNN_GradCAM_Wrapper(model_path, classifier_head).to(device).eval()
        
#         # Find the specific spatial layer
#         target_layer = get_target_layer(full_model)
        
#         # Initialize GradCAM
#         cam = GradCAM(model=full_model, target_layers=[target_layer])

#         # Preprocess image
#         from utils.preprocess import preprocess_image
#         input_tensor = preprocess_image(image_file).to(device)
        
#         # Prepare original image for overlay
#         original_pil = Image.open(image_file).convert("RGB").resize((224, 224))
#         original_np = np.array(original_pil)

#         # Result filtering
#         scored = sorted(
#             [(l, p, i) for i, (l, p) in enumerate(zip(disease_labels, predictions))],
#             key=lambda x: x[1], reverse=True
#         )
#         selected = [item for item in scored[:top_k] if item[1] >= threshold]
#         if not selected: selected = [scored[0]]

#         results = {}
#         for label, prob, idx in selected:
#             targets = [ClassifierOutputTarget(idx)]
            
#             # Generate mask
#             grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            
#             # Create colored overlay
#             visualization = show_cam_on_image(original_np / 255.0, grayscale_cam, use_rgb=True)

#             results[label] = {
#                 "probability": float(prob),
#                 "original": pil_to_base64(original_pil),
#                 "overlay": array_to_base64(visualization),
#             }
        
#         return results

#     except Exception as e:
#         import traceback
#         error_details = traceback.format_exc()
#         print(f"DEBUG GRAD-CAM ERROR:\n{error_details}")
#         return {"error": str(e)}

# # --- Helpers ---
# def pil_to_base64(pil_image):
#     buffered = io.BytesIO()
#     pil_image.save(buffered, format="PNG")
#     return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

# def array_to_base64(array):
#     pil_image = Image.fromarray(array.astype(np.uint8))
#     return pil_to_base64(pil_image)