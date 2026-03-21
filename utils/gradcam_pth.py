import torch
import torch.nn as nn
import numpy as np
import cv2
import io
import base64
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Wrapper for Grad-CAM Compatibility ---
class GradCAMWrapper(nn.Module):
    """
    Wraps the backbone and the new linear layer into one unit.
    Grad-CAM needs to see the final classification output.
    """
    def __init__(self, backbone, classifier_head):
        super().__init__()
        self.backbone = backbone
        self.classifier_head = classifier_head

    def forward(self, x):
        # Your CTransCNN returns a tuple of lists: ([(cnn_feat, trans_feat)],)
        features = self.backbone(x)
        cnn_feat = features[0][0][0]
        trans_feat = features[0][0][1]
        
        # Combine features (Match your training logic, e.g., concat)
        combined = torch.cat([cnn_feat, trans_feat], dim=1)
        return self.classifier_head(combined)

# --- The Main Logic ---
def generate_multi_disease_heatmaps(
    backbone_model,      # The CTransCNN .pth model
    classifier_head,     # Your new 14-label linear layer
    image_file,          # The uploaded image file
    predictions: list,   # Probabilities from the model
    disease_labels: list,# ["Atelectasis", "Cardiomegaly", ...]
    top_k: int = 5,
    threshold: float = 0.1,
):
    # 1. Setup Model and Target Layer
    full_model = GradCAMWrapper(backbone_model, classifier_head).eval()
    
    # Target the last CNN fusion layer
    target_layers = [full_model.backbone.conv_trans_12.fusion_block.conv3]
    cam = GradCAM(model=full_model, target_layers=target_layers)

    # 2. Preprocess Image
    # (Assuming you have a preprocess function similar to your previous one)
    from utils.preprocess import preprocess_image
    input_tensor = preprocess_image(image_file) # Returns [1, 3, 224, 224] tensor
    
    # Original image for display in Next.js
    original_pil = Image.open(image_file).convert("RGB").resize((224, 224))
    original_np = np.array(original_pil)

    # 3. Select Diseases to Visualize (Logic same as your old code)
    scored = sorted(
        [(l, p, i) for i, (l, p) in enumerate(zip(disease_labels, predictions))],
        key=lambda x: x[1], reverse=True
    )
    selected = [item for item in scored[:top_k] if item[1] >= threshold]
    if not selected: selected = [scored[0]]

    results = {}

    # 4. Loop through selected diseases and generate CAMs
    for label, prob, idx in selected:
        # Generate Grad-CAM for this specific disease index
        targets = [ClassifierOutputTarget(idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Create the heatmap overlay
        # Note: original_np / 255.0 scales it to [0,1] for the library
        visualization = show_cam_on_image(original_np / 255.0, grayscale_cam, use_rgb=True)

        # Convert to Base64 (The "Next.js Friendly" part)
        results[label] = {
            "probability": float(prob),
            "original": pil_to_base64(original_pil),
            "heatmap": array_to_base64((grayscale_cam * 255).astype(np.uint8)), # Raw heatmap
            "overlay": array_to_base64(visualization), # Blended image
        }

    return results

# --- Helper functions (Same as your previous script) ---
def pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

def array_to_base64(array):
    pil_image = Image.fromarray(array.astype(np.uint8))
    return pil_to_base64(pil_image)