# Chest X-ray Multilabel Classification

AI-assisted chest X-ray analysis with fast ONNX prediction, on-demand PyTorch Grad-CAM explainability, and PDF report generation.

## What it does

- Upload a chest X-ray
- Run multi-label disease prediction with ONNX
- Select a disease and generate a Grad-CAM explanation with the MedFusionNet `.pth` model
- Fall back to ONNX occlusion if PyTorch explainability is unavailable
- Generate a clinical-style PDF report with RAG + LaTeX
- Compare Grad-CAM against NIH bbox annotations for evaluation

## Architecture at a Glance

- Frontend: Next.js, TypeScript, Zustand
- Backend: FastAPI, Python 3.10+
- Prediction: `models/ctranscnn_1.onnx`
- Explainability: `models/best_medfusionnet.pth`
- Fallback heatmap: ONNX occlusion
- Report generation: OpenAI RAG + LaTeX PDF
- Evaluation: `scripts/compare_gradcam_bbox.py`

## Main Flow

1. User uploads an X-ray in the frontend
2. Backend saves the image and creates a session
3. ONNX model returns disease probabilities
4. User selects a disease and clicks Explain
5. Backend generates Grad-CAM from the PyTorch checkpoint
6. If needed, backend falls back to ONNX occlusion
7. User can generate a PDF medical report

## Key Files

- `api.py` - FastAPI routes
- `inference.py` - ONNX prediction
- `utils/gradcam.py` - Grad-CAM router and fallback
- `utils/gradcam_pth.py` - PyTorch Grad-CAM implementation
- `utils/latex_renderer.py` - PDF rendering
- `scripts/compare_gradcam_bbox.py` - bbox vs Grad-CAM comparison
- `frontend/lib/api.ts` - frontend API client
- `frontend/lib/store.ts` - state management

## Endpoints

- `POST /api/upload`
- `POST /api/inference`
- `POST /api/gradcam`
- `POST /api/report`
- `GET /api/download/{session_id}`
- `GET /api/session/{session_id}`
- `DELETE /api/session/{session_id}`

## Evaluation With NIH BBoxes

The `ref/ref.csv` file contains NIH bounding-box annotations. The comparison script:

- reads the CSV
- draws the ground-truth bbox
- runs Grad-CAM for the matching disease
- saves original, heatmap, and overlay views
- writes a small JSON summary per sample

This is useful for checking whether the explanation lands in the same anatomical region as the annotation.

## Runtime Notes

- The main app can run with the Mac frontend and a local MedFusionNet backend venv.
- The recommended backend venv is `.venv-medfusion`.
- `PYTHON_BIN` can point `start.sh` at a specific Python 3.10 interpreter if needed.
- `LATEX_BIN` can override the PDF compiler path.

## More Detail

See `ARCHITECTURE.txt` for the full system breakdown and data flow.
