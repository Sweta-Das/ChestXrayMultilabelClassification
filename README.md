---
title: Chest X-ray Multilabel Classification
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: FastAPI and Next.js chest X-ray analysis demo with Grad-CAM and report generation
---

# Chest X-ray Multilabel Classification

AI-assisted chest X-ray analysis with fast ONNX prediction, on-demand PyTorch Grad-CAM explainability, and PDF report generation.

## What it does

- Upload a chest X-ray
- Run multi-label disease prediction with ONNX
- Select a disease and generate a Grad-CAM explanation with the MedFusionNet `.pth` model
- Fall back to ONNX occlusion if PyTorch explainability is unavailable
- Generate a clinical-style PDF report with RAG + LaTeX

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

## Runtime Notes

- The main app can run with the Mac frontend and a local MedFusionNet backend venv.
- The recommended backend venv is `.venv-medfusion`.
- `PYTHON_BIN` can point `start.sh` at a specific Python 3.10 interpreter if needed.
- `LATEX_BIN` can override the PDF compiler path.

## Deployment

The repository is now prepared for a single-container **Hugging Face Docker Space**:

- the Next.js frontend is exported statically
- FastAPI serves both the UI and the API from one container
- the app listens on port `7860` in Spaces

The deployable container is defined in [Dockerfile](./Dockerfile).

## More Detail

See `ARCHITECTURE.txt` for the full system breakdown and data flow.
