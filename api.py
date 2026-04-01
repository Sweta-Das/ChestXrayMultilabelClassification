# api.py
"""
FastAPI backend for AI-Assisted Chest X-ray Analysis.
Replaces Streamlit with professional REST API architecture.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from torch import nn
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import os
import uuid
from datetime import datetime
from pathlib import Path
import shutil
import io

# Load environment variables from project .env before importing modules that use them
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Import existing modules
from inference import predict
from rag.medical_reasoner import generate_structured_content
from utils.latex_renderer import render_latex_report
from utils.gradcam import generate_multi_disease_heatmaps


# =============================================================================
# Configuration
# =============================================================================

app = FastAPI(
    title="Chest X-ray AI Analysis API",
    description="AI-assisted chest X-ray analysis with GradCAM explainability",
    version="2.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_origin_regex="http://localhost.*",
)

# Disease labels
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

MODEL_PATH = "models/ctranscnn_1.onnx"
MODEL_PATH1 = "models/best_medfusionnet.pth"

RUNTIME_BASE = Path(os.getenv("APP_OUTPUT_DIR", "/tmp/chestxray-app"))
UPLOAD_DIR = RUNTIME_BASE / "outputs/uploads"
IMAGES_DIR = RUNTIME_BASE / "outputs/images"
REPORTS_DIR = RUNTIME_BASE / "outputs/reports"
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "out"

# Create directories in a writable runtime location.
for dir_path in [UPLOAD_DIR, IMAGES_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Models
# =============================================================================

class PredictionResponse(BaseModel):
    """Response model for inference endpoint."""
    session_id: str
    predictions: Dict[str, float]
    top_diseases: List[Dict[str, Any]]
    timestamp: str


class GradCAMRequest(BaseModel):
    """Request model for GradCAM generation."""
    session_id: str
    top_k: Optional[int] = 5
    threshold: Optional[float] = 0.1
    disease_index: Optional[int] = None


class GradCAMResponse(BaseModel):
    """Response model for GradCAM endpoint."""
    session_id: str
    heatmaps: Dict[str, Dict]


class ReportRequest(BaseModel):
    """Request model for report generation."""
    session_id: str
    patient_id: Optional[str] = None


class ReportResponse(BaseModel):
    """Response model for report generation."""
    session_id: str
    patient_id: str
    report_path: str
    impression: str
    findings: Dict[str, float]


# =============================================================================
# In-memory session storage (for demo; use Redis/DB in production)
# =============================================================================

sessions = {}


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/api/health")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Chest X-ray AI Analysis API",
        "version": "2.0.0"
    }


@app.post("/api/upload", response_model=dict)
async def upload_image(
    file: UploadFile = File(...),
    age: Optional[float] = Form(None),
):
    """
    Upload chest X-ray image.
    
    Returns session_id for subsequent operations.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate session ID
    session_id = f"session_{uuid.uuid4().hex}"
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{session_id}.png"
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Store session data
    sessions[session_id] = {
        "file_path": str(file_path),
        "age": float(age) if age is not None else 48.0,
        "uploaded_at": datetime.now().isoformat(),
        "predictions": None,
        "gradcam": None,
        "report": None
    }
    
    return {
        "session_id": session_id,
        "filename": file.filename,
        "message": "Image uploaded successfully"
    }


@app.post("/api/inference", response_model=PredictionResponse)
async def run_inference(session_id: str = Form(...)):
    """
    Run model inference on uploaded image.
    
    Returns disease probabilities for all 14 classes.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    file_path = session["file_path"]
    age = float(session.get("age", 48.0))
    
    # Run inference
    try:
        probs = predict(file_path, age=age)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    # Create prediction dictionary
    predictions = {
        label: float(prob) 
        for label, prob in zip(DISEASE_LABELS, probs)
    }
    
    # Get top diseases
    top_diseases = sorted(
        [{"disease": k, "probability": v} for k, v in predictions.items()],
        key=lambda x: x["probability"],
        reverse=True
    )[:5]
    
    # Store in session
    session["predictions"] = predictions
    session["probs_list"] = probs
    
    return PredictionResponse(
        session_id=session_id,
        predictions=predictions,
        top_diseases=top_diseases,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/gradcam", response_model=GradCAMResponse)
async def generate_gradcam(request: GradCAMRequest):
    """
    Generate GradCAM heatmaps for top predicted diseases.
    
    Returns base64-encoded heatmap images.
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[request.session_id]
    age = float(session.get("age", 48.0))
    
    if session["predictions"] is None:
        raise HTTPException(
            status_code=400, 
            detail="Run inference first before generating GradCAM"
        )

    cache_bucket = session.setdefault("gradcam_cache", {})
    if request.disease_index is not None:
        cache_key = f"disease:{request.disease_index}"
    else:
        cache_key = f"topk:{request.top_k}:threshold:{request.threshold}"

    cached_heatmaps = cache_bucket.get(cache_key)
    if cached_heatmaps is not None:
        session["gradcam"] = cached_heatmaps
        return GradCAMResponse(
            session_id=request.session_id,
            heatmaps=cached_heatmaps
        )
    
    # Generate GradCAM heatmaps
    try:
        heatmaps = generate_multi_disease_heatmaps(
            model_path=MODEL_PATH,
            image_file=session["file_path"],
            predictions=session["probs_list"],
            disease_labels=DISEASE_LABELS,
            top_k=request.top_k,
            threshold=request.threshold,
            target_index=request.disease_index,
            pth_model_path=MODEL_PATH1,
            age=age,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GradCAM generation failed: {str(e)}")

    # Store in session
    session["gradcam"] = heatmaps
    cache_bucket[cache_key] = heatmaps
    
    return GradCAMResponse(
        session_id=request.session_id,
        heatmaps=heatmaps
    )


@app.post("/api/report", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """
    Generate professional medical report PDF.
    
    Uses RAG-based reasoning and LaTeX rendering.
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[request.session_id]
    
    if session["predictions"] is None:
        raise HTTPException(
            status_code=400,
            detail="Run inference first before generating report"
        )
    
    # Generate patient ID
    patient_id = request.patient_id or f"PID-{uuid.uuid4().hex[:8].upper()}"
    
    # RAG-based medical reasoning
    try:
        structured = generate_structured_content(session["predictions"])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Medical reasoning failed: {str(e)}"
        )
    
    # Build enhanced LaTeX table rows with color-coded significance
    findings_items = list(structured["findings"].items())
    table_rows = []
    for i, (disease, prob) in enumerate(findings_items):
        pct = prob * 100
        # Color-coded significance level
        if pct >= 50:
            sig = r"\textcolor{riskHigh}{\textbf{High}}"
        elif pct >= 30:
            sig = r"\textcolor{riskModerate}{\textbf{Moderate}}"
        else:
            sig = r"\textcolor{riskLow}{\textbf{Low}}"
        # Alternating row shading
        row_bg = r"\rowcolor{tableRowAlt}" if i % 2 == 1 else ""
        table_rows.append(f"{row_bg}{disease} & {pct:.2f}\\% & {sig} \\\\")
    findings_rows = "\n".join(table_rows)

    # Determine primary finding and risk level
    primary_disease, primary_prob = findings_items[0]
    primary_pct = primary_prob * 100
    if primary_pct >= 50:
        risk_level = "HIGH"
        risk_color = "riskHigh"
        risk_desc = "Primary finding exceeds 50\\% confidence threshold. Clinical correlation strongly recommended."
    elif primary_pct >= 30:
        risk_level = "MODERATE"
        risk_color = "riskModerate"
        risk_desc = "Primary finding in moderate confidence range (30--49\\%). Further evaluation may be warranted."
    else:
        risk_level = "LOW"
        risk_color = "riskLow"
        risk_desc = "All findings below 30\\% confidence. No high-probability pathology detected by AI."

    # Generate unique report ID
    report_id = f"RPT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    # Prepare image path for LaTeX
    image_path = Path(session["file_path"]).resolve()
    reports_dir = REPORTS_DIR.resolve()
    relative_image_path = image_path.relative_to(reports_dir.parent)
    relative_image_path = "../" + relative_image_path.as_posix()
    
    # Normalize to ASCII to avoid pdflatex Unicode failures, then escape LaTeX chars
    impression_ascii = structured["impression"].encode("ascii", "ignore").decode("ascii")
    impression_text = latex_escape(impression_ascii)
    
    # LaTeX context with all professional template variables
    patient_age = session.get("age", 48.0)
    context = {
        "PATIENT_ID": patient_id,
        "PATIENT_AGE": str(int(patient_age)) if patient_age is not None else "N/A",
        "REPORT_DATE": datetime.now().strftime("%B %d, %Y at %H:%M"),
        "REPORT_ID": report_id,
        "XRAY_IMAGE_PATH": relative_image_path,
        "FINDINGS_TABLE": findings_rows,
        "IMPRESSION_TEXT": impression_text,
        "PRIMARY_FINDING": latex_escape(primary_disease),
        "PRIMARY_CONFIDENCE": f"{primary_pct:.1f}",
        "RISK_LEVEL": risk_level,
        "RISK_COLOR": risk_color,
        "RISK_DESCRIPTION": risk_desc,
        "NUM_CONDITIONS": "14",
    }
    
    # Render PDF with unique filename per session
    report_filename = f"report_{patient_id}"
    try:
        pdf_path = render_latex_report(
            template_path="templates/medical_report.tex",
            output_dir=str(REPORTS_DIR),
            context=context,
            filename=report_filename
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Report rendering failed: {str(e)}"
        )
    
    # Convert PosixPath to string
    pdf_path = str(pdf_path)
    
    # Store in session
    session["report"] = {
        "patient_id": patient_id,
        "pdf_path": pdf_path,
        "impression": structured["impression"],
        "findings": structured["findings"]
    }
    
    return ReportResponse(
        session_id=request.session_id,
        patient_id=patient_id,
        report_path=pdf_path,
        impression=structured["impression"],
        findings=structured["findings"]
    )


@app.get("/api/download/{session_id}")
async def download_report(session_id: str):
    """
    Download generated PDF report.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if session["report"] is None:
        raise HTTPException(status_code=404, detail="Report not generated")
    
    pdf_path = session["report"]["pdf_path"]
    patient_id = session["report"]["patient_id"]
    
    if not Path(pdf_path).exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"medical_report_{patient_id}.pdf"
    )


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """
    Get full session data.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Return session without large binary data
    return {
        "session_id": session_id,
        "uploaded_at": session["uploaded_at"],
        "has_predictions": session["predictions"] is not None,
        "has_gradcam": session["gradcam"] is not None,
        "has_report": session["report"] is not None,
        "predictions": session["predictions"],
        "report_info": {
            "patient_id": session["report"]["patient_id"],
            "impression": session["report"]["impression"]
        } if session["report"] else None
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete session and cleanup files.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Cleanup files
    try:
        Path(session["file_path"]).unlink(missing_ok=True)
        if session["report"]:
            Path(session["report"]["pdf_path"]).unlink(missing_ok=True)
    except Exception as e:
        print(f"Cleanup error: {e}")
    
    # Remove from sessions
    del sessions[session_id]
    
    return {"message": "Session deleted successfully"}


# =============================================================================
# Utility Functions
# =============================================================================

def latex_escape(text: str) -> str:
    """
    Escape LaTeX special characters.
    """
    replacements = [
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for k, v in replacements:
        text = text.replace(k, v)
    return text


if FRONTEND_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name="frontend")


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("🚀 Starting FastAPI Server")
    print("=" * 80)
    print("📍 API Documentation: http://localhost:8000/docs")
    print("📍 Health Check: http://localhost:8000/")
    print("=" * 80)
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )











###########################################################################################













# # api.py
# """
# FastAPI backend for AI-Assisted Chest X-ray Analysis.
# Replaces Streamlit with professional REST API architecture.
# """

# from fastapi import FastAPI, File, UploadFile, HTTPException, Form
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, JSONResponse
# from pydantic import BaseModel
# import torch
# from torch import nn
# from typing import List, Dict, Optional, Any
# from dotenv import load_dotenv
# import uuid
# from datetime import datetime
# from pathlib import Path
# import shutil
# import io

# import sys
# import 
# os
# from pathlib import Path

# # Get the directory that contains 'ChestXrayMultilabelClassification'
# # If api.py is inside the project, go up to the root
# root_path = str(Path(__file__).resolve().parent.parent) 
# if root_path not in sys.path:
#     sys.path.append(root_path)
    
# from ChestXrayMultilabelClassification.CTransCNN.model.models.backbones.CTransCNN import my_hybird_CTransCNN

# # Load environment variables from project .env before importing modules that use them
# BASE_DIR = Path(__file__).resolve().parent
# load_dotenv(dotenv_path=BASE_DIR / ".env")

# # Import existing modules
# from inference import predict
# from rag.medical_reasoner import generate_structured_content
# from utils.latex_renderer import render_latex_report
# from utils.gradcam import generate_multi_disease_heatmaps


# # =============================================================================
# # Configuration
# # =============================================================================

# app = FastAPI(
#     title="Chest X-ray AI Analysis API",
#     description="AI-assisted chest X-ray analysis with GradCAM explainability",
#     version="2.0.0"
# )

# # Enable CORS for Next.js frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:3000",
#         "http://127.0.0.1:3000",
#         "http://localhost:8000",
#     ],
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
#     allow_headers=["*"],
#     allow_origin_regex="http://localhost.*",
# )

# # Disease labels
# DISEASE_LABELS = [
#     "Atelectasis",
#     "Cardiomegaly",
#     "Effusion",
#     "Infiltration",
#     "Mass",
#     "Nodule",
#     "Pneumonia",
#     "Pneumothorax",
#     "Consolidation",
#     "Edema",
#     "Emphysema",
#     "Fibrosis",
#     "Pleural Thickening",
#     "Hernia",
# ]

# MODEL_PATH = "models/ctranscnn_1.onnx"
# # MODEL_PATH1 = "models/epoch_72.pth"

# # Storage paths
# UPLOAD_DIR = Path("outputs/uploads")
# IMAGES_DIR = Path("outputs/images")
# REPORTS_DIR = Path("outputs/reports")

# # Create directories
# for dir_path in [UPLOAD_DIR, IMAGES_DIR, REPORTS_DIR]:
#     dir_path.mkdir(parents=True, exist_ok=True)


# # =============================================================================
# # Data Models
# # =============================================================================

# class PredictionResponse(BaseModel):
#     """Response model for inference endpoint."""
#     session_id: str
#     predictions: Dict[str, float]
#     top_diseases: List[Dict[str, Any]]
#     timestamp: str


# class GradCAMRequest(BaseModel):
#     """Request model for GradCAM generation."""
#     session_id: str
#     top_k: Optional[int] = 5
#     threshold: Optional[float] = 0.1


# class GradCAMResponse(BaseModel):
#     """Response model for GradCAM endpoint."""
#     session_id: str
#     heatmaps: Dict[str, Dict]


# class ReportRequest(BaseModel):
#     """Request model for report generation."""
#     session_id: str
#     patient_id: Optional[str] = None


# class ReportResponse(BaseModel):
#     """Response model for report generation."""
#     session_id: str
#     patient_id: str
#     report_path: str
#     impression: str
#     findings: Dict[str, float]


# # =============================================================================
# # In-memory session storage (for demo; use Redis/DB in production)
# # =============================================================================

# sessions = {}


# # =============================================================================
# # API Endpoints
# # =============================================================================

# @app.get("/")
# async def root():
#     """Health check endpoint."""
#     return {
#         "status": "healthy",
#         "service": "Chest X-ray AI Analysis API",
#         "version": "2.0.0"
#     }


# @app.post("/api/upload", response_model=dict)
# async def upload_image(file: UploadFile = File(...)):
#     """
#     Upload chest X-ray image.
    
#     Returns session_id for subsequent operations.
#     """
#     # Validate file type
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="File must be an image")
    
#     # Generate session ID
#     session_id = f"session_{uuid.uuid4().hex}"
    
#     # Save uploaded file
#     file_path = UPLOAD_DIR / f"{session_id}.png"
    
#     with file_path.open("wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     # Store session data
#     sessions[session_id] = {
#         "file_path": str(file_path),
#         "uploaded_at": datetime.now().isoformat(),
#         "predictions": None,
#         "gradcam": None,
#         "report": None
#     }
    
#     return {
#         "session_id": session_id,
#         "filename": file.filename,
#         "message": "Image uploaded successfully"
#     }


# @app.post("/api/inference", response_model=PredictionResponse)
# async def run_inference(session_id: str = Form(...)):
#     """
#     Run model inference on uploaded image.
    
#     Returns disease probabilities for all 14 classes.
#     """
#     if session_id not in sessions:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     session = sessions[session_id]
#     file_path = session["file_path"]
    
#     # Run inference
#     try:
#         probs = predict(file_path)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
#     # Create prediction dictionary
#     predictions = {
#         label: float(prob) 
#         for label, prob in zip(DISEASE_LABELS, probs)
#     }
    
#     # Get top diseases
#     top_diseases = sorted(
#         [{"disease": k, "probability": v} for k, v in predictions.items()],
#         key=lambda x: x["probability"],
#         reverse=True
#     )[:5]
    
#     # Store in session
#     session["predictions"] = predictions
#     session["probs_list"] = probs
    
#     return PredictionResponse(
#         session_id=session_id,
#         predictions=predictions,
#         top_diseases=top_diseases,
#         timestamp=datetime.now().isoformat()
#     )

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 1. Initialize the Backbone
# # Using 'tiny' as per your CTransCNN code, out_indices=[12] for the final stage
# backbone = my_hybird_CTransCNN(arch='tiny', out_indices=[12])

# # 2. Initialize the Classifier Head (the 14-label linear layer)
# classifier_head = nn.Linear(640, 14)

# # 3. Load the weights from your .pth file
# MODEL_PATH1 = "models/epoch_72.pth"

# try:
#     print(f"Loading weights from {MODEL_PATH1}...")
#     checkpoint = torch.load(MODEL_PATH1, map_location=DEVICE)
    
#     # If your .pth file contains a combined state_dict:
#     if 'state_dict' in checkpoint:
#         # This depends on how you saved your model. 
#         # You might need to filter keys if they have prefixes like 'backbone.'
#         backbone.load_state_dict(checkpoint['state_dict'], strict=False)
#     else:
#         # If the .pth IS the state_dict
#         backbone.load_state_dict(checkpoint, strict=False)
    
#     print("Model weights loaded successfully.")
# except Exception as e:
#     print(f"Warning: Could not load weights: {e}. Grad-CAM will use random weights.")

# backbone.to(DEVICE).eval()
# classifier_head.to(DEVICE).eval()


# @app.post("/api/gradcam", response_model=GradCAMResponse)
# async def generate_gradcam(request: GradCAMRequest):
#     """
#     Generate GradCAM heatmaps for top predicted diseases.
#     """
#     if request.session_id not in sessions:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     session = sessions[request.session_id]
    
#     if session["predictions"] is None:
#         raise HTTPException(
#             status_code=400, 
#             detail="Run inference first before generating GradCAM"
#         )
    
#     try:
#         # !!! KEY FIX: Pass the 'backbone' and 'classifier_head' OBJECTS, not the path string !!!
#         heatmaps = generate_multi_disease_heatmaps(
#             backbone_model=backbone,          # The loaded PyTorch object
#             classifier_head=classifier_head,  # The loaded PyTorch object
#             image_file=session["file_path"],
#             predictions=session["probs_list"],
#             disease_labels=DISEASE_LABELS,
#             top_k=request.top_k,
#             threshold=request.threshold
#         )
        
#         # Check if the utility returned an internal error
#         if isinstance(heatmaps, dict) and "error" in heatmaps:
#             raise HTTPException(status_code=500, detail=heatmaps["error"])
            
#         # Store in session
#         session["gradcam"] = heatmaps
        
#         return GradCAMResponse(
#             session_id=request.session_id,
#             heatmaps=heatmaps
#         )

#     except Exception as e:
#         import traceback
#         print(traceback.format_exc()) 
#         raise HTTPException(status_code=500, detail=f"GradCAM generation failed: {str(e)}")


# @app.post("/api/report", response_model=ReportResponse)
# async def generate_report(request: ReportRequest):
#     """
#     Generate professional medical report PDF.
    
#     Uses RAG-based reasoning and LaTeX rendering.
#     """
#     if request.session_id not in sessions:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     session = sessions[request.session_id]
    
#     if session["predictions"] is None:
#         raise HTTPException(
#             status_code=400,
#             detail="Run inference first before generating report"
#         )
    
#     # Generate patient ID
#     patient_id = request.patient_id or f"PID-{uuid.uuid4().hex[:8].upper()}"
    
#     # RAG-based medical reasoning
#     try:
#         structured = generate_structured_content(session["predictions"])
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Medical reasoning failed: {str(e)}"
#         )
    
#     # Build enhanced LaTeX table rows with color-coded significance
#     findings_items = list(structured["findings"].items())
#     table_rows = []
#     for i, (disease, prob) in enumerate(findings_items):
#         pct = prob * 100
#         # Color-coded significance level
#         if pct >= 50:
#             sig = r"\textcolor{riskHigh}{\textbf{High}}"
#         elif pct >= 30:
#             sig = r"\textcolor{riskModerate}{\textbf{Moderate}}"
#         else:
#             sig = r"\textcolor{riskLow}{\textbf{Low}}"
#         # Alternating row shading
#         row_bg = r"\rowcolor{tableRowAlt}" if i % 2 == 1 else ""
#         table_rows.append(f"{row_bg}{disease} & {pct:.2f}\\% & {sig} \\\\")
#     findings_rows = "\n".join(table_rows)

#     # Determine primary finding and risk level
#     primary_disease, primary_prob = findings_items[0]
#     primary_pct = primary_prob * 100
#     if primary_pct >= 50:
#         risk_level = "HIGH"
#         risk_color = "riskHigh"
#         risk_desc = "Primary finding exceeds 50\\% confidence threshold. Clinical correlation strongly recommended."
#     elif primary_pct >= 30:
#         risk_level = "MODERATE"
#         risk_color = "riskModerate"
#         risk_desc = "Primary finding in moderate confidence range (30--49\\%). Further evaluation may be warranted."
#     else:
#         risk_level = "LOW"
#         risk_color = "riskLow"
#         risk_desc = "All findings below 30\\% confidence. No high-probability pathology detected by AI."

#     # Generate unique report ID
#     report_id = f"RPT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

#     # Prepare image path for LaTeX
#     image_path = Path(session["file_path"]).resolve()
#     reports_dir = REPORTS_DIR.resolve()
#     relative_image_path = image_path.relative_to(reports_dir.parent)
#     relative_image_path = "../" + relative_image_path.as_posix()
    
#     # Normalize to ASCII to avoid pdflatex Unicode failures, then escape LaTeX chars
#     impression_ascii = structured["impression"].encode("ascii", "ignore").decode("ascii")
#     impression_text = latex_escape(impression_ascii)
    
#     # LaTeX context with all professional template variables
#     context = {
#         "PATIENT_ID": patient_id,
#         "REPORT_DATE": datetime.now().strftime("%B %d, %Y at %H:%M"),
#         "REPORT_ID": report_id,
#         "XRAY_IMAGE_PATH": relative_image_path,
#         "FINDINGS_TABLE": findings_rows,
#         "IMPRESSION_TEXT": impression_text,
#         "PRIMARY_FINDING": latex_escape(primary_disease),
#         "PRIMARY_CONFIDENCE": f"{primary_pct:.1f}",
#         "RISK_LEVEL": risk_level,
#         "RISK_COLOR": risk_color,
#         "RISK_DESCRIPTION": risk_desc,
#         "NUM_CONDITIONS": "14",
#     }
    
#     # Render PDF with unique filename per session
#     report_filename = f"report_{patient_id}"
#     try:
#         pdf_path = render_latex_report(
#             template_path="templates/medical_report.tex",
#             output_dir=str(REPORTS_DIR),
#             context=context,
#             filename=report_filename
#         )
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Report rendering failed: {str(e)}"
#         )
    
#     # Convert PosixPath to string
#     pdf_path = str(pdf_path)
    
#     # Store in session
#     session["report"] = {
#         "patient_id": patient_id,
#         "pdf_path": pdf_path,
#         "impression": structured["impression"],
#         "findings": structured["findings"]
#     }
    
#     return ReportResponse(
#         session_id=request.session_id,
#         patient_id=patient_id,
#         report_path=pdf_path,
#         impression=structured["impression"],
#         findings=structured["findings"]
#     )


# @app.get("/api/download/{session_id}")
# async def download_report(session_id: str):
#     """
#     Download generated PDF report.
#     """
#     if session_id not in sessions:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     session = sessions[session_id]
    
#     if session["report"] is None:
#         raise HTTPException(status_code=404, detail="Report not generated")
    
#     pdf_path = session["report"]["pdf_path"]
#     patient_id = session["report"]["patient_id"]
    
#     if not Path(pdf_path).exists():
#         raise HTTPException(status_code=404, detail="Report file not found")
    
#     return FileResponse(
#         pdf_path,
#         media_type="application/pdf",
#         filename=f"medical_report_{patient_id}.pdf"
#     )


# @app.get("/api/session/{session_id}")
# async def get_session(session_id: str):
#     """
#     Get full session data.
#     """
#     if session_id not in sessions:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     session = sessions[session_id]
    
#     # Return session without large binary data
#     return {
#         "session_id": session_id,
#         "uploaded_at": session["uploaded_at"],
#         "has_predictions": session["predictions"] is not None,
#         "has_gradcam": session["gradcam"] is not None,
#         "has_report": session["report"] is not None,
#         "predictions": session["predictions"],
#         "report_info": {
#             "patient_id": session["report"]["patient_id"],
#             "impression": session["report"]["impression"]
#         } if session["report"] else None
#     }


# @app.delete("/api/session/{session_id}")
# async def delete_session(session_id: str):
#     """
#     Delete session and cleanup files.
#     """
#     if session_id not in sessions:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     session = sessions[session_id]
    
#     # Cleanup files
#     try:
#         Path(session["file_path"]).unlink(missing_ok=True)
#         if session["report"]:
#             Path(session["report"]["pdf_path"]).unlink(missing_ok=True)
#     except Exception as e:
#         print(f"Cleanup error: {e}")
    
#     # Remove from sessions
#     del sessions[session_id]
    
#     return {"message": "Session deleted successfully"}


# # =============================================================================
# # Utility Functions
# # =============================================================================

# def latex_escape(text: str) -> str:
#     """
#     Escape LaTeX special characters.
#     """
#     replacements = [
#         ("&", r"\&"),
#         ("%", r"\%"),
#         ("$", r"\$"),
#         ("#", r"\#"),
#         ("_", r"\_"),
#         ("{", r"\{"),
#         ("}", r"\}"),
#         ("~", r"\textasciitilde{}"),
#         ("^", r"\textasciicircum{}"),
#     ]
#     for k, v in replacements:
#         text = text.replace(k, v)
#     return text


# # =============================================================================
# # Run Server
# # =============================================================================

# if __name__ == "__main__":
#     import uvicorn
    
#     print("=" * 80)
#     print("🚀 Starting FastAPI Server")
#     print("=" * 80)
#     print("📍 API Documentation: http://localhost:8000/docs")
#     print("📍 Health Check: http://localhost:8000/")
#     print("=" * 80)
    
#     uvicorn.run(
#         "api:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True
#     )
