"""
Digital Health Twin – FastAPI Backend
======================================
REST API that exposes ML prediction, RAG querying, and combined
insight generation for the Digital Health Twin system.

Endpoints:
    GET  /              → API info + health check
    GET  /health        → Health check
    GET  /model-info    → Model performance metrics
    POST /predict       → ML risk prediction
    POST /ask           → RAG-based Q&A
    POST /insight       → Combined ML + RAG insight
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

# ── FastAPI App ────────────────────────────────────────────────────────
app = FastAPI(
    title="Digital Health Twin API",
    description=(
        "AI-Powered Digital Health Twin system integrating ML prediction, "
        "RAG-based retrieval, and explainable insight generation for "
        "predictive healthcare with mental and physical health integration."
    ),
    version="1.0.0",
    contact={"name": "Shashi", "url": "https://github.com/shashi"},
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve UI static files
UI_DIR = PROJ_ROOT / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")


# ── Pydantic Models ───────────────────────────────────────────────────

class PatientData(BaseModel):
    """Patient health data for prediction."""
    age: float = Field(..., ge=1, le=120, description="Patient age in years")
    sex: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    cp: int = Field(0, ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(120, description="Resting blood pressure (mmHg)")
    chol: float = Field(200, description="Serum cholesterol (mg/dL)")
    fbs: int = Field(0, ge=0, le=1, description="Fasting blood sugar > 120 mg/dL")
    restecg: int = Field(0, ge=0, le=2, description="Resting ECG results")
    thalach: float = Field(150, description="Max heart rate achieved")
    exang: int = Field(0, ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(0, description="ST depression")
    slope: int = Field(1, ge=0, le=2, description="Peak exercise ST segment slope")
    ca: int = Field(0, ge=0, le=4, description="Number of major vessels (fluoroscopy)")
    thal: int = Field(2, ge=0, le=3, description="Thalassemia type")
    stress_level: float = Field(5, ge=1, le=10, description="Stress level (1-10)")
    sleep_quality: float = Field(7, ge=1, le=10, description="Sleep quality (1-10)")
    activity_hours: float = Field(5, ge=0, le=15, description="Weekly activity hours")
    anxiety_score: int = Field(7, ge=0, le=21, description="Anxiety score (GAD-7)")
    bmi: float = Field(25, ge=15, le=50, description="Body Mass Index")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 58, "sex": 1, "cp": 2, "trestbps": 145,
                    "chol": 260, "fbs": 1, "restecg": 0, "thalach": 130,
                    "exang": 1, "oldpeak": 2.5, "slope": 1, "ca": 2, "thal": 2,
                    "stress_level": 8.0, "sleep_quality": 3.5,
                    "activity_hours": 1.0, "anxiety_score": 15, "bmi": 32.5,
                }
            ]
        }
    }


class AskQuery(BaseModel):
    """Natural language query for RAG system."""
    query: str = Field(..., min_length=3, description="Your health question")
    top_k: int = Field(5, ge=1, le=20, description="Number of similar patients to retrieve")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "What health risks does a 55-year-old male with high cholesterol have?", "top_k": 5}
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    risk_label: str
    risk_level: str
    probability: float
    confidence: float
    model_used: str
    contributing_factors: List[Dict]


class AskResponse(BaseModel):
    query: str
    response: str
    sources: List[Dict]
    num_sources: int


class InsightResponse(BaseModel):
    insight: str
    ml_prediction: Dict
    rag_context: str
    similar_patients: List[Dict]
    recommendations: List[str]
    risk_factors: List[Dict]


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """API info and health check."""
    return {
        "name": "Digital Health Twin API",
        "version": "1.0.0",
        "description": "AI-Powered predictive healthcare with ML + RAG integration",
        "endpoints": {
            "POST /predict": "ML-based health risk prediction",
            "POST /ask": "RAG-based health question answering",
            "POST /insight": "Combined ML + RAG insight generation",
            "GET /health": "Health check",
            "GET /model-info": "Model performance metrics",
            "GET /docs": "Interactive API documentation (Swagger UI)",
        },
        "ui": "/ui/index.html",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "digital-health-twin"}


@app.get("/model-info")
async def model_info():
    """Return model performance metrics."""
    metrics_path = PROJ_ROOT / "models" / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(404, "Metrics not found. Train models first.")

    with open(metrics_path) as f:
        metrics = json.load(f)

    return {"models": metrics}


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """
    Predict health risk using trained ML models.
    Returns risk label, probability, and contributing factors.
    """
    try:
        from insight.engine import predict_risk
        result = predict_risk(patient.model_dump())
        return result
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask(query: AskQuery):
    """
    Ask a health-related question using RAG pipeline.
    Retrieves relevant patient contexts and generates a response.
    """
    try:
        from rag.retriever import ask as rag_ask
        result = rag_ask(query.query, top_k=query.top_k)
        return result
    except Exception as e:
        raise HTTPException(500, f"RAG query failed: {str(e)}")


@app.post("/insight", response_model=InsightResponse)
async def insight(patient: PatientData):
    """
    Generate comprehensive health insight combining ML prediction
    and RAG-retrieved context. This is the core Digital Health Twin
    functionality.
    """
    try:
        from insight.engine import generate_insight
        result = generate_insight(patient.model_dump())
        return result
    except Exception as e:
        raise HTTPException(500, f"Insight generation failed: {str(e)}")


@app.get("/ui")
async def serve_ui():
    """Serve the demo UI."""
    ui_path = UI_DIR / "index.html"
    if ui_path.exists():
        return FileResponse(str(ui_path))
    raise HTTPException(404, "UI not found")


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("Starting Digital Health Twin API …")
    print("Docs: http://localhost:8000/docs")
    print("UI:   http://localhost:8000/ui/index.html")
    uvicorn.run(app, host="0.0.0.0", port=8000)
