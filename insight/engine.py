"""
Insight Engine
===============
Combines ML prediction results with RAG-retrieved context to generate
explainable, actionable health insights for the Digital Health Twin.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

PROJ_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJ_ROOT / "models"

# Add project root to path for RAG imports
sys.path.insert(0, str(PROJ_ROOT))

# Lazy-loaded models
_scaler = None
_model = None
_feature_cols = None


def _load_model():
    """Load the best ML model (XGBoost) and scaler."""
    global _scaler, _model, _feature_cols
    if _model is None:
        _scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        _feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
        # Try XGBoost first, fall back to Random Forest
        try:
            _model = joblib.load(MODELS_DIR / "xgboost_model.pkl")
        except Exception:
            _model = joblib.load(MODELS_DIR / "random_forest.pkl")
    return _model, _scaler, _feature_cols


def predict_risk(patient_data: Dict) -> Dict:
    """
    Run ML prediction on patient data.
    Returns risk label, probability, and contributing factors.
    """
    model, scaler, feature_cols = _load_model()

    # Build feature vector
    features = []
    for col in feature_cols:
        val = patient_data.get(col, 0)
        features.append(float(val))

    X = np.array([features])

    # Check if model needs scaling (Logistic Regression does, XGBoost doesn't)
    model_name = type(model).__name__
    if model_name in ["LogisticRegression"]:
        X = scaler.transform(X)

    # Predict
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    risk_level = (
        "critical" if probability > 0.8
        else "high" if probability > 0.6
        else "moderate" if probability > 0.4
        else "low"
    )

    # Identify contributing factors
    factors = _identify_risk_factors(patient_data)

    return {
        "prediction": prediction,
        "risk_label": risk_label,
        "risk_level": risk_level,
        "probability": round(probability, 4),
        "confidence": round(max(probability, 1 - probability), 4),
        "model_used": model_name,
        "contributing_factors": factors,
    }


def _identify_risk_factors(data: Dict) -> List[Dict]:
    """Identify key risk factors from patient data using clinical thresholds."""
    factors = []

    # Cholesterol
    chol = data.get("chol", 0)
    if chol > 240:
        factors.append({"factor": "High cholesterol", "value": f"{chol} mg/dL",
                         "severity": "high", "detail": "Cholesterol above 240 mg/dL significantly increases cardiovascular risk."})
    elif chol > 200:
        factors.append({"factor": "Borderline cholesterol", "value": f"{chol} mg/dL",
                         "severity": "moderate", "detail": "Cholesterol 200-240 mg/dL warrants monitoring."})

    # Blood pressure
    bp = data.get("trestbps", 0)
    if bp > 140:
        factors.append({"factor": "Hypertension", "value": f"{bp} mmHg",
                         "severity": "high", "detail": "Resting BP above 140 mmHg indicates hypertension."})
    elif bp > 120:
        factors.append({"factor": "Elevated blood pressure", "value": f"{bp} mmHg",
                         "severity": "moderate", "detail": "Pre-hypertensive range."})

    # BMI
    bmi = data.get("bmi", 0)
    if bmi > 30:
        factors.append({"factor": "Obesity", "value": f"BMI {bmi:.1f}",
                         "severity": "high", "detail": "BMI above 30 classified as obese."})
    elif bmi > 25:
        factors.append({"factor": "Overweight", "value": f"BMI {bmi:.1f}",
                         "severity": "moderate", "detail": "BMI 25-30 classified as overweight."})

    # Heart rate
    thalach = data.get("thalach", 0)
    if thalach and thalach < 120:
        factors.append({"factor": "Low max heart rate", "value": f"{thalach} bpm",
                         "severity": "moderate", "detail": "Low exercise heart rate may indicate cardiac limitation."})

    # Stress
    stress = data.get("stress_level", 0)
    if stress > 7:
        factors.append({"factor": "High stress", "value": f"{stress}/10",
                         "severity": "high", "detail": "Elevated stress increases cardiovascular risk."})

    # Anxiety
    anxiety = data.get("anxiety_score", 0)
    if anxiety > 14:
        factors.append({"factor": "Severe anxiety", "value": f"GAD-7: {anxiety}/21",
                         "severity": "high", "detail": "Severe anxiety strongly correlates with cardiac events."})
    elif anxiety > 9:
        factors.append({"factor": "Moderate anxiety", "value": f"GAD-7: {anxiety}/21",
                         "severity": "moderate", "detail": "Moderate anxiety warrants clinical attention."})

    # Sleep
    sleep = data.get("sleep_quality", 0)
    if sleep and sleep < 4:
        factors.append({"factor": "Poor sleep quality", "value": f"{sleep}/10",
                         "severity": "high", "detail": "Poor sleep strongly linked to cardiovascular disease."})

    # Activity
    activity = data.get("activity_hours", 0)
    if activity is not None and activity < 2:
        factors.append({"factor": "Sedentary lifestyle", "value": f"{activity} hrs/week",
                         "severity": "moderate", "detail": "Less than 2 hours/week of activity increases risk."})

    # Age
    age = data.get("age", 0)
    if age > 60:
        factors.append({"factor": "Advanced age", "value": f"{age} years",
                         "severity": "moderate", "detail": "Age above 60 is an independent risk factor."})

    # Sort by severity
    severity_order = {"high": 0, "moderate": 1, "low": 2}
    factors.sort(key=lambda x: severity_order.get(x["severity"], 3))

    return factors


def generate_insight(patient_data: Dict) -> Dict:
    """
    CORE: Combine ML prediction + RAG context -> explainable insight.
    This is the heart of the Digital Health Twin system.

    Pipeline:
        1. ML model predicts risk probability + identifies contributing factors
        2. RAG retrieves the most similar patient profiles from the vector DB
        3. Comparative analysis quantifies how the patient differs from norms
        4. All signals are fused into a human-readable, clinician-style narrative
    """
    from rag.retriever import retrieve_context, generate_response

    # ── Step 1: ML Prediction ──────────────────────────────────────────
    ml_result = predict_risk(patient_data)
    prob = ml_result["probability"]
    risk_label = ml_result["risk_label"]
    factors = ml_result["contributing_factors"]

    # ── Step 2: RAG Retrieval ──────────────────────────────────────────
    age = patient_data.get("age", "unknown")
    sex = "male" if patient_data.get("sex", 0) == 1 else "female"
    query = (
        f"Health risks for a {age}-year-old {sex} patient with "
        f"cholesterol {patient_data.get('chol', 'unknown')} mg/dL, "
        f"blood pressure {patient_data.get('trestbps', 'unknown')} mmHg, "
        f"BMI {patient_data.get('bmi', 'unknown')}, "
        f"stress level {patient_data.get('stress_level', 'unknown')}/10"
    )

    similar_patients = retrieve_context(query, top_k=5)
    rag_response = generate_response(query, similar_patients)

    # ── Step 3: Comparative analysis against retrieved cohort ──────────
    n_similar = len(similar_patients)
    n_high_risk = sum(
        1 for p in similar_patients if p["metadata"].get("target") == 1
    )
    similar_risk_pct = (n_high_risk / n_similar * 100) if n_similar else 0

    # Aggregate stats from similar patients for comparison
    avg_age = _safe_avg(similar_patients, "age")
    avg_bmi = _safe_avg(similar_patients, "bmi")
    avg_stress = _safe_avg(similar_patients, "stress_level")
    avg_anxiety = _safe_avg(similar_patients, "anxiety_score")

    # ── Step 4: Build explainable narrative ─────────────────────────────
    insight_text = _build_narrative(
        patient_data, ml_result, factors,
        n_similar, n_high_risk, similar_risk_pct,
        avg_bmi, avg_stress, avg_anxiety, avg_age,
    )

    # ── Step 5: Recommendations ────────────────────────────────────────
    recommendations = _generate_recommendations(ml_result, factors)

    return {
        "insight": insight_text,
        "ml_prediction": ml_result,
        "rag_context": rag_response,
        "similar_patients": [
            {
                "patient_id": p["patient_id"],
                "similarity": round(1 - p["distance"], 4),
                "risk": "High" if p["metadata"].get("target") == 1 else "Low",
            }
            for p in similar_patients
        ],
        "recommendations": recommendations,
        "risk_factors": factors,
    }


# ── Narrative builder  ────────────────────────────────────────────

def _safe_avg(patients: List[Dict], key: str) -> float:
    """Average a metadata field across retrieved patients, defaulting to 0."""
    vals = [p["metadata"].get(key, 0) for p in patients if p["metadata"].get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _build_narrative(
    data: Dict,
    ml: Dict,
    factors: List[Dict],
    n_similar: int,
    n_high_risk: int,
    similar_risk_pct: float,
    avg_bmi: float,
    avg_stress: float,
    avg_anxiety: float,
    avg_age: float,
) -> str:
    """
    Weave ML prediction, risk factors, and RAG-retrieved cohort stats
    into a single cohesive, clinician-style paragraph.

    Target style (from task spec):
        "High risk due to elevated BMI, low activity, and stress indicators."
    """
    prob = ml["probability"]
    risk_label = ml["risk_label"]

    # ---- opening sentence: verdict + top causes ----
    if factors:
        # Pick the top 3 most impactful phrases for the "due to" clause
        cause_phrases = _factor_phrases(data, factors[:5])
        due_clause = ", ".join(cause_phrases[:3])
        if len(cause_phrases) > 3:
            due_clause += f", and {len(cause_phrases) - 3} additional concern(s)"
        opening = f"{risk_label} (probability {prob*100:.1f}%) due to {due_clause}."
    else:
        opening = (
            f"{risk_label} (probability {prob*100:.1f}%). "
            "No major clinical risk factors were identified."
        )

    # ---- cohort comparison sentence ----
    cohort = (
        f"Among {n_similar} similar patient profiles retrieved from the knowledge base, "
        f"{n_high_risk} ({similar_risk_pct:.0f}%) had diagnosed heart disease"
    )
    # Add a relative-risk remark
    if similar_risk_pct >= 80:
        cohort += " -- a strongly elevated prevalence indicating heightened concern."
    elif similar_risk_pct >= 50:
        cohort += " -- above population baseline, warranting further investigation."
    else:
        cohort += "."

    # ---- mental-physical cross-domain sentence ----
    mental_physical = _mental_physical_sentence(data, factors)

    # ---- comparative deviation sentence ----
    deviations = []
    bmi = data.get("bmi", 0)
    if bmi and avg_bmi:
        delta = bmi - avg_bmi
        if abs(delta) > 2:
            direction = "above" if delta > 0 else "below"
            deviations.append(f"BMI is {abs(delta):.1f} points {direction} the cohort average ({avg_bmi:.1f})")
    stress = data.get("stress_level", 0)
    if stress and avg_stress:
        delta = stress - avg_stress
        if abs(delta) > 1.5:
            direction = "higher" if delta > 0 else "lower"
            deviations.append(f"stress level is {abs(delta):.1f} points {direction} than similar patients ({avg_stress:.1f}/10)")

    deviation_sentence = ""
    if deviations:
        deviation_sentence = "Compared to the retrieved cohort, this patient's " + "; ".join(deviations) + "."

    # ---- assemble ----
    parts = [opening, cohort]
    if mental_physical:
        parts.append(mental_physical)
    if deviation_sentence:
        parts.append(deviation_sentence)

    return " ".join(parts)


def _factor_phrases(data: Dict, factors: List[Dict]) -> List[str]:
    """
    Convert structured risk factors into short natural-language phrases
    suitable for a "due to X, Y, and Z" clause.
    """
    phrases = []
    for f in factors:
        name = f["factor"]
        val = f["value"]
        if name == "High cholesterol":
            phrases.append(f"elevated cholesterol ({val})")
        elif name == "Borderline cholesterol":
            phrases.append(f"borderline cholesterol ({val})")
        elif name == "Hypertension":
            phrases.append(f"hypertension (BP {val})")
        elif name == "Elevated blood pressure":
            phrases.append(f"elevated blood pressure ({val})")
        elif name == "Obesity":
            phrases.append(f"elevated BMI ({val})")
        elif name == "Overweight":
            phrases.append(f"overweight ({val})")
        elif name == "Low max heart rate":
            phrases.append(f"low exercise heart rate ({val})")
        elif name == "High stress":
            phrases.append(f"high stress indicators ({val})")
        elif "anxiety" in name.lower():
            phrases.append(f"{name.lower()} ({val})")
        elif name == "Poor sleep quality":
            phrases.append(f"poor sleep quality ({val})")
        elif name == "Sedentary lifestyle":
            phrases.append(f"low physical activity ({val})")
        elif name == "Advanced age":
            phrases.append(f"advanced age ({val})")
        else:
            phrases.append(name.lower())
    return phrases


def _mental_physical_sentence(data: Dict, factors: List[Dict]) -> str:
    """
    If both mental and physical risk factors are present, highlight the
    cross-domain interaction -- this is the novel contribution of the
    Digital Health Twin.
    """
    physical_flags = {"High cholesterol", "Hypertension", "Obesity",
                      "Overweight", "Low max heart rate", "Elevated blood pressure",
                      "Borderline cholesterol"}
    mental_flags = {"High stress", "Severe anxiety", "Moderate anxiety",
                    "Poor sleep quality", "Sedentary lifestyle"}

    has_physical = any(f["factor"] in physical_flags for f in factors)
    has_mental = any(f["factor"] in mental_flags for f in factors)

    if has_physical and has_mental:
        return (
            "Notably, this patient presents concurrent physical and mental health risk factors, "
            "which research shows can compound cardiovascular risk by 1.5-2x through "
            "stress-mediated inflammation and autonomic dysfunction."
        )
    elif has_mental and not has_physical:
        return (
            "While traditional cardiac biomarkers appear within acceptable ranges, "
            "elevated mental health indicators alone can significantly increase long-term "
            "cardiovascular risk through chronic stress pathways."
        )
    return ""


def _generate_recommendations(ml_result: Dict, factors: List[Dict]) -> List[str]:
    """Generate actionable health recommendations."""
    recs = []
    prob = ml_result["probability"]

    if prob > 0.7:
        recs.append("🔴 Urgent: Schedule comprehensive cardiovascular evaluation immediately.")
    elif prob > 0.5:
        recs.append("🟡 Schedule cardiovascular screening within 2 weeks.")

    for f in factors:
        if f["factor"] == "High cholesterol":
            recs.append("💊 Consult physician about cholesterol management (diet/statins).")
        elif f["factor"] == "Hypertension":
            recs.append("💊 Monitor blood pressure daily. Consider antihypertensive therapy.")
        elif f["factor"] == "Obesity":
            recs.append("🏃 Aim for gradual weight loss (0.5-1 kg/week) through diet and exercise.")
        elif f["factor"] == "High stress":
            recs.append("🧘 Implement stress management: meditation, therapy, or structured relaxation.")
        elif "anxiety" in f["factor"].lower():
            recs.append("🧠 Refer to mental health professional for anxiety assessment (GAD-7 screening).")
        elif f["factor"] == "Poor sleep quality":
            recs.append("😴 Address sleep hygiene. Consider sleep study if quality remains poor.")
        elif f["factor"] == "Sedentary lifestyle":
            recs.append("🚶 Start with 150 min/week moderate activity (walking, cycling).")

    if not recs:
        recs.append("✅ Continue healthy lifestyle. Annual checkup recommended.")

    return recs


if __name__ == "__main__":
    # Test with a sample patient
    sample = {
        "age": 58, "sex": 1, "cp": 2, "trestbps": 145, "chol": 260,
        "fbs": 1, "restecg": 0, "thalach": 130, "exang": 1, "oldpeak": 2.5,
        "slope": 1, "ca": 2, "thal": 2,
        "stress_level": 8.0, "sleep_quality": 3.5, "activity_hours": 1.0,
        "anxiety_score": 15, "bmi": 32.5
    }

    print("=" * 60)
    print("  INSIGHT ENGINE – Test Run")
    print("=" * 60)
    result = generate_insight(sample)
    print(f"\n🧠 Insight: {result['insight']}")
    print(f"\n📊 ML Result: {result['ml_prediction']['risk_label']} "
          f"({result['ml_prediction']['probability']*100:.1f}%)")
    print(f"\n🔍 RAG Context: {result['rag_context'][:200]}…")
    print(f"\n💡 Recommendations:")
    for r in result["recommendations"]:
        print(f"   {r}")
