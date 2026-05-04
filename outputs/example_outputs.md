# Example Outputs

## POST /predict

### Request
```json
{
    "age": 58, "sex": 1, "cp": 2, "trestbps": 145,
    "chol": 260, "fbs": 1, "restecg": 0, "thalach": 130,
    "exang": 1, "oldpeak": 2.5, "slope": 1, "ca": 2, "thal": 2,
    "stress_level": 8.0, "sleep_quality": 3.5,
    "activity_hours": 1.0, "anxiety_score": 15, "bmi": 32.5
}
```

### Response
```json
{
    "prediction": 1,
    "risk_label": "High Risk",
    "risk_level": "high",
    "probability": 0.8734,
    "confidence": 0.8734,
    "model_used": "XGBClassifier",
    "contributing_factors": [
        {"factor": "High cholesterol", "value": "260 mg/dL", "severity": "high", "detail": "Cholesterol above 240 mg/dL significantly increases cardiovascular risk."},
        {"factor": "Hypertension", "value": "145 mmHg", "severity": "high", "detail": "Resting BP above 140 mmHg indicates hypertension."},
        {"factor": "Obesity", "value": "BMI 32.5", "severity": "high", "detail": "BMI above 30 classified as obese."},
        {"factor": "High stress", "value": "8/10", "severity": "high", "detail": "Elevated stress increases cardiovascular risk."},
        {"factor": "Severe anxiety", "value": "GAD-7: 15/21", "severity": "high", "detail": "Severe anxiety strongly correlates with cardiac events."},
        {"factor": "Poor sleep quality", "value": "3.5/10", "severity": "high", "detail": "Poor sleep strongly linked to cardiovascular disease."},
        {"factor": "Sedentary lifestyle", "value": "1.0 hrs/week", "severity": "moderate", "detail": "Less than 2 hours/week of activity increases risk."}
    ]
}
```

---

## POST /ask

### Request
```json
{
    "query": "What health risks does a 55-year-old male with high cholesterol have?",
    "top_k": 5
}
```

### Response
```json
{
    "query": "What health risks does a 55-year-old male with high cholesterol have?",
    "response": "Based on analysis of 5 similar patient profiles, 4/5 (80%) have heart disease. Key risk factors include: elevated BMI (29.8), high stress levels (7.2/10). Average age in this cohort: 56 years. Recommend comprehensive cardiovascular screening and mental health assessment.",
    "sources": [
        {"patient_id": "P0045", "similarity": 0.9231, "age": 56, "risk": "High"},
        {"patient_id": "P0112", "similarity": 0.9105, "age": 54, "risk": "High"},
        {"patient_id": "P0089", "similarity": 0.8942, "age": 57, "risk": "High"},
        {"patient_id": "P0201", "similarity": 0.8834, "age": 55, "risk": "High"},
        {"patient_id": "P0023", "similarity": 0.8721, "age": 53, "risk": "Low"}
    ],
    "num_sources": 5
}
```

---

## POST /insight

### Request
```json
{
    "age": 62, "sex": 1, "cp": 3, "trestbps": 155,
    "chol": 290, "fbs": 1, "restecg": 2, "thalach": 115,
    "exang": 1, "oldpeak": 3.1, "slope": 2, "ca": 3, "thal": 3,
    "stress_level": 9, "sleep_quality": 3, "activity_hours": 1,
    "anxiety_score": 17, "bmi": 33.5
}
```

### Response
```json
{
    "insight": "High Risk (91.2% probability). Critical factors: high cholesterol, hypertension, obesity, high stress, severe anxiety, poor sleep quality. Additional concerns: sedentary lifestyle, advanced age, low max heart rate. Among 5 similar patient profiles, 100% had diagnosed heart disease.",
    "ml_prediction": {
        "prediction": 1,
        "risk_label": "High Risk",
        "risk_level": "critical",
        "probability": 0.912,
        "confidence": 0.912,
        "model_used": "XGBClassifier",
        "contributing_factors": [...]
    },
    "rag_context": "Based on analysis of 5 similar patient profiles, 5/5 (100%) have heart disease. Key risk factors include: elevated BMI (32.1), high stress levels (8.4/10), significant anxiety (16/21). Average age in this cohort: 61 years. Recommend comprehensive cardiovascular screening and mental health assessment.",
    "similar_patients": [
        {"patient_id": "P0156", "similarity": 0.9456, "risk": "High"},
        {"patient_id": "P0078", "similarity": 0.9312, "risk": "High"},
        {"patient_id": "P0234", "similarity": 0.9198, "risk": "High"},
        {"patient_id": "P0167", "similarity": 0.9087, "risk": "High"},
        {"patient_id": "P0092", "similarity": 0.8945, "risk": "High"}
    ],
    "recommendations": [
        "🔴 Urgent: Schedule comprehensive cardiovascular evaluation immediately.",
        "💊 Consult physician about cholesterol management (diet/statins).",
        "💊 Monitor blood pressure daily. Consider antihypertensive therapy.",
        "🏃 Aim for gradual weight loss (0.5-1 kg/week) through diet and exercise.",
        "🧘 Implement stress management: meditation, therapy, or structured relaxation.",
        "🧠 Refer to mental health professional for anxiety assessment (GAD-7 screening).",
        "😴 Address sleep hygiene. Consider sleep study if quality remains poor.",
        "🚶 Start with 150 min/week moderate activity (walking, cycling)."
    ],
    "risk_factors": [...]
}
```
