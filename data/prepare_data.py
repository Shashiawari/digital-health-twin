"""
Data Preparation Module
========================
Downloads the UCI Heart Disease dataset and augments it with
synthetic mental-health features to demonstrate physical + mental
health integration in the Digital Health Twin system.
"""

import pandas as pd
import numpy as np
import os

# ── UCI Heart Disease dataset (Cleveland) ──────────────────────────────
# Original 14-attribute subset used in most ML research.
# Source: https://archive.ics.uci.edu/dataset/45/heart+disease

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def download_heart_dataset(save_path: str = "data/heart.csv") -> pd.DataFrame:
    """Download the Cleveland heart-disease dataset from UCI."""
    print("[1/4] Downloading UCI Heart Disease dataset ...")
    df = pd.read_csv(UCI_URL, header=None, names=COLUMN_NAMES, na_values="?")

    # Binarise multi-class target -> 0 = no disease, 1 = disease
    df["target"] = (df["target"] > 0).astype(int)

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    print(f"   -> {len(df)} samples, {len(df.columns)} features loaded.")
    return df


def augment_with_mental_health(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic mental-health features correlated with cardiac risk.
    This simulates a real-world scenario where wearable / self-reported
    mental health data enriches the patient profile.
    """
    print("[2/4] Augmenting with synthetic mental-health features ...")
    np.random.seed(42)
    n = len(df)

    # Stress level (1-10): higher for positive-target patients
    base_stress = np.random.normal(5, 1.5, n)
    df["stress_level"] = np.clip(
        base_stress + df["target"] * np.random.normal(2, 0.5, n), 1, 10
    ).round(1)

    # Sleep quality (1-10): lower for positive-target patients
    base_sleep = np.random.normal(7, 1.2, n)
    df["sleep_quality"] = np.clip(
        base_sleep - df["target"] * np.random.normal(1.5, 0.5, n), 1, 10
    ).round(1)

    # Physical activity (hours/week): lower for positive-target
    base_activity = np.random.normal(5, 2, n)
    df["activity_hours"] = np.clip(
        base_activity - df["target"] * np.random.normal(2, 0.8, n), 0, 15
    ).round(1)

    # Anxiety score (GAD-7 inspired, 0-21): higher for positive-target
    base_anxiety = np.random.normal(7, 3, n)
    df["anxiety_score"] = np.clip(
        base_anxiety + df["target"] * np.random.normal(4, 1.5, n), 0, 21
    ).round(0).astype(int)

    # BMI: synthesised from age + target correlation
    base_bmi = np.random.normal(26, 4, n)
    df["bmi"] = np.clip(
        base_bmi + df["target"] * np.random.normal(3, 1, n), 16, 45
    ).round(1)

    print(f"   -> Added 5 mental/lifestyle features. Total: {len(df.columns)} columns.")
    return df


def generate_patient_notes(df: pd.DataFrame) -> pd.DataFrame:
    """Generate natural-language patient summaries for RAG embedding."""
    print("[3/4] Generating patient narrative notes ...")

    def _make_note(row):
        sex_str = "Male" if row["sex"] == 1 else "Female"
        cp_map = {0: "typical angina", 1: "atypical angina",
                  2: "non-anginal pain", 3: "asymptomatic"}
        cp_str = cp_map.get(int(row["cp"]), "unknown")
        risk = "positive" if row["target"] == 1 else "negative"

        note = (
            f"Patient is a {int(row['age'])}-year-old {sex_str} presenting with "
            f"{cp_str} chest pain. Resting blood pressure: {int(row['trestbps'])} mmHg. "
            f"Serum cholesterol: {int(row['chol'])} mg/dL. "
            f"Fasting blood sugar {'>' if row['fbs'] else '<='} 120 mg/dL. "
            f"Maximum heart rate achieved: {int(row['thalach'])} bpm. "
            f"Exercise-induced angina: {'yes' if row['exang'] else 'no'}. "
            f"ST depression (oldpeak): {row['oldpeak']:.1f}. "
            f"BMI: {row['bmi']:.1f}. "
            f"Stress level: {row['stress_level']:.0f}/10. "
            f"Sleep quality: {row['sleep_quality']:.0f}/10. "
            f"Weekly physical activity: {row['activity_hours']:.1f} hours. "
            f"Anxiety score (GAD-7): {int(row['anxiety_score'])}/21. "
            f"Heart disease diagnosis: {risk}."
        )
        return note

    df["patient_note"] = df.apply(_make_note, axis=1)
    print("   -> Patient notes generated.")
    return df


def main():
    os.makedirs("data", exist_ok=True)

    df = download_heart_dataset()
    df = augment_with_mental_health(df)
    df = generate_patient_notes(df)

    # Add patient IDs
    df.insert(0, "patient_id", [f"P{str(i).zfill(4)}" for i in range(len(df))])

    save_path = os.path.join(os.path.dirname(__file__), "heart.csv")
    df.to_csv(save_path, index=False)
    print(f"[4/4] Dataset saved -> {save_path}  ({len(df)} rows, {len(df.columns)} cols)")
    print("\nSample row:")
    print(df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
