"""
ML Training Pipeline
=====================
Trains baseline (Logistic Regression, Random Forest) and advanced
(XGBoost, PyTorch Neural Network) models on the augmented heart-disease
dataset.  Evaluates each model and saves artefacts for serving.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE.parent / "data" / "heart.csv"
MODELS_DIR = BASE
METRICS_PATH = BASE / "metrics.json"

# Features to use (exclude patient_id, patient_note, target)
FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    "stress_level", "sleep_quality", "activity_hours",
    "anxiety_score", "bmi"
]
TARGET_COL = "target"


def load_data():
    """Load CSV and split into train / test."""
    print("=" * 60)
    print("  DIGITAL HEALTH TWIN – ML TRAINING PIPELINE")
    print("=" * 60)
    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset: {len(df)} samples, {len(FEATURE_COLS)} features")
    print(f"Class distribution:\n{df[TARGET_COL].value_counts().to_dict()}\n")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("✓ Scaler saved.\n")

    return X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler


def evaluate(name, model, X_test, y_test, metrics_dict):
    """Evaluate and print metrics."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"── {name} ──")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Confusion Matrix: {cm}\n")

    metrics_dict[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
    }
    return metrics_dict


# ── BASELINE 1: Logistic Regression ───────────────────────────────────
def train_logistic_regression(X_train, X_test, y_train, y_test, metrics):
    print("[1/4] Training Logistic Regression …")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODELS_DIR / "logistic_regression.pkl")
    return evaluate("Logistic Regression", model, X_test, y_test, metrics)


# ── BASELINE 2: Random Forest ─────────────────────────────────────────
def train_random_forest(X_train, X_test, y_train, y_test, metrics):
    print("[2/4] Training Random Forest …")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODELS_DIR / "random_forest.pkl")
    return evaluate("Random Forest", model, X_test, y_test, metrics)


# ── ADVANCED: XGBoost ─────────────────────────────────────────────────
def train_xgboost(X_train, X_test, y_train, y_test, metrics):
    print("[3/4] Training XGBoost …")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODELS_DIR / "xgboost_model.pkl")
    return evaluate("XGBoost", model, X_test, y_test, metrics)


# ── BONUS: PyTorch Neural Network ─────────────────────────────────────
def train_pytorch_nn(X_train, X_test, y_train, y_test, metrics):
    """Simple feed-forward neural network using PyTorch."""
    print("[4/4] Training PyTorch Neural Network …")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        class HealthNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.net(x)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {device}")

        X_tr = torch.FloatTensor(X_train).to(device)
        y_tr = torch.FloatTensor(y_train).to(device)
        X_te = torch.FloatTensor(X_test).to(device)

        dataset = TensorDataset(X_tr, y_tr)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = HealthNet(X_train.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        # Training loop
        model.train()
        for epoch in range(100):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 25 == 0:
                print(f"    Epoch {epoch+1}/100  Loss: {epoch_loss/len(loader):.4f}")

        # Evaluate
        model.eval()
        with torch.no_grad():
            y_prob = model(X_te).squeeze().cpu().numpy()
            y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        print(f"── PyTorch NN ──")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  Confusion Matrix: {cm}\n")

        metrics["PyTorch NN"] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
        }

        # Save model
        torch.save(model.state_dict(), MODELS_DIR / "pytorch_nn.pth")
        # Save architecture info for loading later
        joblib.dump({"input_dim": X_train.shape[1]}, MODELS_DIR / "pytorch_nn_config.pkl")
        print("✓ PyTorch model saved.\n")

    except ImportError:
        print("  ⚠ PyTorch not available – skipping NN.\n")

    return metrics


# ── Feature Importance (SHAP) ─────────────────────────────────────────
def compute_shap(X_test, model_path="xgboost_model.pkl"):
    """Compute SHAP values for explainability."""
    print("Computing SHAP feature importance …")
    try:
        import shap
        model = joblib.load(MODELS_DIR / model_path)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        importance = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(FEATURE_COLS, importance.round(4).tolist()))
        sorted_fi = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        print("  Top features:")
        for k, v in list(sorted_fi.items())[:5]:
            print(f"    {k}: {v:.4f}")
        print()
        return sorted_fi
    except Exception as e:
        print(f"  ⚠ SHAP computation failed: {e}\n")
        return {}


def main():
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = load_data()
    metrics = {}

    # Baseline models (use scaled data)
    metrics = train_logistic_regression(X_train_s, X_test_s, y_train, y_test, metrics)
    metrics = train_random_forest(X_train, X_test, y_train, y_test, metrics)

    # Advanced models
    metrics = train_xgboost(X_train, X_test, y_train, y_test, metrics)
    metrics = train_pytorch_nn(X_train_s, X_test_s, y_train, y_test, metrics)

    # SHAP explainability
    shap_importance = compute_shap(X_test)
    if shap_importance:
        metrics["feature_importance_shap"] = shap_importance

    # Save feature columns list for inference
    joblib.dump(FEATURE_COLS, MODELS_DIR / "feature_cols.pkl")

    # Save all metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ All metrics saved → {METRICS_PATH}")
    print("✓ Training pipeline complete!")


if __name__ == "__main__":
    main()
