# src/train.py
"""
Training a baseline predictive maintenance model.

Pipeline overview:
- Load raw dataset tracked by DVC
- Parse date and create time-based features
- Train a baseline classifier
- Save trained model artifact to /models
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib


# ---- Paths (keep them relative to repo root) ----
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "raw" / "predictive_maintenance_dataset.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "baseline_logreg.joblib"

TARGET_COL = "failure"
DATE_COL = "date"
DEVICE_COL = "device"


def main() -> None:
    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    # Basic time feature engineering
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df["year"] = df[DATE_COL].dt.year
        df["month"] = df[DATE_COL].dt.month
        df["day"] = df[DATE_COL].dt.day

    # Drop non-numeric / identifier columns
    drop_cols = [DATE_COL, DEVICE_COL]
    X = df.drop(columns=[c for c in drop_cols + [TARGET_COL] if c in df.columns])
    y = df[TARGET_COL]

    # Performing Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline (scaling + logistic regression)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    # Training the model
    model.fit(X_train, y_train)

    # Evaluation
    y_proba = model.predict_proba(X_test)[:, 1]

    threshold = 0.20
    y_pred = (y_proba >= threshold).astype(int)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    
    # Ensure presence of models folder exists 
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Saving model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()