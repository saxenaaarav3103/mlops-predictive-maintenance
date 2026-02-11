"""
Training an XGBoost predictive maintenance model.

Purpose:
- Train a stronger nonlinear model than logistic regression
- Keep preprocessing identical to baseline for fair comparison
- Save trained model for later MLflow + deployment stages
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier


# Paths (same structure as baseline) 
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "raw" / "predictive_maintenance_dataset.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "xgb_model.joblib"

TARGET_COL = "failure"
DATE_COL = "date"
DEVICE_COL = "device"

def main() -> None:
    # Load dataset 
    df = pd.read_csv(DATA_PATH)

    # Time feature engineering (same as baseline) 
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df["year"] = df[DATE_COL].dt.year
        df["month"] = df[DATE_COL].dt.month
        df["day"] = df[DATE_COL].dt.day

    # Separate features and target 
    drop_cols = [DATE_COL, DEVICE_COL]
    X = df.drop(columns=[c for c in drop_cols + [TARGET_COL] if c in df.columns])
    y = df[TARGET_COL]

    # Train-test split (same settings as baseline) 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    
    # Handle class imbalance for XGBoost 
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Initialize XGBoost classifier 
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    # Training the model
    model.fit(X_train, y_train)
    
    # Prediction probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Alarm budget thresholding (1% inspection capacity)
    alarm_fraction = 0.01
    n_alerts = int(len(y_proba) * alarm_fraction)

    # Selecting Highest-risk machines
    sorted_indices = np.argsort(y_proba)[::-1]
    alert_indices = sorted_indices[:n_alerts]

    # Creating Binary prediction vector
    y_pred_budget = np.zeros_like(y_proba, dtype=int)
    y_pred_budget[alert_indices] = 1

    # Metrics
    precision_budget = precision_score(y_test, y_pred_budget, zero_division=0)
    recall_budget = recall_score(y_test, y_pred_budget, zero_division=0)

    print(f"\n=== Alarm-Budget Evaluation (Top {alarm_fraction*100:.0f}% alerts) ===")
    print(f"Precision: {precision_budget:.4f}")
    print(f"Recall: {recall_budget:.4f}")

    cm_budget = confusion_matrix(y_test, y_pred_budget)
    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm_budget)
    
    # Threshold tuning: maximize recall subject to minimum precision
    thresholds = np.linspace(0.01, 0.99, 200)

    min_precision = 0.20
    best_threshold = 0.5
    best_recall = -1.0
    best_precision = -1.0

    for t in thresholds:
        y_pred_temp = (y_proba >= t).astype(int)

        precision = precision_score(y_test, y_pred_temp, zero_division=0)
        recall = recall_score(y_test, y_pred_temp, zero_division=0)

        # keep only thresholds that satisfy precision constraint, then maximize recall
        if precision >= min_precision and recall > best_recall:
            best_recall = recall
            best_precision = precision
            best_threshold = t

    print(
        f"\nBest threshold (min precision={min_precision:.2f}): {best_threshold:.2f} | "
        f"precision={best_precision:.3f}, recall={best_recall:.3f}"
    )

    # Default threshold prediction (0.5)
    y_pred = (y_proba >= best_threshold).astype(int)

    print("\n=== XGBoost Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    # Ensuring 'models' folder exists 
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Saving model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
