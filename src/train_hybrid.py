"""
Creating a Hybrid predictive maintenance model.

Idea:
- Detect unusual behaviour using Isolation Forest(anomaly detection)
- Use anomaly score as an extra feature
- Train XGBoost on enriched feature space
- Basically Anomaly Detection --> Classification
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import IsolationForest

from xgboost import XGBClassifier

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "raw" / "predictive_maintenance_dataset.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "hybrid_xgb_model.joblib"

TARGET_COL = "failure"
DATE_COL = "date"
DEVICE_COL = "device"

def main() -> None:
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    
    # Time feature engineering
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df["year"] = df[DATE_COL].dt.year
        df["month"] = df[DATE_COL].dt.month
        df["day"] = df[DATE_COL].dt.day
        
    # Separate features and target 
    drop_cols = [DATE_COL, DEVICE_COL]
    X = df.drop(columns=[c for c in drop_cols + [TARGET_COL] if c in df.columns])
    y = df[TARGET_COL]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    
    # Train Isolation Forest on Normal data only 
    normal_data = X_train[y_train == 0]

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.001,
        random_state=42,
    )

    iso_forest.fit(normal_data)

    # Generate anomaly scores 
    X_train["anomaly_score"] = -iso_forest.decision_function(X_train)
    X_test["anomaly_score"] = -iso_forest.decision_function(X_test)
    
    # Handle class imbalance for XGBoost 
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Initialize XGBoost classifier 
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    # Train hybrid model 
    model.fit(X_train, y_train)
    
    # Prediction probabilities 
    y_proba = model.predict_proba(X_test)[:, 1]

    # Search best threshold based on recall constraint 
    from sklearn.metrics import precision_score, recall_score
    import numpy as np

    thresholds = np.linspace(0.01, 0.99, 50)

    best_threshold = 0.5
    best_recall = 0.0
    min_precision = 0.10   # acceptable false alarm level

    for t in thresholds:
        y_pred_temp = (y_proba >= t).astype(int)

        precision = precision_score(y_test, y_pred_temp, zero_division=0)
        recall = recall_score(y_test, y_pred_temp, zero_division=0)

        if precision >= min_precision and recall > best_recall:
            best_recall = recall
            best_threshold = t

    print(f"\nBest threshold (min precision=0.10): {best_threshold:.2f}")
    
    # Final prediction using best threshold
    y_pred = (y_proba >= best_threshold).astype(int)

    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

    print("\n=== HYBRID MODEL REPORT ===")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    
    # Ensure models directory exists 
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained hybrid model 
    joblib.dump(model, MODEL_PATH)

    print(f"\nHybrid model saved to: {MODEL_PATH}")
    
if __name__ == "__main__":
    main()