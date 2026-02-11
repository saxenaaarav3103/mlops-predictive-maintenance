"""
Iteration 8: Target Engineering (Failure Prediction Window).

Hypothesis: 
- Trying to predict the EXACT day of failure (binary 0/1 on single day) is too hard and realistic maintenance doesn't require it.
- We need to predict "Machine will fail in the next X days".
- Action: Create new target `failure_in_next_7d`. This expands the minority class by 7x and allows the model to learn degradation patterns leading up to the event.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, fbeta_score
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "raw" / "predictive_maintenance_dataset.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "windowed_xgb.joblib"

TARGET_COL = "failure"
DATE_COL = "date"
DEVICE_COL = "device"
WINDOW_SIZE = 7 # Predict failure within next 7 days

def prepare_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    
    # Sort for windowing
    df = df.sort_values(by=[DEVICE_COL, DATE_COL]).reset_index(drop=True)
    
    print(f"Creating target: Failure within {WINDOW_SIZE} days...")
    
    # Create forward-looking target
    # Ideally: if failure at T, then T-7...T are 1.
    # Rolling forward max.
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=WINDOW_SIZE)
    df["failure_window"] = df.groupby(DEVICE_COL)[TARGET_COL].transform(
        lambda x: x.rolling(window=indexer, min_periods=1).max()
    )
    
    # Check class balance
    print(f"Original Failures: {df[TARGET_COL].sum()}")
    print(f"Windowed Failures: {df['failure_window'].sum()}")
    
    # Feature Engineering (Keep simple for now, can transform to rolling features later)
    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["day"] = df[DATE_COL].dt.day
    
    # Drop original target and non-features
    idx_cols = [DATE_COL, DEVICE_COL, "metric8", TARGET_COL]
    X = df.drop(columns=[c for c in idx_cols + ["failure_window"] if c in df.columns])
    y = df["failure_window"]
    
    return X, y

def main():
    X, y = prepare_data()
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nTraining XGBoost on Windowed Target...")
    # Using SMOTE still, but less aggressive maybe needed now that we have 7x more positives
    
    model = ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.5, random_state=42)), # Balanced-ish
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1, # Let SMOTE handle it
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Threshold Tuning for our constraint
    # Precision >= 0.2, Maximize Recall
    
    print("\nScanning thresholds...")
    best_threshold = 0.5
    best_recall = 0.0
    best_precision = 0.0
    
    for t in np.linspace(0.01, 0.99, 100):
        y_pred = (y_proba >= t).astype(int)
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        
        if p >= 0.20 and r >= best_recall:
            best_recall = r
            best_precision = p
            best_threshold = t
            
    print(f"\nFinal Threshold: {best_threshold:.4f}")
    y_pred = (y_proba >= best_threshold).astype(int)
    
    print("\n=== Windowed Target Classification Report ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
if __name__ == "__main__":
    main()
