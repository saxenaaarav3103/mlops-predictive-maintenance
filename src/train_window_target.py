"""
Expanding the prediction window from 1 day to 14 days.

We are trying to predict whether a machine will fail in the next 14 days.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, fbeta_score
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
WINDOW_SIZE = 14  # Predict failure within next 14 days


def add_temporal_features(df):
    metric_cols = [f"metric{i}" for i in range(1, 10) if f"metric{i}" in df.columns and f"metric{i}" != "metric8"]

    df["dayofweek"] = df[DATE_COL].dt.dayofweek

    for col in metric_cols:
        device_series = df.groupby(DEVICE_COL)[col]

        df[f"{col}_lag1"] = device_series.shift(1)
        df[f"{col}_lag3"] = device_series.shift(3)
        df[f"{col}_delta1"] = df[col] - df[f"{col}_lag1"]

        prior_values = device_series.shift(1)
        device_groups = prior_values.groupby(df[DEVICE_COL])
        
        df[f"{col}_roll3_mean"] = (
            device_groups.rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"{col}_roll7_mean"] = (
            device_groups.rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"{col}_roll7_std"] = (
            device_groups.rolling(window=7, min_periods=1).std().reset_index(level=0, drop=True)
        )
        # New: Longer window to capture baseline
        df[f"{col}_roll14_mean"] = (
            device_groups.rolling(window=14, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        # New: Volatility and relative change
        df[f"{col}_relative_change"] = df[col] / (df[f"{col}_roll7_mean"] + 1e-5)

    feature_cols = [c for c in df.columns if c not in [DATE_COL, DEVICE_COL, TARGET_COL, "failure_window"]]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    return df


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

    # Calendar + temporal feature engineering using past values only.
    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["day"] = df[DATE_COL].dt.day
    df = add_temporal_features(df)

    # Drop original target and non-features
    idx_cols = [DATE_COL, DEVICE_COL, "metric8", TARGET_COL]
    X = df.drop(columns=[c for c in idx_cols + ["failure_window"] if c in df.columns])
    y = df["failure_window"]

    return df, X, y


def time_aware_split(df, X, y):
    ordered_idx = df.sort_values(DATE_COL).index
    n_samples = len(ordered_idx)

    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    train_idx = ordered_idx[:train_end]
    val_idx = ordered_idx[train_end:val_end]
    test_idx = ordered_idx[val_end:]

    return (
        X.loc[train_idx],
        X.loc[val_idx],
        X.loc[test_idx],
        y.loc[train_idx],
        y.loc[val_idx],
        y.loc[test_idx],
    )


def find_best_threshold(y_true, y_proba, min_precision=0.20):
    best_threshold = 0.5
    best_recall = 0.0
    best_precision = 0.0

    for threshold in np.linspace(0.01, 0.99, 100):
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        if precision >= min_precision and recall >= best_recall:
            best_recall = recall
            best_precision = precision
            best_threshold = threshold

    return best_threshold, best_precision, best_recall


def main():
    df, X, y = prepare_data()

    # Chronological train/validation/test split
    X_train, X_val, X_test, y_train, y_val, y_test = time_aware_split(df, X, y)

    print("\nTraining XGBoost on Windowed Target...")
    # Using SMOTE still, but less aggressive maybe needed now that we have 7x more positives

    model = ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.5, random_state=42)),  # Balanced-ish
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            min_child_weight=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,  # Let SMOTE handle it
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)

    print("\nScanning thresholds on validation set...")
    y_val_proba = model.predict_proba(X_val)[:, 1]
    best_threshold, best_precision, best_recall = find_best_threshold(y_val, y_val_proba, min_precision=0.25)

    print(
        f"\nValidation Threshold: {best_threshold:.4f} "
        f"(precision={best_precision:.4f}, recall={best_recall:.4f})"
    )

    y_proba = model.predict_proba(X_test)[:, 1]
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
