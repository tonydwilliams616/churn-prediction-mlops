"""
Pipeline Stage 3: Evaluation.

Runs as a SageMaker Processing job. Loads the trained model and the
held-out test set (untouched since the processing stage split it off) and
writes a metrics.json report. This report is what Stage 4 (the quality
gate, handled by the orchestrating script) reads to decide whether the
model is good enough to register.

Handles the model arriving as either a raw .tar.gz (as SageMaker training
jobs produce) or an already-extracted directory, since Processing job
inputs don't auto-extract tar.gz files the way endpoint deployment does.
"""

import json
import os
import tarfile

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_INPUT_DIR = "/opt/ml/processing/model"
TEST_INPUT_DIR = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/evaluation"


def load_model_artifacts(model_dir: str):
    """Extracts model.tar.gz if present, then loads the joblib files."""
    tar_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        print(f"Extracting {tar_path}")
        with tarfile.open(tar_path) as tar:
            tar.extractall(model_dir)

    model = joblib.load(os.path.join(model_dir, "churn_model.joblib"))
    feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.joblib"))
    return model, feature_columns


if __name__ == "__main__":
    model, feature_columns = load_model_artifacts(MODEL_INPUT_DIR)

    test_path = os.path.join(TEST_INPUT_DIR, "test.csv")
    print(f"Reading test data from {test_path}")
    test_df = pd.read_csv(test_path)

    X_test = test_df.reindex(columns=feature_columns, fill_value=0)
    y_test = test_df["Churn"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print("Evaluation metrics:", json.dumps(metrics, indent=2))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics written to {OUTPUT_DIR}/metrics.json")
