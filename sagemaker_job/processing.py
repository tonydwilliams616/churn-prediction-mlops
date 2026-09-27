"""
Pipeline Stage 1: Processing.

Runs inside a SageMaker Processing job. Reads the raw CSV, applies the same
cleaning and feature engineering as src/data.py and src/features.py
(self-contained here for the same reason as entry_point.py - avoiding the
complexity of packaging src/ into a container), and produces a train/test
split ready for the next pipeline stage.

Processing jobs don't use the SM_CHANNEL_* environment variable convention
that training jobs do - instead, input/output locations are fixed local
paths that the orchestrating script configures via the CreateProcessingJob
API's ProcessingInputs/ProcessingOutputs.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIR = "/opt/ml/processing/input"
TRAIN_OUTPUT_DIR = "/opt/ml/processing/train"
TEST_OUTPUT_DIR = "/opt/ml/processing/test"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


if __name__ == "__main__":
    input_path = os.path.join(INPUT_DIR, "telco_churn.csv")
    print(f"Reading raw data from {input_path}")
    df = pd.read_csv(input_path)

    df = clean_data(df)
    df = engineer_features(df)
    print(f"Processed shape: {df.shape}")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Churn"]
    )
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    train_df.to_csv(os.path.join(TRAIN_OUTPUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(TEST_OUTPUT_DIR, "test.csv"), index=False)

    print("Processing complete.")
