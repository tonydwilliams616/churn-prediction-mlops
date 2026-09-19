"""
SageMaker script-mode training entry point.

This runs INSIDE a managed SageMaker training container, not on your laptop.
SageMaker's scikit-learn container calls this script automatically once the
training job starts, passing configuration via environment variables and
command-line arguments rather than the hardcoded local paths we used in
src/pipeline.py.

Deliberately self-contained: rather than importing from src/ (which would
require packaging that directory alongside this script and managing import
paths inside the container), this reimplements the same cleaning / feature
engineering / training logic directly. The steps are identical to
src/data.py, src/features.py, and src/train.py - just written for a
container environment instead of a local one.
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Hyperparameters - passed in by the launcher script (or later, by a
    # SageMaker hyperparameter tuning job). Defaults match the winning
    # configuration found in notebook 07.
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--max-features", type=str, default="log2")

    # SageMaker sets these environment variables automatically inside the
    # container. SM_MODEL_DIR is where anything we save gets picked up and
    # uploaded to S3 as model.tar.gz once training finishes. SM_CHANNEL_TRAIN
    # is where SageMaker downloads our "train" input channel to before the
    # script runs - this is how the raw CSV from S3 actually ends up on disk
    # inside the container.
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

    return parser.parse_args()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Same fix as src/data.py: TotalCharges hidden blanks -> 0."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same steps as src/features.py: drop ID, encode target, one-hot encode."""
    df = df.copy()
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    categorical_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


if __name__ == "__main__":
    args = parse_args()

    csv_path = os.path.join(args.train, "telco_churn.csv")
    print(f"Reading training data from {csv_path}")
    df = pd.read_csv(csv_path)

    df = clean_data(df)
    df = engineer_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    print(f"Training on {X.shape[0]} rows, {X.shape[1]} features.")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=42,
    )
    model.fit(X, y)

    # Anything written to model_dir gets automatically packaged into
    # model.tar.gz and uploaded to S3 when the training job completes.
    joblib.dump(model, os.path.join(args.model_dir, "churn_model.joblib"))
    joblib.dump(X.columns.tolist(), os.path.join(args.model_dir, "feature_columns.joblib"))

    print(f"Model and feature columns saved to {args.model_dir}")
