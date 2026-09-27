"""
Pipeline Stage 2: Training.

Unlike sagemaker_job/entry_point.py (which cleans and encodes a raw CSV
itself), this script expects data that's ALREADY been through the
processing stage - already numeric, already has a Churn column, ready to
train on directly. This split exists because in a real pipeline, each
stage should do exactly one job; re-doing cleaning inside the training
stage would mean two places could disagree about what "clean" means.
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--max-features", type=str, default="log2")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    csv_path = os.path.join(args.train, "train.csv")
    print(f"Reading pre-processed training data from {csv_path}")
    df = pd.read_csv(csv_path)

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

    joblib.dump(model, os.path.join(args.model_dir, "churn_model.joblib"))
    joblib.dump(X.columns.tolist(), os.path.join(args.model_dir, "feature_columns.joblib"))

    print(f"Model and feature columns saved to {args.model_dir}")
