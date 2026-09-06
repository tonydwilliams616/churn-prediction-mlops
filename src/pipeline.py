"""
End-to-end training pipeline for the Telco churn project.

This is the "whole story" of notebooks 01 through 08, condensed into one script
that runs unattended: raw CSV in, trained model saved to disk out. No manual
cell-by-cell execution, no re-reading markdown explanations - just the pipeline
doing the same work automatically.

Run it from the project root with:
    python -m src.pipeline

This is the shape of thing that eventually runs inside a SageMaker Processing
or Training job instead of on your laptop - the same logic, just triggered by
AWS infrastructure instead of by you typing a command.
"""

import argparse

from src.data import load_raw_data, clean_data
from src.features import engineer_features, split_features_and_target
from src.train import train_model, save_model


def run_pipeline(data_path: str, model_dir: str) -> None:
    print(f"Loading raw data from {data_path} ...")
    df = load_raw_data(data_path)
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

    print("Cleaning data ...")
    df = clean_data(df)

    print("Engineering features ...")
    df = engineer_features(df)
    X, y = split_features_and_target(df)
    print(f"  Final feature matrix: {X.shape[0]} rows, {X.shape[1]} columns.")

    print("Training model ...")
    model = train_model(X, y)

    print(f"Saving model to {model_dir} ...")
    save_model(model, X.columns.tolist(), model_dir)

    print("Pipeline complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Telco churn model end-to-end.")
    parser.add_argument(
        "--data-path",
        default="data/telco_churn.csv",
        help="Path to the raw Telco churn CSV (default: data/telco_churn.csv)",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory to save the trained model into (default: models)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.data_path, args.model_dir)
