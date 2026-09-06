"""
Model training and persistence for the Telco churn project.

This is the notebook 07/08 logic, extracted: train a Random Forest using the
hyperparameters we already found via RandomizedSearchCV, and save it to disk in
a reusable format.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Winning hyperparameters found via RandomizedSearchCV in notebook 07.
# Hardcoded here deliberately - the search itself is a one-time exploratory
# step, not something we want to re-run every time we retrain.
BEST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 4,
    "max_features": "log2",
    "random_state": 42,
}


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Train a Random Forest on the given features and target using BEST_PARAMS."""
    model = RandomForestClassifier(**BEST_PARAMS)
    model.fit(X, y)
    return model


def save_model(model: RandomForestClassifier, feature_columns: list, model_dir: str) -> None:
    """
    Save the trained model and its expected feature column order to model_dir.

    Saving the column order alongside the model guards against a real production
    bug: feeding the model a row with columns in a different order than it was
    trained on, which can silently produce garbage predictions.
    """
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "churn_model.joblib"))
    joblib.dump(feature_columns, os.path.join(model_dir, "feature_columns.joblib"))


def load_model(model_dir: str):
    """Load a previously saved model and its expected feature columns."""
    model = joblib.load(os.path.join(model_dir, "churn_model.joblib"))
    feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.joblib"))
    return model, feature_columns
