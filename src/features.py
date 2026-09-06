"""
Feature engineering for the Telco churn project.

This is the notebook 04 logic, extracted: drop the identifier column, encode the
target, and one-hot encode the remaining categorical columns.
"""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the cleaned dataset into a fully numeric table ready for modeling.

    - Drops customerID (pure identifier, no predictive signal).
    - Encodes Churn as 0/1.
    - One-hot encodes every remaining text column, dropping the first category
      of each to avoid redundant columns.
    """
    df = df.copy()

    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    categorical_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def split_features_and_target(df: pd.DataFrame, target_col: str = "Churn"):
    """Separate the feature matrix X from the target vector y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
