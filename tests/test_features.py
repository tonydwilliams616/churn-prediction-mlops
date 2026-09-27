"""
Tests for src/features.py.
"""

import pandas as pd

from src.features import engineer_features, split_features_and_target


def _sample_cleaned_dataframe() -> pd.DataFrame:
    """A small, already-cleaned DataFrame matching the real schema's shape."""
    return pd.DataFrame({
        "customerID": ["0001-AAA", "0002-BBB", "0003-CCC"],
        "tenure": [1, 24, 60],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "Churn": ["Yes", "No", "No"],
    })


def test_engineer_features_drops_customer_id():
    """customerID has no predictive value and should never reach the model."""
    df = _sample_cleaned_dataframe()

    result = engineer_features(df)

    assert "customerID" not in result.columns


def test_engineer_features_encodes_churn_as_binary():
    """Churn should become 0/1, with Yes -> 1 and No -> 0."""
    df = _sample_cleaned_dataframe()

    result = engineer_features(df)

    assert result["Churn"].tolist() == [1, 0, 0]


def test_engineer_features_one_hot_encodes_categoricals():
    """
    A categorical column with N categories should become N-1 new columns
    after drop_first=True - here, Contract has 3 categories, so we expect
    exactly 2 new Contract_* columns, and no leftover text column.
    """
    df = _sample_cleaned_dataframe()

    result = engineer_features(df)

    contract_columns = [col for col in result.columns if col.startswith("Contract_")]
    assert len(contract_columns) == 2
    assert "Contract" not in result.columns


def test_engineer_features_output_is_fully_numeric():
    """
    The whole point of feature engineering: nothing but numeric/boolean
    columns should remain once this function is done.
    """
    df = _sample_cleaned_dataframe()

    result = engineer_features(df)

    non_numeric_columns = result.select_dtypes(include=["object", "str"]).columns.tolist()
    assert non_numeric_columns == []


def test_split_features_and_target_separates_correctly():
    """X should have every column except the target; y should be just the target."""
    df = pd.DataFrame({
        "tenure": [1, 24, 60],
        "MonthlyCharges": [29.85, 56.95, 89.10],
        "Churn": [1, 0, 0],
    })

    X, y = split_features_and_target(df)

    assert "Churn" not in X.columns
    assert list(X.columns) == ["tenure", "MonthlyCharges"]
    assert y.tolist() == [1, 0, 0]
