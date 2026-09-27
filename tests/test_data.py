"""
Tests for src/data.py.

These use small, hand-built DataFrames rather than the real CSV - tests
should be fast, self-contained, and not depend on external files. Each test
checks one specific behavior, so a failure tells you exactly what broke.
"""

import pandas as pd

from src.data import clean_data


def test_clean_data_converts_total_charges_to_numeric():
    """TotalCharges should end up as a numeric dtype, not text."""
    df = pd.DataFrame({
        "TotalCharges": ["29.85", "1889.5", "108.15"],
    })

    result = clean_data(df)

    assert pd.api.types.is_numeric_dtype(result["TotalCharges"])


def test_clean_data_fills_blank_total_charges_with_zero():
    """
    The specific bug found in notebook 02: blank strings in TotalCharges
    (not proper NaN) should become 0 after cleaning, not stay missing.
    """
    df = pd.DataFrame({
        "TotalCharges": ["29.85", " ", "108.15"],
    })

    result = clean_data(df)

    assert result["TotalCharges"].isnull().sum() == 0
    assert result["TotalCharges"].iloc[1] == 0


def test_clean_data_does_not_mutate_the_original_dataframe():
    """
    clean_data() should return a new DataFrame, not modify the caller's copy
    in place - important so calling it can never surprise the caller by
    changing data they still hold a reference to elsewhere.
    """
    original = pd.DataFrame({
        "TotalCharges": ["29.85", " ", "108.15"],
    })
    original_dtype = original["TotalCharges"].dtype

    clean_data(original)

    assert original["TotalCharges"].dtype == original_dtype
