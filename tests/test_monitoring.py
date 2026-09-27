"""
Tests for src/monitoring.py.
"""

import pandas as pd

from src.monitoring import build_baseline, check_drift


def test_check_drift_reports_zero_for_identical_data():
    """
    Comparing a dataset against a baseline built from that exact same data
    should report ~zero drift on every column - the fundamental sanity
    check that proves the mechanism itself is sound.
    """
    df = pd.DataFrame({
        "tenure": [1, 12, 24, 36, 48, 60, 6, 18, 30, 42],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month",
                     "One year", "Two year", "Month-to-month", "One year",
                     "Two year", "Month-to-month"],
    })

    baseline = build_baseline(df, numeric_columns=["tenure"], categorical_columns=["Contract"])
    report = check_drift(baseline, df)

    assert report["drifted"] is False
    assert report["columns"]["tenure"]["psi"] == 0.0
    assert report["columns"]["Contract"]["psi"] == 0.0


def test_check_drift_flags_a_clearly_shifted_categorical_column():
    """
    A new dataset where a category's frequency has changed dramatically
    should be flagged as drifted.
    """
    baseline_df = pd.DataFrame({
        "Contract": ["Month-to-month"] * 25 + ["One year"] * 25 + ["Two year"] * 50,
    })
    # A dramatically different mix: almost entirely Month-to-month now.
    shifted_df = pd.DataFrame({
        "Contract": ["Month-to-month"] * 95 + ["Two year"] * 5,
    })

    baseline = build_baseline(baseline_df, numeric_columns=[], categorical_columns=["Contract"])
    report = check_drift(baseline, shifted_df)

    assert report["drifted"] is True
    assert report["columns"]["Contract"]["drifted"] is True


def test_check_drift_ignores_columns_not_in_the_baseline():
    """Only columns present in the baseline should appear in the report."""
    baseline_df = pd.DataFrame({"tenure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    new_df = pd.DataFrame({
        "tenure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "SomeUnrelatedColumn": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })

    baseline = build_baseline(baseline_df, numeric_columns=["tenure"], categorical_columns=[])
    report = check_drift(baseline, new_df)

    assert "SomeUnrelatedColumn" not in report["columns"]
