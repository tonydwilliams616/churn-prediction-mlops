"""
Data loading and cleaning for the Telco churn project.

This module holds the logic from notebooks 01, 02, and part of 04 - reading the
raw CSV and fixing the data quality issues we found by hand during EDA. Nothing
in here should be new: it's the same steps, just written as reusable functions
instead of notebook cells.
"""

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw Telco churn CSV exactly as downloaded, no cleaning applied."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the data quality fixes discovered during EDA (notebook 02).

    Specifically:
    - TotalCharges is stored as text with 11 hidden blank strings, all belonging
      to brand-new customers with tenure == 0. Convert to numeric and fill those
      with 0, since that's the logically correct value for a customer who hasn't
      been billed yet.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df
