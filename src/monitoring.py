"""
Lightweight, dependency-free drift detection for the Telco churn project.

Uses the Population Stability Index (PSI) - a standard, widely-used industry
metric for measuring how much a distribution has shifted between two
datasets. This is the same underlying concept SageMaker Model Monitor uses
internally, just implemented here with plain pandas/numpy instead of paid,
always-on AWS infrastructure.

How PSI works, in plain terms: split a column's values into buckets (bins),
compare what percentage of rows fell into each bucket in the BASELINE data
vs. what percentage falls into each bucket in NEW data, and combine those
differences into one number. A PSI of 0 means identical distributions; the
larger it gets, the more the data has shifted.

Common industry interpretation of the PSI score itself:
    PSI < 0.1  -> no significant change
    0.1 - 0.25 -> moderate shift, worth watching
    PSI > 0.25 -> significant drift, worth investigating
"""

import json

import numpy as np
import pandas as pd

DEFAULT_PSI_THRESHOLD = 0.25
DEFAULT_N_BINS = 10
EPSILON = 1e-4  # avoids divide-by-zero / log(0) when a bucket is empty


def _population_stability_index(baseline_props: pd.Series, current_props: pd.Series) -> float:
    """
    Core PSI calculation, given two aligned Series of bucket proportions
    (already indexed by the same bucket labels, summing to ~1.0 each).
    """
    aligned = pd.concat(
        [baseline_props.rename("baseline"), current_props.rename("current")],
        axis=1,
    ).fillna(0)

    # Replace zeros so we never divide by zero or take log(0) - a bucket
    # with genuinely zero occurrences still gets a small non-zero stand-in.
    aligned = aligned.replace(0, EPSILON)

    psi = ((aligned["current"] - aligned["baseline"]) * np.log(aligned["current"] / aligned["baseline"])).sum()
    return float(psi)


def _build_numeric_baseline(series: pd.Series, n_bins: int = DEFAULT_N_BINS) -> dict:
    """
    Splits a numeric column into n_bins buckets using baseline quantiles
    (so each bucket holds roughly the same share of baseline rows), and
    records both the bucket edges and the baseline proportions per bucket.
    """
    bin_edges = pd.qcut(series, q=n_bins, duplicates="drop", retbins=True)[1].tolist()
    # Extend the outer edges so future data outside the original range still
    # falls into the first/last bucket rather than being dropped entirely.
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    binned = pd.cut(series, bins=bin_edges)
    proportions = binned.value_counts(normalize=True).sort_index()

    return {
        "bin_edges": bin_edges,
        "proportions": {str(interval): prop for interval, prop in proportions.items()},
    }


def _build_categorical_baseline(series: pd.Series) -> dict:
    """Records the baseline proportion of each category value."""
    proportions = series.value_counts(normalize=True)
    return {"proportions": proportions.to_dict()}


def build_baseline(df: pd.DataFrame, numeric_columns: list, categorical_columns: list) -> dict:
    """
    Builds a baseline reference from a DataFrame - intended to be run once,
    against your known-good training data, and saved for future comparisons.
    """
    baseline = {"numeric": {}, "categorical": {}}

    for col in numeric_columns:
        baseline["numeric"][col] = _build_numeric_baseline(df[col])

    for col in categorical_columns:
        baseline["categorical"][col] = _build_categorical_baseline(df[col])

    return baseline


def save_baseline(baseline: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)


def load_baseline(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def check_drift(baseline: dict, new_df: pd.DataFrame, psi_threshold: float = DEFAULT_PSI_THRESHOLD) -> dict:
    """
    Compares new_df against a previously built baseline, column by column,
    and returns a report: each column's PSI score, whether it crossed the
    threshold, and an overall drifted flag if ANY column crossed it.
    """
    report = {"columns": {}, "drifted": False}

    for col, col_baseline in baseline.get("numeric", {}).items():
        if col not in new_df.columns:
            continue
        bin_edges = col_baseline["bin_edges"]
        baseline_props = pd.Series(col_baseline["proportions"])

        binned = pd.cut(new_df[col], bins=bin_edges)
        current_props = binned.value_counts(normalize=True).sort_index()
        current_props.index = current_props.index.map(str)

        psi = _population_stability_index(baseline_props, current_props)
        drifted = psi > psi_threshold
        report["columns"][col] = {"psi": round(psi, 4), "drifted": drifted}
        report["drifted"] = report["drifted"] or drifted

    for col, col_baseline in baseline.get("categorical", {}).items():
        if col not in new_df.columns:
            continue
        baseline_props = pd.Series(col_baseline["proportions"])
        current_props = new_df[col].value_counts(normalize=True)

        psi = _population_stability_index(baseline_props, current_props)
        drifted = psi > psi_threshold
        report["columns"][col] = {"psi": round(psi, 4), "drifted": drifted}
        report["drifted"] = report["drifted"] or drifted

    return report
