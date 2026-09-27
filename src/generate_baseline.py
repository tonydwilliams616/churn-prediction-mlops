"""
Generates a drift-detection baseline from the training data and saves it to
monitoring/baseline_stats.json.

Run this ONCE, right after training a model you're happy with - the baseline
represents "what normal looked like" when the model was trained. Re-run it
only when you deliberately retrain on new data and want a fresh reference
point, not on every code change.

Run from the project root:
    python -m src.generate_baseline
"""

import argparse
import os

from src.data import clean_data, load_raw_data
from src.monitoring import build_baseline, save_baseline

# The columns we watch for drift - a mix of the numeric features and the
# categorical features notebook 03/06 identified as most predictive of
# churn (Contract especially). Not every column needs monitoring; picking
# the ones that actually matter to the model keeps this fast and focused.
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = ["Contract", "InternetService", "PaymentMethod"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/telco_churn.csv")
    parser.add_argument("--output-path", default="monitoring/baseline_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading training data from {args.data_path} ...")
    df = load_raw_data(args.data_path)
    df = clean_data(df)

    print(f"Building baseline from {len(df)} rows ...")
    baseline = build_baseline(df, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_baseline(baseline, args.output_path)

    print(f"Baseline saved to {args.output_path}")


if __name__ == "__main__":
    main()
