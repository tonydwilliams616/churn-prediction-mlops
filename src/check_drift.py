"""
Compares new data against the saved baseline and reports whether any
monitored column has drifted.

Right now, with no live production traffic for this project, this is run
against the SAME raw CSV as a working demonstration of the mechanism -
notice the printed PSI scores come out at 0.0, confirming the check
correctly finds "no drift" when comparing data against itself. In a real
deployment, --new-data would point at freshly collected data (e.g. a fresh
export, or predictions logged from the live endpoint) instead.

Exits with a non-zero status code if drift is detected, so this can be
wired into a scheduled CI job that visibly fails (a red X in GitHub Actions)
when something needs a human to look at it.

Run from the project root:
    python -m src.check_drift
    python -m src.check_drift --new-data path/to/fresh_data.csv
"""

import argparse
import sys

from src.data import clean_data, load_raw_data
from src.monitoring import check_drift, load_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new-data",
        default="data/telco_churn.csv",
        help="Path to the data to check for drift (default: the original training CSV, as a working demo)",
    )
    parser.add_argument("--baseline-path", default="monitoring/baseline_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading baseline from {args.baseline_path} ...")
    baseline = load_baseline(args.baseline_path)

    print(f"Loading data to check from {args.new_data} ...")
    df = load_raw_data(args.new_data)
    df = clean_data(df)

    report = check_drift(baseline, df)

    print()
    print("Drift report:")
    print(f"{'Column':<30} {'PSI':>8}   Status")
    print("-" * 55)
    for col, result in report["columns"].items():
        status = "DRIFTED" if result["drifted"] else "ok"
        print(f"{col:<30} {result['psi']:>8}   {status}")

    print()
    if report["drifted"]:
        print("Drift detected in one or more columns - investigate before trusting current predictions.")
        sys.exit(1)
    else:
        print("No significant drift detected.")


if __name__ == "__main__":
    main()
