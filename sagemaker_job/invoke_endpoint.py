"""
Sends a test prediction request to a deployed endpoint.

Run from the project root:
    python -m sagemaker_job.invoke_endpoint --endpoint-name churn-prediction-20260926180000
"""

import argparse
import json

import boto3


# A realistic-looking sample customer. You don't need to provide every one of
# the model's 30 expected columns - inference.py's predict_fn fills in any
# missing one-hot columns as 0, so a partial, human-friendly request works.
SAMPLE_CUSTOMER = {
    "SeniorCitizen": 0,
    "tenure": 2,
    "MonthlyCharges": 85.50,
    "TotalCharges": 171.00,
    "Contract_Two year": False,
    "InternetService_Fiber optic": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runtime = boto3.client("sagemaker-runtime")

    print(f"Sending sample customer to endpoint: {args.endpoint_name}")
    print(f"Request payload: {json.dumps(SAMPLE_CUSTOMER, indent=2)}")

    response = runtime.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="application/json",
        Body=json.dumps(SAMPLE_CUSTOMER),
    )

    result = json.loads(response["Body"].read().decode())
    print()
    print("Response from endpoint:")
    print(json.dumps(result, indent=2))

    prediction = result["predictions"][0]
    probability = result["churn_probability"][0]
    print()
    print(f"Predicted churn: {'Yes' if prediction == 1 else 'No'}")
    print(f"Churn probability: {probability:.2%}")


if __name__ == "__main__":
    main()
