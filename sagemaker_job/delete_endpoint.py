"""
Deletes an endpoint and its associated endpoint config and model.

A serverless endpoint doesn't bill for idle time the way a real-time
endpoint does, so leaving it running costs very little - but it's still a
real, listed AWS resource. This script exists so tearing it down is a single
command whenever you want to tidy up, rather than three manual console
clicks across three different resource types.

Run from the project root:
    python -m sagemaker_job.delete_endpoint --resource-name churn-prediction-20260926180000
"""

import argparse

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resource-name",
        required=True,
        help="The name printed by deploy_endpoint.py - used for the endpoint, its config, and the model, all sharing one name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sm_client = boto3.client("sagemaker")

    print(f"Deleting endpoint: {args.resource_name}")
    sm_client.delete_endpoint(EndpointName=args.resource_name)

    print(f"Deleting endpoint config: {args.resource_name}")
    sm_client.delete_endpoint_config(EndpointConfigName=args.resource_name)

    print(f"Deleting model: {args.resource_name}")
    sm_client.delete_model(ModelName=args.resource_name)

    print("Cleanup complete.")


if __name__ == "__main__":
    main()
