"""
Deploys the trained churn model behind a SageMaker Serverless Inference
endpoint.

Written directly against boto3 (AWS's foundational SDK) rather than the
higher-level `sagemaker` package - the boto3 API for endpoint creation has
been stable for years, unlike the sagemaker package's v3 rewrite we ran into
during training. This also shows you the actual underlying AWS API calls
(CreateModel, CreateEndpointConfig, CreateEndpoint), which map directly to
concepts the exam expects you to know.

Uses SERVERLESS inference deliberately, not a standard real-time endpoint.
A real-time endpoint runs 24/7 and bills continuously even at zero traffic -
easy to forget about and rack up real cost. A serverless endpoint scales to
zero when idle: you only pay for the actual milliseconds of compute used
when you send it a request. For a low-traffic portfolio project, this is the
correct choice, not just the cheaper one - matching inference type to
traffic pattern is a real architectural decision, not a shortcut.

Run from the project root:
    python -m sagemaker_job.deploy_endpoint \\
        --role-arn arn:aws:iam::<account-id>:role/churn-prediction-mlops-sagemaker-execution-role \\
        --bucket churn-prediction-mlops-dev-<suffix> \\
        --model-s3-uri s3://<bucket>/sagemaker-output/<job-name>/output/model.tar.gz
"""

import argparse
import io
import tarfile
import time
from datetime import datetime, timezone

import boto3
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", required=True, help="sagemaker_execution_role_arn from terraform output")
    parser.add_argument("--bucket", required=True, help="bucket_name from terraform output")
    parser.add_argument(
        "--model-s3-uri",
        required=True,
        help="S3 URI of the trained model.tar.gz, printed at the end of launch_training_job.py",
    )
    parser.add_argument("--memory-size-mb", type=int, default=2048, help="Serverless memory allocation (MB)")
    parser.add_argument("--max-concurrency", type=int, default=5, help="Max concurrent serverless invocations")
    return parser.parse_args()


def package_inference_code(bucket: str, region: str) -> str:
    """
    Packages sagemaker_job/inference.py into a tar.gz and uploads it to S3.

    This replicates what the higher-level SDK's source_dir mechanism does
    automatically during training - the container looks for a specific
    environment variable pointing at this tarball to know which script
    defines model_fn/input_fn/predict_fn/output_fn.
    """
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        tar.add("sagemaker_job/inference.py", arcname="inference.py")
    tar_buffer.seek(0)

    s3_client = boto3.client("s3", region_name=region)
    code_key = "sagemaker-code/inference-sourcedir.tar.gz"
    s3_client.put_object(Bucket=bucket, Key=code_key, Body=tar_buffer.getvalue())

    code_s3_uri = f"s3://{bucket}/{code_key}"
    print(f"Uploaded inference code to {code_s3_uri}")
    return code_s3_uri


def main() -> None:
    args = parse_args()

    session = Session()
    region = session.boto_region_name

    sm_client = boto3.client("sagemaker", region_name=region)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    resource_name = f"churn-prediction-{timestamp}"

    # Same prebuilt scikit-learn container as training, but the INFERENCE
    # variant - a different image, purpose-built for serving rather than
    # training.
    inference_image = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type="ml.m5.large",
        image_scope="inference",
    )
    print(f"Using inference container image: {inference_image}")

    code_s3_uri = package_inference_code(args.bucket, region)

    # SAGEMAKER_PROGRAM and SAGEMAKER_SUBMIT_DIRECTORY are how the container
    # knows which script to load and call model_fn/input_fn/predict_fn/
    # output_fn from - the inference equivalent of the training container's
    # SM_SOURCE_DIR / SAGEMAKER_TRAINING_MODULE environment variables we saw
    # in the training job logs.
    print(f"Creating model: {resource_name}")
    sm_client.create_model(
        ModelName=resource_name,
        ExecutionRoleArn=args.role_arn,
        Containers=[
            {
                "Image": inference_image,
                "Mode": "SingleModel",
                "ModelDataUrl": args.model_s3_uri,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": code_s3_uri,
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_REGION": region,
                },
            }
        ],
    )

    print(f"Creating endpoint config: {resource_name}")
    sm_client.create_endpoint_config(
        EndpointConfigName=resource_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": resource_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": args.memory_size_mb,
                    "MaxConcurrency": args.max_concurrency,
                },
            }
        ],
    )

    print(f"Creating endpoint: {resource_name}")
    sm_client.create_endpoint(
        EndpointName=resource_name,
        EndpointConfigName=resource_name,
    )

    print("Waiting for endpoint to become InService (this can take a few minutes)...")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=resource_name, WaiterConfig={"Delay": 15, "MaxAttempts": 60})

    description = sm_client.describe_endpoint(EndpointName=resource_name)
    print(f"Endpoint status: {description['EndpointStatus']}")
    print(f"Endpoint name: {resource_name}")
    print()
    print("Save this endpoint name - you'll need it to invoke or delete the endpoint.")


if __name__ == "__main__":
    main()
