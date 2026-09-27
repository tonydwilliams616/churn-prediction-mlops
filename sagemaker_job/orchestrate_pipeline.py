"""
Runs the full pipeline: Processing -> Training -> Evaluation -> Conditional
Registration, using boto3 directly rather than the native SageMaker
Pipelines service.

This achieves the same conceptual outcome a managed Pipeline would - a
repeatable, automated multi-stage workflow with a quality gate before a
model is registered - with orchestration logic we control end-to-end,
using the same boto3-direct approach that's proven reliable throughout
this project's training and deployment work.

Run from the project root:
    python -m sagemaker_job.orchestrate_pipeline \\
        --role-arn arn:aws:iam::<account-id>:role/churn-prediction-mlops-sagemaker-execution-role \\
        --bucket churn-prediction-mlops-dev-<suffix>
"""

import argparse
import json
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData, OutputDataConfig, SourceCode

ROC_AUC_THRESHOLD = 0.75  # the quality gate - below this, the model is not registered
MODEL_PACKAGE_GROUP_NAME = "churn-prediction-models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--instance-type", default="ml.m5.large")
    return parser.parse_args()


def upload_script(s3_client, bucket: str, local_path: str, s3_key: str) -> str:
    """
    Uploads a single .py file to S3 as-is - NOT wrapped in a tar.gz.

    Unlike ModelTrainer's SourceCode mechanism (which transparently unpacks
    a tar.gz inside the training container), a Processing job's S3Input just
    downloads whatever object you point it at, keeping its original S3
    filename. Wrapping the script in a tar.gz here would just download an
    unopened tar.gz into the container - there's no equivalent "auto-extract"
    step for Processing jobs at the raw boto3 level, so the plain .py file
    has to be the thing sitting in S3 already.
    """
    with open(local_path, "rb") as f:
        s3_client.put_object(Bucket=bucket, Key=s3_key, Body=f.read())
    uri = f"s3://{bucket}/{s3_key}"
    print(f"Uploaded {local_path} to {uri}")
    return uri


def run_processing_job(sm_client, job_name: str, role_arn: str, image_uri: str,
                        code_s3_uri: str, input_s3_uri: str, train_output_s3: str,
                        test_output_s3: str, instance_type: str) -> None:
    print(f"\n=== Stage 1: Processing ({job_name}) ===")
    sm_client.create_processing_job(
        ProcessingJobName=job_name,
        RoleArn=role_arn,
        AppSpecification={
            "ImageUri": image_uri,
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/processing.py"],
        },
        ProcessingInputs=[
            {
                "InputName": "raw-data",
                "S3Input": {
                    "S3Uri": input_s3_uri,
                    "LocalPath": "/opt/ml/processing/input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": code_s3_uri,
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "train",
                    "S3Output": {
                        "S3Uri": train_output_s3,
                        "LocalPath": "/opt/ml/processing/train",
                        "S3UploadMode": "EndOfJob",
                    },
                },
                {
                    "OutputName": "test",
                    "S3Output": {
                        "S3Uri": test_output_s3,
                        "LocalPath": "/opt/ml/processing/test",
                        "S3UploadMode": "EndOfJob",
                    },
                },
            ]
        },
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": instance_type,
                "VolumeSizeInGB": 5,
            }
        },
        StoppingCondition={"MaxRuntimeInSeconds": 1800},
    )

    waiter = sm_client.get_waiter("processing_job_completed_or_stopped")
    waiter.wait(ProcessingJobName=job_name, WaiterConfig={"Delay": 15, "MaxAttempts": 60})

    status = sm_client.describe_processing_job(ProcessingJobName=job_name)["ProcessingJobStatus"]
    print(f"Processing job status: {status}")
    if status != "Completed":
        raise RuntimeError(f"Processing job did not complete successfully: {status}")


def run_training_job(role_arn: str, region: str, session, train_input_s3: str,
                      output_s3: str, instance_type: str, base_job_name: str) -> str:
    print(f"\n=== Stage 2: Training ===")
    training_image = image_uris.retrieve(
        framework="sklearn", region=region, version="1.2-1",
        py_version="py3", instance_type=instance_type, image_scope="training",
    )

    source_code = SourceCode(
        source_dir="sagemaker_job",
        command="python train_from_processed.py",
    )
    compute = Compute(instance_type=instance_type, instance_count=1)
    output_data_config = OutputDataConfig(s3_output_path=output_s3)

    model_trainer = ModelTrainer(
        training_image=training_image,
        source_code=source_code,
        compute=compute,
        output_data_config=output_data_config,
        role=role_arn,
        sagemaker_session=session,
        base_job_name=base_job_name,
    )

    train_data = InputData(channel_name="train", data_source=train_input_s3)
    model_trainer.train(input_data_config=[train_data])

    sm_client = boto3.client("sagemaker", region_name=region)
    recent_jobs = sm_client.list_training_jobs(
        NameContains=base_job_name, SortBy="CreationTime", SortOrder="Descending", MaxResults=1,
    )
    job_name = recent_jobs["TrainingJobSummaries"][0]["TrainingJobName"]
    description = sm_client.describe_training_job(TrainingJobName=job_name)
    model_s3_uri = description["ModelArtifacts"]["S3ModelArtifacts"]

    print(f"Training job finished: {job_name}")
    print(f"Model artifact: {model_s3_uri}")
    return model_s3_uri


def run_evaluation_job(sm_client, job_name: str, role_arn: str, image_uri: str,
                        code_s3_uri: str, model_s3_uri: str, test_s3_uri: str,
                        evaluation_output_s3: str, instance_type: str) -> dict:
    print(f"\n=== Stage 3: Evaluation ({job_name}) ===")
    sm_client.create_processing_job(
        ProcessingJobName=job_name,
        RoleArn=role_arn,
        AppSpecification={
            "ImageUri": image_uri,
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/evaluate.py"],
        },
        ProcessingInputs=[
            {
                "InputName": "model",
                "S3Input": {
                    "S3Uri": model_s3_uri,
                    "LocalPath": "/opt/ml/processing/model",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "test-data",
                "S3Input": {
                    "S3Uri": test_s3_uri,
                    "LocalPath": "/opt/ml/processing/test",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": code_s3_uri,
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "evaluation",
                    "S3Output": {
                        "S3Uri": evaluation_output_s3,
                        "LocalPath": "/opt/ml/processing/evaluation",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": instance_type,
                "VolumeSizeInGB": 5,
            }
        },
        StoppingCondition={"MaxRuntimeInSeconds": 1800},
    )

    waiter = sm_client.get_waiter("processing_job_completed_or_stopped")
    waiter.wait(ProcessingJobName=job_name, WaiterConfig={"Delay": 15, "MaxAttempts": 60})

    status = sm_client.describe_processing_job(ProcessingJobName=job_name)["ProcessingJobStatus"]
    print(f"Evaluation job status: {status}")
    if status != "Completed":
        raise RuntimeError(f"Evaluation job did not complete successfully: {status}")

    s3_client = boto3.client("s3")
    bucket, key = evaluation_output_s3.replace("s3://", "").split("/", 1)
    metrics_obj = s3_client.get_object(Bucket=bucket, Key=f"{key}/metrics.json")
    metrics = json.loads(metrics_obj["Body"].read())
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    return metrics


def ensure_model_package_group(sm_client, group_name: str) -> None:
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription="Churn prediction models approved by the automated pipeline",
        )
        print(f"Created model package group: {group_name}")
    except ClientError as e:
        if "already exists" in str(e):
            print(f"Model package group already exists: {group_name}")
        else:
            raise


def register_model(sm_client, group_name: str, model_s3_uri: str, image_uri: str, metrics: dict) -> str:
    print(f"\n=== Stage 5: Registration ===")
    response = sm_client.create_model_package(
        ModelPackageGroupName=group_name,
        ModelApprovalStatus="PendingManualApproval",
        InferenceSpecification={
            "Containers": [{"Image": image_uri, "ModelDataUrl": model_s3_uri}],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        # CustomerMetadataProperties is a simple, reliable way to attach our
        # evaluation metrics to the registered model package for reference -
        # a full ModelMetrics report (with a separate statistics.json in S3)
        # is a further extension we could add later, but isn't required for
        # the quality gate itself, which already happened above.
        CustomerMetadataProperties={k: str(v) for k, v in metrics.items()},
    )
    model_package_arn = response["ModelPackageArn"]
    print(f"Registered model package: {model_package_arn}")
    print("Status: PendingManualApproval - review it in the SageMaker Model Registry console before deploying.")
    return model_package_arn


def main() -> None:
    args = parse_args()
    session = Session()
    region = session.boto_region_name
    sm_client = boto3.client("sagemaker", region_name=region)
    s3_client = boto3.client("s3", region_name=region)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    training_image = image_uris.retrieve(
        framework="sklearn", region=region, version="1.2-1",
        py_version="py3", instance_type=args.instance_type, image_scope="training",
    )

    # Upload each stage's code once, up front. The timestamp lives in the
    # FOLDER path, not the filename - the actual object must be named
    # exactly "processing.py" / "evaluate.py" so it lands at the exact local
    # path each job's ContainerEntrypoint expects.
    processing_code_uri = upload_script(
        s3_client, args.bucket,
        "sagemaker_job/processing.py",
        f"pipeline-code/{timestamp}/processing.py",
    )
    evaluate_code_uri = upload_script(
        s3_client, args.bucket,
        "sagemaker_job/evaluate.py",
        f"pipeline-code/{timestamp}/evaluate.py",
    )

    train_output_s3 = f"s3://{args.bucket}/pipeline/train"
    test_output_s3 = f"s3://{args.bucket}/pipeline/test"
    training_job_output_s3 = f"s3://{args.bucket}/pipeline/model-output"
    evaluation_output_s3 = f"s3://{args.bucket}/pipeline/evaluation"

    # Stage 1: Processing
    run_processing_job(
        sm_client, f"churn-pipeline-process-{timestamp}", args.role_arn, training_image,
        processing_code_uri, f"s3://{args.bucket}/data/telco_churn.csv",
        train_output_s3, test_output_s3, args.instance_type,
    )

    # Stage 2: Training
    model_s3_uri = run_training_job(
        args.role_arn, region, session, train_output_s3, training_job_output_s3,
        args.instance_type, f"churn-pipeline-train-{timestamp}",
    )

    # Stage 3: Evaluation
    metrics = run_evaluation_job(
        sm_client, f"churn-pipeline-eval-{timestamp}", args.role_arn, training_image,
        evaluate_code_uri, model_s3_uri, test_output_s3, evaluation_output_s3, args.instance_type,
    )

    # Stage 4: Conditional quality gate
    print(f"\n=== Stage 4: Quality Gate ===")
    roc_auc = metrics["roc_auc"]
    print(f"ROC-AUC: {roc_auc:.4f}  (threshold: {ROC_AUC_THRESHOLD})")

    if roc_auc < ROC_AUC_THRESHOLD:
        print("Model did NOT meet the quality threshold. Stopping pipeline - not registering.")
        return

    print("Model meets the quality threshold. Proceeding to registration.")

    # Stage 5: Registration
    ensure_model_package_group(sm_client, MODEL_PACKAGE_GROUP_NAME)
    inference_image = image_uris.retrieve(
        framework="sklearn", region=region, version="1.2-1",
        py_version="py3", instance_type=args.instance_type, image_scope="inference",
    )
    register_model(sm_client, MODEL_PACKAGE_GROUP_NAME, model_s3_uri, inference_image, metrics)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
