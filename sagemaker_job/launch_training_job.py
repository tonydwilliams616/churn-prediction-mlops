"""
Launches a SageMaker training job that runs sagemaker_job/entry_point.py inside a
managed AWS container, reading data from S3 and writing the trained model
back to S3.

This script runs on YOUR laptop - it doesn't do any training itself. It just
tells SageMaker "start a training job with these settings," and AWS handles
spinning up a container, running entry_point.py inside it, and tearing the
container down again once training finishes. You're billed only for the
minutes the job actually runs (typically 5-10 minutes for a job this size).

Written against SageMaker Python SDK v3, which replaced the old framework-
specific Estimator classes (SKLearn, PyTorch, etc.) with a single unified
ModelTrainer class. Training now requires an explicit container image URI
rather than a "framework_version" shorthand - image_uris.retrieve() below
looks up AWS's own prebuilt scikit-learn training container for us.

Run from the project root:
    python -m sagemaker_job.launch_training_job \\
        --role-arn arn:aws:iam::<account-id>:role/churn-prediction-mlops-sagemaker-execution-role \\
        --bucket churn-prediction-mlops-dev-<suffix>

Both values come directly from `terraform output` in the terraform/ folder.
"""

import argparse

import boto3
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData, OutputDataConfig, SourceCode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", required=True, help="sagemaker_execution_role_arn from terraform output")
    parser.add_argument("--bucket", required=True, help="bucket_name from terraform output")
    parser.add_argument(
        "--instance-type",
        default="ml.m5.large",
        help="EC2 instance type the training container runs on (default: ml.m5.large)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_data_s3_uri = f"s3://{args.bucket}/data"
    output_s3_uri = f"s3://{args.bucket}/sagemaker-output"

    session = Session()
    region = session.boto_region_name

    # Looks up AWS's prebuilt scikit-learn training container image for this
    # region/instance type - the equivalent of what framework_version="1.2-1"
    # used to do implicitly in the old SDK.
    training_image = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type=args.instance_type,
        image_scope="training",
    )
    print(f"Using training container image: {training_image}")

    # SourceCode tells SageMaker to sync our local sagemaker_job/ folder into
    # the container at runtime, then run this command inside it. Hyperparameters
    # are passed as plain CLI flags now, rather than the old estimator's
    # hyperparameters={} dict.
    source_code = SourceCode(
        source_dir="sagemaker_job",
        command=(
            "python entry_point.py "
            "--n-estimators 400 "
            "--max-depth 10 "
            "--min-samples-split 5 "
            "--min-samples-leaf 4 "
            "--max-features log2"
        ),
    )

    compute = Compute(
        instance_type=args.instance_type,
        instance_count=1,
    )

    output_data_config = OutputDataConfig(s3_output_path=output_s3_uri)

    model_trainer = ModelTrainer(
        training_image=training_image,
        source_code=source_code,
        compute=compute,
        output_data_config=output_data_config,
        role=args.role_arn,
        sagemaker_session=session,
        base_job_name="churn-prediction-training",
    )

    # "train" is the channel name - inside the container, entry_point.py reads
    # this same name back out via the SM_CHANNEL_TRAIN environment variable,
    # which the training toolkit sets regardless of SDK version.
    train_data = InputData(channel_name="train", data_source=train_data_s3_uri)

    print(f"Starting SageMaker training job, reading data from {train_data_s3_uri}")
    print(f"Trained model will be saved under {output_s3_uri}")

    training_job = model_trainer.train(input_data_config=[train_data])
    print(f"Training job finished: {training_job.name}")

    # Asking the AWS API directly for the final artifact location, rather than
    # relying on an SDK attribute that may differ between versions - this call
    # is stable regardless of which SDK version launched the job.
    sm_client = boto3.client("sagemaker", region_name=region)
    description = sm_client.describe_training_job(TrainingJobName=training_job.name)
    model_s3_uri = description["ModelArtifacts"]["S3ModelArtifacts"]

    print(f"Model artifact location: {model_s3_uri}")


if __name__ == "__main__":
    main()

