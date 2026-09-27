"""
A native SageMaker Pipeline, built using the @step function decorator
(SageMaker's "Remote Function" feature) rather than raw boto3 orchestration.

This is a genuinely different mechanism from orchestrate_pipeline.py: here,
AWS's Pipelines service owns execution, retries, and the dependency graph.
We just write plain Python functions and decorate them - the decorator
handles serializing arguments/return values between steps and building the
DAG automatically from which function calls which.

Reuses the already-processed train.csv/test.csv sitting in S3 from the
earlier boto3-orchestrated pipeline run, rather than re-deriving the
processing logic - this pipeline covers Training -> Evaluation ->
Conditional Registration.

Run from the project root:
    python -m sagemaker_job.native_pipeline \\
        --role-arn arn:aws:iam::<account-id>:role/churn-prediction-mlops-sagemaker-execution-role \\
        --bucket churn-prediction-mlops-dev-<suffix>
"""

import argparse
import io
import json
import os
import tarfile

import boto3
from botocore.exceptions import ClientError

ROC_AUC_THRESHOLD = 0.75
MODEL_PACKAGE_GROUP_NAME = "churn-prediction-models"


def train_model(train_csv_s3_uri: str, role_arn: str) -> str:
    """
    Trains a Random Forest on the given S3 CSV and returns the S3 URI of the
    uploaded model artifact.

    Runs INSIDE a SageMaker-managed job once decorated with @step, but the
    function body itself is plain Python - no SM_CHANNEL_* environment
    variables here, since @step handles moving data in and out for us.
    Because this isn't a "real" TrainingJob resource, there's no automatic
    model.tar.gz packaging - we do that ourselves explicitly.
    """
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    s3 = boto3.client("s3")
    bucket, key = train_csv_s3_uri.replace("s3://", "").split("/", 1)
    local_csv = "/tmp/train.csv"
    s3.download_file(bucket, key, local_csv)

    df = pd.read_csv(local_csv)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    model = RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_split=5,
        min_samples_leaf=4, max_features="log2", random_state=42,
    )
    model.fit(X, y)

    os.makedirs("/tmp/model", exist_ok=True)
    joblib.dump(model, "/tmp/model/churn_model.joblib")
    joblib.dump(X.columns.tolist(), "/tmp/model/feature_columns.joblib")

    tar_path = "/tmp/model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add("/tmp/model/churn_model.joblib", arcname="churn_model.joblib")
        tar.add("/tmp/model/feature_columns.joblib", arcname="feature_columns.joblib")

    output_key = "native-pipeline/model/model.tar.gz"
    s3.upload_file(tar_path, bucket, output_key)
    model_s3_uri = f"s3://{bucket}/{output_key}"
    print(f"Model uploaded to {model_s3_uri}")
    return model_s3_uri


def evaluate_model(model_s3_uri: str, test_csv_s3_uri: str) -> float:
    """
    Scores the trained model against the held-out test set and returns just
    the ROC-AUC as a plain float - simple enough to reference directly in a
    ConditionStep without needing JsonGet-style property extraction.
    """
    import joblib
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    s3 = boto3.client("s3")

    model_bucket, model_key = model_s3_uri.replace("s3://", "").split("/", 1)
    s3.download_file(model_bucket, model_key, "/tmp/model.tar.gz")
    os.makedirs("/tmp/model_extracted", exist_ok=True)
    with tarfile.open("/tmp/model.tar.gz") as tar:
        tar.extractall("/tmp/model_extracted")

    model = joblib.load("/tmp/model_extracted/churn_model.joblib")
    feature_columns = joblib.load("/tmp/model_extracted/feature_columns.joblib")

    test_bucket, test_key = test_csv_s3_uri.replace("s3://", "").split("/", 1)
    s3.download_file(test_bucket, test_key, "/tmp/test.csv")
    test_df = pd.read_csv("/tmp/test.csv")

    X_test = test_df.reindex(columns=feature_columns, fill_value=0)
    y_test = test_df["Churn"]
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"Evaluation ROC-AUC: {roc_auc:.4f}")
    return float(roc_auc)


def register_model(model_s3_uri: str, roc_auc: float, region: str) -> str:
    """
    Registers the model to the Model Registry - identical logic to the
    already-proven register_model() in orchestrate_pipeline.py, reused here
    inside a pipeline step that only runs if the ConditionStep passes.
    """
    from sagemaker.core import image_uris

    sm_client = boto3.client("sagemaker", region_name=region)

    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelPackageGroupDescription="Churn prediction models approved by the native pipeline",
        )
    except ClientError as e:
        if "already exists" not in str(e):
            raise

    inference_image = image_uris.retrieve(
        framework="sklearn", region=region, version="1.2-1",
        py_version="py3", instance_type="ml.m5.large", image_scope="inference",
    )

    response = sm_client.create_model_package(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
        ModelApprovalStatus="PendingManualApproval",
        InferenceSpecification={
            "Containers": [{"Image": inference_image, "ModelDataUrl": model_s3_uri}],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        CustomerMetadataProperties={"roc_auc": str(roc_auc)},
    )
    model_package_arn = response["ModelPackageArn"]
    print(f"Registered model package: {model_package_arn}")
    return model_package_arn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--instance-type", default="ml.m5.large")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Deferred imports - these pull in the pipeline-specific SDK pieces we
    # confirmed exist via introspection, kept separate from the plain
    # step-function bodies above so those bodies stay easy to unit test
    # without needing the full pipeline machinery imported.
    from sagemaker.core import image_uris
    from sagemaker.core.helper.session_helper import Session
    from sagemaker.core.workflow.conditions import ConditionGreaterThanOrEqualTo
    from sagemaker.core.workflow.pipeline_context import PipelineSession
    from sagemaker.mlops.workflow.condition_step import ConditionStep
    from sagemaker.mlops.workflow.function_step import step
    from sagemaker.mlops.workflow.pipeline import Pipeline

    session = Session()
    region = session.boto_region_name
    pipeline_session = PipelineSession()

    # @step tries to auto-detect a default container image based on the
    # LOCAL Python version, but only supports 3.8/3.10 - most modern local
    # setups run newer Python (3.12 here), so that auto-detection fails.
    # Passing our already-proven sklearn container image explicitly
    # sidesteps this entirely, the same image used everywhere else in this
    # project.
    step_image_uri = image_uris.retrieve(
        framework="sklearn", region=region, version="1.2-1",
        py_version="py3", instance_type=args.instance_type, image_scope="training",
    )

    train_csv_s3_uri = f"s3://{args.bucket}/pipeline/train/train.csv"
    test_csv_s3_uri = f"s3://{args.bucket}/pipeline/test/test.csv"

    # The prebuilt scikit-learn framework container includes the smaller
    # "SageMaker Training Toolkit," but NOT the full sagemaker SDK package
    # that @step's own invocation mechanism needs to bootstrap and run our
    # function inside the container. Without this, the container has no way
    # to even start running our code - it fails before our function body
    # ever executes. Pinned to match the local environment's installed
    # version exactly, for the same serialization-compatibility reason we
    # needed matching Python versions.
    dependencies_path = "sagemaker_job/requirements.txt"

    step_train = step(
        train_model,
        name="TrainModel",
        role=args.role_arn,
        instance_type=args.instance_type,
        image_uri=step_image_uri,
        dependencies=dependencies_path,
    )
    step_evaluate = step(
        evaluate_model,
        name="EvaluateModel",
        role=args.role_arn,
        instance_type=args.instance_type,
        image_uri=step_image_uri,
        dependencies=dependencies_path,
    )
    step_register = step(
        register_model,
        name="RegisterModel",
        role=args.role_arn,
        instance_type=args.instance_type,
        image_uri=step_image_uri,
        dependencies=dependencies_path,
    )

    model_s3_uri = step_train(train_csv_s3_uri, args.role_arn)
    roc_auc = step_evaluate(model_s3_uri, test_csv_s3_uri)
    register_result = step_register(model_s3_uri, roc_auc, region)

    condition_step = ConditionStep(
        name="CheckQualityGate",
        conditions=[ConditionGreaterThanOrEqualTo(left=roc_auc, right=ROC_AUC_THRESHOLD)],
        if_steps=[register_result],
        else_steps=[],
    )

    pipeline = Pipeline(
        name="churn-prediction-native-pipeline",
        steps=[condition_step],
        sagemaker_session=pipeline_session,
    )

    print("Upserting pipeline definition...")
    pipeline.upsert(role_arn=args.role_arn)

    print("Starting pipeline execution...")
    execution = pipeline.start()
    print(f"Pipeline execution ARN: {execution.arn}")
    print("Waiting for execution to complete (this can take several minutes)...")
    execution.wait(delay=20, max_attempts=90)

    status = execution.describe()["PipelineExecutionStatus"]
    print(f"Final pipeline status: {status}")


if __name__ == "__main__":
    main()
