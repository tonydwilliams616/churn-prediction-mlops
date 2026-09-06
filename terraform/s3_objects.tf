# This file tells Terraform to upload specific local files into the S3 bucket
# we created in s3.tf, so the bucket contents are managed as code alongside the
# bucket itself - rather than someone having to remember to run `aws s3 cp` by
# hand after every apply.

# The raw dataset - the same file our pipeline reads locally from data/.
resource "aws_s3_object" "raw_data" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "data/telco_churn.csv"
  source = "../data/telco_churn.csv"

  # etag lets Terraform detect when the local file has changed since the last
  # apply, by comparing an MD5 hash of its contents. Without this, Terraform
  # would only re-upload the file if you changed something in the resource
  # block itself (like the key) - it wouldn't notice the file's contents were
  # edited. filemd5() reads the file and computes that hash automatically.
  etag = filemd5("../data/telco_churn.csv")
}

# The trained model artifact from notebook 08 / src/pipeline.py.
resource "aws_s3_object" "model_artifact" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "models/churn_model.joblib"
  source = "../models/churn_model.joblib"
  etag   = filemd5("../models/churn_model.joblib")
}

# The matching expected feature-column order, saved alongside the model -
# needed together, since the model is meaningless without knowing which column
# order it expects.
resource "aws_s3_object" "model_feature_columns" {
  bucket = aws_s3_bucket.data_bucket.id
  key    = "models/feature_columns.joblib"
  source = "../models/feature_columns.joblib"
  etag   = filemd5("../models/feature_columns.joblib")
}
