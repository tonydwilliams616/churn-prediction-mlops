# This file defines the actual S3 bucket that will hold our data and trained
# model artifacts in AWS, replacing the local-only data/ and models/ folders
# we've been using so far.

# S3 bucket names must be GLOBALLY unique across every AWS account on Earth -
# not just unique to you. "churn-prediction-mlops" is almost certainly already
# taken by someone. A random_id resource generates a short random suffix we can
# append, so the bucket name is virtually guaranteed to be free without you
# having to guess-and-check names manually.
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "data_bucket" {
  bucket = "${var.project_name}-${var.environment}-${random_id.bucket_suffix.hex}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# Versioning keeps every previous version of a file when it's overwritten,
# rather than silently replacing it. If you accidentally upload a corrupted
# dataset or a broken model file, you can roll back to the previous version -
# genuinely useful for exactly the kind of "oops" moments a real ML project has.
resource "aws_s3_bucket_versioning" "data_bucket_versioning" {
  bucket = aws_s3_bucket.data_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

# This blocks all forms of public access to the bucket by default. There is no
# reason customer data or a trained model should ever be publicly readable on
# the internet - this is a security best practice the exam explicitly expects
# you to know (Domain 4 territory, but worth setting up correctly from day one
# rather than bolting it on later).
resource "aws_s3_bucket_public_access_block" "data_bucket_public_access_block" {
  bucket = aws_s3_bucket.data_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Server-side encryption ensures data is encrypted automatically at rest inside
# S3, using AES-256. This costs nothing extra and there's essentially no reason
# not to enable it - another Domain 4-flavored default worth having from the
# start.
resource "aws_s3_bucket_server_side_encryption_configuration" "data_bucket_encryption" {
  bucket = aws_s3_bucket.data_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
