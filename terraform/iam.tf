# Trust policy: defines WHO is allowed to assume this role. Here, only the
# SageMaker service itself can assume it - not IAM users, not other services.
data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution_role" {
  name               = "${var.project_name}-sagemaker-execution-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# Permissions policy: defines WHAT the role can do once assumed. Scoped to
# exactly this bucket (and its contents) rather than a broad s3:* / "*"
# resource grant - least privilege.
data "aws_iam_policy_document" "sagemaker_s3_access" {
  statement {
    sid    = "ListBucket"
    effect = "Allow"
    actions = [
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.data_bucket.arn,
    ]
  }

  statement {
    sid    = "ReadWriteObjects"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = [
      "${aws_s3_bucket.data_bucket.arn}/*",
    ]
  }
}

resource "aws_iam_policy" "sagemaker_s3_access" {
  name   = "${var.project_name}-sagemaker-s3-access"
  policy = data.aws_iam_policy_document.sagemaker_s3_access.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_s3_access_attachment" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = aws_iam_policy.sagemaker_s3_access.arn
}

# AWS's own managed policy covering the baseline permissions SageMaker needs
# to operate (CloudWatch logging, ECR image pulls for training/serving
# containers, etc.) - the S3 policy above only covers our specific bucket, so
# this fills in everything else SageMaker requires generically.
resource "aws_iam_role_policy_attachment" "sagemaker_full_access_attachment" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}
