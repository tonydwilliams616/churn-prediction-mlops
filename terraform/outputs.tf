# Outputs are values Terraform prints after it finishes creating resources -
# useful for grabbing information you'll need elsewhere (like the exact bucket
# name to use when uploading files or configuring the pipeline), without having
# to go dig through the AWS console.

output "bucket_name" {
  description = "Name of the S3 bucket created for this project"
  value       = aws_s3_bucket.data_bucket.id
}

output "bucket_arn" {
  description = "ARN (Amazon Resource Name) of the S3 bucket - the unique identifier AWS uses internally, needed later when we write IAM permissions"
  value       = aws_s3_bucket.data_bucket.arn
}

output "raw_data_s3_uri" {
  description = "Full S3 URI of the uploaded raw dataset, ready to hand to a SageMaker job later"
  value       = "s3://${aws_s3_bucket.data_bucket.id}/${aws_s3_object.raw_data.key}"
}

output "model_artifact_s3_uri" {
  description = "Full S3 URI of the uploaded trained model"
  value       = "s3://${aws_s3_bucket.data_bucket.id}/${aws_s3_object.model_artifact.key}"
}

output "sagemaker_execution_role_arn" {
  description = "ARN of the IAM role SageMaker jobs should assume - needed when we define training/inference jobs"
  value       = aws_iam_role.sagemaker_execution_role.arn
}
