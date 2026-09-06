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
