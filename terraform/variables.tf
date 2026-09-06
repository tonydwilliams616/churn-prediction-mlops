# Variables are Terraform's way of parameterizing configuration - values you can
# change without editing the actual resource definitions. Think of this file as
# the "settings" for everything else in this folder.

variable "aws_region" {
  description = "AWS region to deploy resources into"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short name used to prefix resource names, keeping everything for this project easy to identify in the AWS console"
  type        = string
  default     = "churn-prediction-mlops"
}

variable "environment" {
  description = "Deployment environment - useful later if you ever add a separate 'staging' or 'prod' setup"
  type        = string
  default     = "dev"
}
