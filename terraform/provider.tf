# ALL INFRASTRUCTURE DEPLOYED IN DEV ENV 3524...........

# This file tells Terraform two things: which "provider" (cloud/service) we're
# working with, and which version of Terraform and that provider we expect.
#
# Why pin versions? Terraform and its providers get updated over time, sometimes
# with breaking changes. Pinning versions here means your infrastructure won't
# suddenly behave differently just because you (or a teammate) installed a newer
# Terraform version. This is a real Domain 3 concept: reproducible infrastructure.

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

# The AWS provider is what actually knows how to talk to AWS's API on our
# behalf. It uses the credentials you already configured locally via
# `aws configure` - we never put credentials directly in Terraform files.
provider "aws" {
  region = var.aws_region
}
