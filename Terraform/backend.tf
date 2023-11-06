terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
  backend "s3" {
    bucket  = "noah-terraform-remote-state"
    key     = "modal/cutout-gen-config"
    region  = "ap-southeast-1"
    profile = "noahTest"
  }
}

provider "aws" {
  region = "ap-southeast-1"
  profile = "noahTest"
}
