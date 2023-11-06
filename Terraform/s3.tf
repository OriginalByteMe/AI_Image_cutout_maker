module "s3_bucket" {
  source = "terraform-aws-modules/s3-bucket/aws"

  bucket = "cutout-image-store"
  acl    = "private"

  control_object_ownership = true
  object_ownership         = "ObjectWriter"

  lifecycle_rule = [
    {
      id      = "expire"
      status  = "Enabled"
      enabled = true

      expiration = {
        days = 1
      }
    }
  ]
}
