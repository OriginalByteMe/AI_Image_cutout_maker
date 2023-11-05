import os
import boto3
from botocore.exceptions import ClientError


class Boto3Client:
    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name=os.environ["AWS_REGION"],
        )

    def download_from_s3(self, save_path, image_name):
        s3_client = boto3.client("s3")
        file_path = os.path.join(save_path, image_name)
        try:
            s3_client.download_file(os.environ["CUTOUT_BUCKET"], f"cutouts/{image_name}/cutout-{image_name}", file_path)
        except ClientError as e:
            print("BOTO error: ",e)
            return None

        return file_path

    def upload_to_s3(self, file_body, image_name):
        self.s3.put_object(Body=file_body, Bucket=os.environ["CUTOUT_BUCKET"], Key=f"images/{image_name}")

    def generate_presigned_url(self, image_name, expiration=3600):
        try:
            response = self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": os.environ["CUTOUT_BUCKET"], "Key": f"cutouts/{image_name}/cutout-{image_name}"},
                ExpiresIn=expiration,
            )
        except ClientError as e:
            print(e)
            return None

        return response
