import os
import boto3
from botocore.exceptions import ClientError


class Boto3Client:
    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            endpoint_url="https://13583f5ff84f5693a4a859a769743849.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name="auto",
        )

    def download_from_s3(self, save_path, bucket_name, key):
        s3_client = boto3.client("s3")

        file_path = os.path.join(save_path, key)
        try:
            s3_client.download_file(bucket_name, key, file_path)
        except ClientError as e:
            print("BOTO error: ",e)
            return None

        return file_path

    def upload_to_s3(self, bucket_name, file_body, key):
        self.s3.put_object(Body=file_body, Bucket=bucket_name, Key=key)

    def generate_presigned_url(self, bucket_name, key, expiration=3600):
        try:
            response = self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": key},
                ExpiresIn=expiration,
            )
        except ClientError as e:
            print(e)
            return None

        return response
