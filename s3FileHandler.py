import os
import boto3
from botocore.exceptions import ClientError


class Boto3Client:
    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            endpoint_url="https://13583f5ff84f5693a4a859a769743849.r2.cloudflarestorage.com",
            aws_access_key_id="52733bb777295dbf8912df8ae9549466",
            aws_secret_access_key="8bfc25e62071097ae93aed260702353d9341a80ee352e68561f9ac95e983055f",
            region_name="auto",
        ).meta.client

    def download_from_s3(bucket_name, key):
        s3_client = boto3.client("s3")

        file_path = os.path.join(os.getcwd(), key)
        try:
            s3_client.download_file(bucket_name, key, file_path)
        except ClientError as e:
            print(e)
            return None

        return file_path

    def upload_to_s3(self, bucket_name, file_body, key):
        s3_client = self.s3.Bucket(bucket_name)
        s3_client.put_object(Body=file_body, Key=key)

    def generate_presigned_url(self, bucket_name, key, expiration=3600):
        s3_client = self.s3.meta.client
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
