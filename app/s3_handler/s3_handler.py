import logging
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError


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
            s3_client.download_file(
                os.environ["CUTOUT_BUCKET"], f"images/{image_name}", file_path
            )
        except ClientError as e:
            print("BOTO error: ", e)
            print(
                f"File {image_name} not found in bucket {os.environ['CUTOUT_BUCKET']}"
            )
            return None

        return file_path

    def upload_to_s3(self, file_body, folder, image_name):
        try:
            self.s3.put_object(
                Body=file_body,
                Bucket=os.environ["CUTOUT_BUCKET"],
                Key=f"{folder}/{image_name}",
            )
            logging.info(f"Successfully uploaded {image_name} to {folder}")
        except NoCredentialsError as e:
            logging.error("No AWS credentials found")
            raise
        except BotoCoreError as e:
            logging.error(f"An error occurred with Boto3: {e}")
            raise
        except Exception as e:
            logging.error(f"An error occurred while uploading the image: {e}")
            raise

    def generate_presigned_urls(self, folder, expiration=3600):
        try:
            response = self.s3.list_objects_v2(
                Bucket=os.environ["CUTOUT_BUCKET"], Prefix=folder
            )
            urls = []
            for obj in response.get("Contents", []):
                key = obj["Key"]
                url = self.s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": os.environ["CUTOUT_BUCKET"], "Key": key},
                    ExpiresIn=expiration,
                )
                # Get object metadata
                metadata = self.s3.head_object(
                    Bucket=os.environ["CUTOUT_BUCKET"], Key=key
                )["Metadata"]
                urls.append((url, metadata))
        except ClientError as e:
            print(e)
            return None

        return urls

    def generate_presigned_url_with_metadata(self, folder, key, expiration=3600):
        try:
            # Generate presigned URL
            url = self.s3.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": os.environ["CUTOUT_BUCKET"],
                    "Key": f"{folder}/{key}",
                },
                ExpiresIn=expiration,
            )
            # Get object metadata
            metadata = self.s3.head_object(
                Bucket=os.environ["CUTOUT_BUCKET"], Key=f"{folder}/{key}"
            )["Metadata"]
        except ClientError as e:
            logging.error("An error occurred: %s", e)
            return None

        return url, metadata
