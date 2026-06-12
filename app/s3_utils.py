"""Single, shared S3 client used by both the gateway and the GPU service.

All operations are in-memory (bytes in / bytes out) — nothing is written to
the container filesystem unless a caller explicitly chooses to.
"""

import os
from functools import lru_cache

import boto3

# S3 key prefixes
IMAGES_PREFIX = "images"
VIDEOS_PREFIX = "videos"
CUTOUTS_PREFIX = "cutouts"
STICKERS_PREFIX = "stickers"
VIDEO_CUTOUTS_PREFIX = "video-cutouts"

PRESIGN_EXPIRY_SECONDS = 3600


class S3Client:
    def __init__(self, bucket: str | None = None):
        self.bucket = bucket or os.environ["CUTOUT_BUCKET"]
        # Credentials/region come from the environment (Modal secret).
        self._client = boto3.client("s3", region_name=os.environ.get("AWS_REGION"))

    def upload_bytes(self, data: bytes, key: str, content_type: str) -> str:
        self._client.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
        )
        return key

    def download_bytes(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self._client.exceptions.ClientError:
            return False

    def head(self, key: str) -> dict:
        return self._client.head_object(Bucket=self.bucket, Key=key)

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys

    def presign(self, key: str, expires: int = PRESIGN_EXPIRY_SECONDS) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )


@lru_cache(maxsize=1)
def get_s3() -> S3Client:
    """Process-wide singleton so containers reuse one boto3 client."""
    return S3Client()
