import logging
import os

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from modal import Image, Mount, Secret, Stub, asgi_app

stub = Stub(name="s3_handler")

app = FastAPI()

origins = [
    "http://localhost:3000",  # localdevelopment
    "https://cutouts.noahrijkaard.com",  # main website
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================
#  API Endpoints
# ================================================
@app.post("/upload-image")
async def upload_image_to_s3(image: UploadFile = File(...)):
    """Upload an image to S3.

    Args:
        image (UploadFile, optional): File to upload to s3 . Defaults to File(...).

    Returns:
        str: Message indicating whether the upload was successful.
    """
    from botocore.exceptions import BotoCoreError, NoCredentialsError
    from s3_handler import Boto3Client

    s3_client = Boto3Client()
    try:
        s3_client.upload_to_s3(image.file, "images", image.filename)
    except NoCredentialsError as e:
        raise HTTPException(status_code=401, detail="No AWS credentials found") from e
    except BotoCoreError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An error occurred while uploading the image"
        ) from e
    return {"message": "Image uploaded successfully", "status_code": 200}


@app.get("/generate-presigned-urls/{image_name}")
async def generate_presigned_urls(image_name: str):
    """Generate presigned urls for the cutouts of an image.

    Args:
        image_name (str): Name of image to generate presigned urls for.

    Returns:
        List[str]: List of presigned urls for the cutouts of an image.
    """
    from s3_handler import Boto3Client

    s3_client = Boto3Client()
    return s3_client.generate_presigned_urls(f"cutouts/{image_name}")


@app.get("/get-image/{image_name}")
async def get_image(image_name: str):
    """Get an image from S3.

    Args:
        image_name (str): Name of image to get.

    Returns:
        FileResponse: FileResponse object containing the image.
    """
    from fastapi.responses import FileResponse
    from s3_handler import Boto3Client

    s3_client = Boto3Client()
    data = s3_client.generate_presigned_url_with_metadata("images", image_name)
    if data is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return data


@stub.function(
    image=Image.debian_slim().pip_install(
        "boto3",
        "fastapi",
        "starlette",
        "uvicorn",
        "python-multipart",
        "pydantic",
        "requests",
        "httpx",
        "httpcore",
        "httpx[http2]",
        "httpx[http1]",
    ),
    mounts=[Mount.from_local_python_packages("s3_handler")],
    secret=Secret.from_name("my-aws-secret"),
)
@asgi_app()
def main():
    return app


# =================================
#  Modal s3 functions
# =================================
# @stub.function(
#     image=Image.debian_slim().pip_install("boto3", "fastapi", "starlette", "uvicorn", "python-multipart", "pydantic", "requests", "httpx", "httpcore", "httpx[http2]", "httpx[http1]"), mounts=[Mount.from_local_python_packages("s3_handler")], secret=Secret.from_name("my-aws-secret")
# )
# def upload_to_s3(file_body, folder, image_name):
#     from s3_handler import Boto3Client
#     from botocore.exceptions import ClientError, BotoCoreError, NoCredentialsError

#     s3_client = Boto3Client()

#     try:
#         s3_client.upload_to_s3(file_body, folder, image_name)
#         logging.info(f"Successfully uploaded {image_name} to {folder}")
#     except NoCredentialsError as e:
#         logging.error("No AWS credentials found")
#         raise
#     except BotoCoreError as e:
#         logging.error(f"An error occurred with Boto3: {e}")
#         raise
#     except Exception as e:
#         logging.error(f"An error occurred while uploading the image: {e}")
#         raise
