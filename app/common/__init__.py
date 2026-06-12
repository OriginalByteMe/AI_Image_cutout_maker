"""Shared Modal resources: app, container images, volumes and secrets.

Everything that needs to be referenced from more than one module lives here so
that `modal deploy -m app.main` can wire the whole application together.
"""

import os

import modal

APP_NAME = "cutout-generator"

# Pinned for reproducible image builds. Bump deliberately when upgrading SAM 3.
SAM3_GIT_COMMIT = "8e451d5eb43c817b64ae7577fb7b9ae223db88a9"

# GPU used for inference. SAM 3 is an 848M parameter model; A10G (24GB) is a
# good price/performance default. Override at deploy time with e.g.
# `CUTOUT_GPU=L40S modal deploy -m app.main`.
GPU_TYPE = os.environ.get("CUTOUT_GPU", "A10G")

# Where the HuggingFace checkpoint cache volume is mounted inside containers.
CACHE_DIR = "/cache"

app = modal.App(APP_NAME)

# Persists SAM 3 checkpoints across container restarts so only the very first
# cold start pays the download cost.
model_cache = modal.Volume.from_name("sam3-model-cache", create_if_missing=True)

# AWS credentials for the S3 bucket (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
# AWS_REGION, CUTOUT_BUCKET).
aws_secret = modal.Secret.from_name("my-aws-secret")

# HuggingFace token with access to the gated facebook/sam3 repository.
# Create with: modal secret create huggingface-secret HF_TOKEN=hf_...
huggingface_secret = modal.Secret.from_name("huggingface-secret")

# GPU image: PyTorch + SAM 3 + ffmpeg (for video matting output).
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision")
    .pip_install(f"git+https://github.com/facebookresearch/sam3.git@{SAM3_GIT_COMMIT}")
    .pip_install(
        "opencv-python-headless~=4.10",
        "pillow~=11.0",
        "boto3~=1.35",
        "huggingface_hub",
    )
    .env({"HF_HOME": f"{CACHE_DIR}/huggingface"})
    .add_local_python_source("app")
)

# Lightweight CPU image for the FastAPI gateway.
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi~=0.115",
        "pydantic~=2.9",
        "python-multipart~=0.0.20",
        "boto3~=1.35",
    )
    .add_local_python_source("app")
)
