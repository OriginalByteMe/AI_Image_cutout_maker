"""Public HTTP API.

A lightweight CPU container serves FastAPI and fans work out to the GPU
service. Bulk image requests run in parallel — Modal autoscales SAM 3
containers, so N images take roughly the time of one.
"""

import asyncio
import os
import uuid

import modal
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.common import app, aws_secret, web_image
from app.cutout.service import SAM3Service
from app.s3_utils import IMAGES_PREFIX, VIDEOS_PREFIX, get_s3
from app.schemas import (
    CutoutRequest,
    CutoutResponse,
    ImageResult,
    JobStarted,
    JobStatus,
    UploadedFile,
    UploadResponse,
    VideoCutoutRequest,
)

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/webm"}

web_app = FastAPI(
    title="AI Cutout Maker",
    description="Text-prompted image & video cutouts powered by SAM 3.",
    version="2.0.0",
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_name(filename: str | None) -> str:
    base = os.path.basename(filename or "upload")
    return f"{uuid.uuid4().hex[:8]}_{base.replace(' ', '_')}"


async def _store_upload(
    file: UploadFile, prefix: str, allowed_types: set[str], max_bytes: int
) -> UploadedFile:
    if file.content_type not in allowed_types:
        raise HTTPException(415, f"Unsupported content type: {file.content_type}")
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(413, f"File too large (max {max_bytes // 1024 // 1024} MB)")
    s3 = get_s3()
    name = _safe_name(file.filename)
    key = f"{prefix}/{name}"
    s3.upload_bytes(data, key, file.content_type)
    return UploadedFile(
        name=name, url=s3.presign(key), size=len(data), content_type=file.content_type
    )


@web_app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": "sam3"}


@web_app.get("/warmup")
async def warmup() -> dict:
    """Pre-boot a GPU container without blocking the caller."""
    call = SAM3Service().warm.spawn()
    return {"status": "warming", "job_id": call.object_id}


@web_app.post("/images", response_model=UploadResponse)
async def upload_images(files: list[UploadFile]) -> UploadResponse:
    if not files:
        raise HTTPException(400, "No files provided")
    if len(files) > 50:
        raise HTTPException(400, "Too many files (max 50 per request)")
    uploaded = [
        await _store_upload(f, IMAGES_PREFIX, ALLOWED_IMAGE_TYPES, MAX_IMAGE_BYTES)
        for f in files
    ]
    return UploadResponse(files=uploaded)


@web_app.get("/images/{image_name}")
async def get_image(image_name: str) -> dict:
    s3 = get_s3()
    key = f"{IMAGES_PREFIX}/{image_name}"
    if not s3.exists(key):
        raise HTTPException(404, "Image not found")
    head = s3.head(key)
    return {
        "name": image_name,
        "url": s3.presign(key),
        "size": head["ContentLength"],
        "content_type": head.get("ContentType"),
    }


@web_app.post("/cutouts", response_model=CutoutResponse)
async def create_cutouts(request: CutoutRequest) -> CutoutResponse:
    """Cut prompted concepts out of one or many images, in parallel."""
    service = SAM3Service()
    tasks = [
        service.segment_image.remote.aio(
            image_name=name,
            prompts=request.prompts,
            create_stickers=request.create_stickers,
            score_threshold=request.score_threshold,
            max_instances_per_prompt=request.max_instances_per_prompt,
        )
        for name in request.image_names
    ]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for name, outcome in zip(request.image_names, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            results.append(ImageResult(image_name=name, cutouts=[], error=str(outcome)))
        else:
            results.append(ImageResult(**outcome))
    return CutoutResponse(results=results)


@web_app.post("/videos", response_model=UploadResponse)
async def upload_video(files: list[UploadFile]) -> UploadResponse:
    if len(files) != 1:
        raise HTTPException(400, "Upload exactly one video per request")
    uploaded = await _store_upload(
        files[0], VIDEOS_PREFIX, ALLOWED_VIDEO_TYPES, MAX_VIDEO_BYTES
    )
    return UploadResponse(files=[uploaded])


@web_app.post("/video-cutouts", response_model=JobStarted)
async def create_video_cutout(request: VideoCutoutRequest) -> JobStarted:
    """Start an async video matting job; poll /jobs/{job_id} for the result."""
    s3 = get_s3()
    if not s3.exists(f"{VIDEOS_PREFIX}/{request.video_name}"):
        raise HTTPException(404, "Video not found — upload it via /videos first")
    call = SAM3Service().segment_video.spawn(
        video_name=request.video_name, prompt=request.prompt
    )
    return JobStarted(job_id=call.object_id)


@web_app.get("/jobs/{job_id}", response_model=JobStatus)
async def job_status(job_id: str) -> JobStatus:
    try:
        call = modal.FunctionCall.from_id(job_id)
    except Exception:
        raise HTTPException(404, "Unknown job id") from None
    try:
        result = await call.get.aio(timeout=0)
        return JobStatus(job_id=job_id, status="done", result=result)
    except TimeoutError:
        return JobStatus(job_id=job_id, status="running")
    except Exception as exc:  # noqa: BLE001 - surface remote failure to client
        return JobStatus(job_id=job_id, status="failed", error=str(exc))


@app.function(
    image=web_image,
    secrets=[aws_secret],
    scaledown_window=300,
    timeout=600,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def api():
    return web_app
