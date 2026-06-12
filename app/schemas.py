"""Pydantic request/response models for the public API."""

from pydantic import BaseModel, Field


class CutoutRequest(BaseModel):
    image_names: list[str] = Field(..., min_length=1, max_length=50)
    prompts: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Open-vocabulary concepts to cut out, e.g. ['dog', 'red hat']",
    )
    create_stickers: bool = True
    score_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_instances_per_prompt: int = Field(20, ge=1, le=100)


class Cutout(BaseModel):
    key: str
    url: str
    prompt: str
    score: float
    width: int
    height: int
    sticker_key: str | None = None
    sticker_url: str | None = None


class ImageResult(BaseModel):
    image_name: str
    cutouts: list[Cutout] = []
    error: str | None = None


class CutoutResponse(BaseModel):
    results: list[ImageResult]


class UploadedFile(BaseModel):
    name: str
    url: str
    size: int
    content_type: str | None = None


class UploadResponse(BaseModel):
    files: list[UploadedFile]


class VideoCutoutRequest(BaseModel):
    video_name: str
    prompt: str = Field(..., min_length=1)


class JobStarted(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # "running" | "done" | "failed"
    result: dict | None = None
    error: str | None = None
