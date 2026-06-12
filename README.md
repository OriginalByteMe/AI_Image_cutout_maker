# AI Image Cutout Maker

Text-prompted image & video cutouts powered by [SAM 3 (Segment Anything with
Concepts)](https://github.com/facebookresearch/sam3), deployed serverlessly on
[Modal](https://modal.com).

Describe what you want cut out — "dog", "person in a red hat" — and the
service finds **every instance** of that concept, cuts it out with a
transparent background, and (optionally) packages each one as a
WhatsApp-compatible sticker (512x512 WebP, < 100 KB). Videos work too: the
concept is tracked across frames and rendered as a transparent-background
WebM.

## Architecture

```
                ┌─────────────────────────────────────────────┐
                │ Modal app: cutout-generator                 │
 Frontend ───►  │  api (CPU, FastAPI gateway)                 │
                │   ├── uploads / presigned URLs ──► S3       │
                │   └── fan-out via .remote/.spawn            │
                │  SAM3Service (GPU, A10G)                    │
                │   ├── @modal.enter(): model loaded once     │
                │   ├── segment_image()  - parallel per image │
                │   └── segment_video()  - async job          │
                └─────────────────────────────────────────────┘
```

Key properties:

- **Models load once per container** (`@modal.enter`), not once per request.
- **Bulk requests fan out** — Modal autoscales GPU containers, so 10 images
  take roughly the time of 1.
- **Checkpoints are cached in a Modal Volume**, so only the very first cold
  start pays the HuggingFace download.
- **One model replaces two**: SAM 3's open-vocabulary text prompting replaces
  the old GroundingDINO + SAM v1 pipeline.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/warmup` | Pre-boot a GPU container (non-blocking) |
| `POST` | `/images` | Upload up to 50 images (multipart `files`) |
| `GET` | `/images/{name}` | Presigned URL + metadata for an uploaded image |
| `POST` | `/cutouts` | `{image_names, prompts, create_stickers, score_threshold}` → cutout + sticker URLs per image |
| `POST` | `/videos` | Upload one video (mp4/mov/webm, ≤ 100 MB) |
| `POST` | `/video-cutouts` | `{video_name, prompt}` → `{job_id}` (async) |
| `GET` | `/jobs/{job_id}` | Poll a video job: `running` / `done` / `failed` |

Interactive docs are served at `/docs` on the deployed gateway URL.

## Setup

1. **Modal secrets** (one-time):

   ```bash
   # AWS credentials for the S3 bucket
   modal secret create my-aws-secret \
     AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
     AWS_REGION=... CUTOUT_BUCKET=cutout-image-store

   # HuggingFace token — request access to the gated checkpoints first:
   # https://huggingface.co/facebook/sam3
   modal secret create huggingface-secret HF_TOKEN=hf_...
   ```

2. **Deploy:**

   ```bash
   pip install modal
   modal deploy -m app.main
   ```

   To use a bigger GPU: `CUTOUT_GPU=L40S modal deploy -m app.main`.

3. **Local dev loop:**

   ```bash
   modal serve -m app.main   # hot-reloading dev deployment
   pytest                    # unit tests (no GPU needed)
   ruff check app tests      # lint
   ```

## CI/CD

`.github/workflows/ci-cd.yml` lints and tests on every push/PR, then deploys
to Modal on pushes to `main` (env `main`) and `develop` (env `dev`). Requires
`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` repository secrets.

## Repository layout

```
app/
├── main.py              # deploy entrypoint (modal deploy -m app.main)
├── common/              # Modal app, images, volume, secrets
├── api.py               # FastAPI gateway (CPU container)
├── schemas.py           # Pydantic request/response models
├── s3_utils.py          # single shared S3 client
└── cutout/
    ├── service.py       # SAM3Service GPU class
    └── postprocess.py   # mask → transparent PNG / WhatsApp sticker (pure, tested)
tests/                   # unit tests for postprocess
Terraform/               # S3 bucket (note: objects expire after 1 day)
legacy_code/             # pre-SAM3 implementation, kept for reference
```

> **Note:** the Terraform-managed bucket has a 1-day lifecycle expiry, so
> uploads, cutouts and stickers disappear after 24h by design.
