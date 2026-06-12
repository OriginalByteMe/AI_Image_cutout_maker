"""GPU-backed SAM 3 inference service.

A single Modal Cls loads the model once per container (`@modal.enter`) and
serves both image and video segmentation. Containers are reused across
requests, so the per-request cost is inference only — no model loading.
"""

import io
import os
import shutil
import subprocess
import tempfile
import time

import modal

from app.common import (
    CACHE_DIR,
    GPU_TYPE,
    app,
    aws_secret,
    gpu_image,
    huggingface_secret,
    model_cache,
)
from app.cutout import postprocess
from app.s3_utils import (
    CUTOUTS_PREFIX,
    IMAGES_PREFIX,
    STICKERS_PREFIX,
    VIDEO_CUTOUTS_PREFIX,
    VIDEOS_PREFIX,
    get_s3,
)

with gpu_image.imports():
    import cv2
    import numpy as np
    import torch
    from PIL import Image

# Hard cap so a single video job stays within sane GPU-time bounds.
MAX_VIDEO_FRAMES = 1200


@app.cls(
    image=gpu_image,
    gpu=GPU_TYPE,
    volumes={CACHE_DIR: model_cache},
    secrets=[aws_secret, huggingface_secret],
    scaledown_window=300,
    timeout=1800,
)
class SAM3Service:
    @modal.enter()
    def load_model(self):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        t0 = time.monotonic()
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        # The video predictor is heavy; build it lazily on first video job.
        self.video_predictor = None
        # Persist any freshly downloaded checkpoints for future cold starts.
        model_cache.commit()
        print(f"SAM 3 image model loaded in {time.monotonic() - t0:.1f}s")

    def _get_video_predictor(self):
        if self.video_predictor is None:
            from sam3.model_builder import build_sam3_video_predictor

            t0 = time.monotonic()
            self.video_predictor = build_sam3_video_predictor()
            model_cache.commit()
            print(f"SAM 3 video predictor loaded in {time.monotonic() - t0:.1f}s")
        return self.video_predictor

    @modal.method()
    def warm(self) -> str:
        """No-op used to pre-boot a GPU container."""
        return "warm"

    @modal.method()
    def segment_image(
        self,
        image_name: str,
        prompts: list[str],
        create_stickers: bool = True,
        score_threshold: float = 0.5,
        max_instances_per_prompt: int = 20,
    ) -> dict:
        """Cut every instance of each prompted concept out of one image.

        Returns a dict matching `app.schemas.ImageResult`.
        """
        s3 = get_s3()
        image_bytes = s3.download_bytes(f"{IMAGES_PREFIX}/{image_name}")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_rgb = np.asarray(pil_image)
        stem = os.path.splitext(image_name)[0]

        cutouts = []
        with torch.inference_mode():
            state = self.processor.set_image(pil_image)
            for prompt in prompts:
                output = self.processor.set_text_prompt(state=state, prompt=prompt)
                masks = output["masks"]
                scores = output["scores"]
                if hasattr(masks, "cpu"):
                    masks = masks.cpu().numpy()
                if hasattr(scores, "cpu"):
                    scores = scores.cpu().numpy()
                masks = np.asarray(masks)
                scores = np.asarray(scores).reshape(-1)

                # Best matches first, capped per prompt.
                order = np.argsort(-scores)[:max_instances_per_prompt]
                slug = postprocess.slugify(prompt)
                for rank, idx in enumerate(order):
                    score = float(scores[idx])
                    if score < score_threshold:
                        continue
                    mask = np.squeeze(masks[idx])
                    try:
                        rgba = postprocess.mask_to_cutout(image_rgb, mask)
                    except ValueError:
                        continue  # empty mask

                    key = f"{CUTOUTS_PREFIX}/{stem}/{slug}_{rank}.png"
                    s3.upload_bytes(postprocess.encode_png(rgba), key, "image/png")
                    entry = {
                        "key": key,
                        "url": s3.presign(key),
                        "prompt": prompt,
                        "score": score,
                        "width": int(rgba.shape[1]),
                        "height": int(rgba.shape[0]),
                        "sticker_key": None,
                        "sticker_url": None,
                    }
                    if create_stickers:
                        sticker_key = f"{STICKERS_PREFIX}/{stem}/{slug}_{rank}.webp"
                        s3.upload_bytes(
                            postprocess.make_sticker(rgba), sticker_key, "image/webp"
                        )
                        entry["sticker_key"] = sticker_key
                        entry["sticker_url"] = s3.presign(sticker_key)
                    cutouts.append(entry)

        return {"image_name": image_name, "cutouts": cutouts, "error": None}

    @modal.method()
    def segment_video(self, video_name: str, prompt: str) -> dict:
        """Cut a prompted concept out of a video.

        Tracks every instance of the concept through the video and renders a
        transparent-background WebM (VP9 with alpha). Returns S3 key and a
        presigned URL.
        """
        s3 = get_s3()
        predictor = self._get_video_predictor()
        workdir = tempfile.mkdtemp(prefix="video-cutout-")
        session_id = None
        try:
            video_path = os.path.join(workdir, video_name.replace("/", "_"))
            with open(video_path, "wb") as f:
                f.write(s3.download_bytes(f"{VIDEOS_PREFIX}/{video_name}"))

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > MAX_VIDEO_FRAMES:
                cap.release()
                raise ValueError(
                    f"Video has {frame_count} frames; maximum is {MAX_VIDEO_FRAMES} "
                    f"(~{MAX_VIDEO_FRAMES / 30:.0f}s at 30fps). Trim it and retry."
                )

            response = predictor.handle_request(
                request=dict(type="start_session", resource_path=video_path)
            )
            session_id = response["session_id"]
            predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=prompt,
                )
            )

            masks_per_frame: dict[int, np.ndarray] = {}
            for response in predictor.handle_stream_request(
                request=dict(type="propagate_in_video", session_id=session_id)
            ):
                outputs = response["outputs"]
                binary_masks = np.asarray(outputs["out_binary_masks"])
                if binary_masks.size:
                    # Union of all tracked instances for this frame.
                    masks_per_frame[response["frame_index"]] = binary_masks.any(axis=0)

            frames_dir = os.path.join(workdir, "frames")
            os.makedirs(frames_dir)
            frame_idx = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                rgba = np.zeros((*frame_bgr.shape[:2], 4), dtype=np.uint8)
                mask = masks_per_frame.get(frame_idx)
                if mask is not None:
                    mask = np.squeeze(np.asarray(mask)).astype(bool)
                    if mask.shape != frame_bgr.shape[:2]:
                        mask = (
                            cv2.resize(
                                mask.astype(np.uint8),
                                (frame_bgr.shape[1], frame_bgr.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            > 0
                        )
                    rgba[..., :3] = frame_bgr[..., ::-1]
                    rgba[..., 3] = mask.astype(np.uint8) * 255
                cv2.imwrite(
                    os.path.join(frames_dir, f"{frame_idx:06d}.png"),
                    cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
                )
                frame_idx += 1
            cap.release()

            stem = os.path.splitext(video_name)[0]
            slug = postprocess.slugify(prompt)
            out_path = os.path.join(workdir, "out.webm")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    f"{fps:.3f}",
                    "-i",
                    os.path.join(frames_dir, "%06d.png"),
                    "-c:v",
                    "libvpx-vp9",
                    "-pix_fmt",
                    "yuva420p",
                    "-auto-alt-ref",
                    "0",
                    "-b:v",
                    "0",
                    "-crf",
                    "32",
                    out_path,
                ],
                check=True,
                capture_output=True,
            )

            key = f"{VIDEO_CUTOUTS_PREFIX}/{stem}/{slug}.webm"
            with open(out_path, "rb") as f:
                s3.upload_bytes(f.read(), key, "video/webm")

            return {
                "video_name": video_name,
                "prompt": prompt,
                "key": key,
                "url": s3.presign(key),
                "frame_count": frame_idx,
                "fps": fps,
            }
        finally:
            if session_id is not None:
                try:
                    predictor.handle_request(
                        request=dict(type="close_session", session_id=session_id)
                    )
                except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                    print(f"Failed to close video session: {exc}")
            shutil.rmtree(workdir, ignore_errors=True)
