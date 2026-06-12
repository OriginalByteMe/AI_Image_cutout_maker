"""Pure image post-processing: masks -> transparent cutouts and stickers.

Deliberately free of Modal/torch imports so it can be unit-tested anywhere.
"""

import io
import re

import numpy as np
from PIL import Image

# WhatsApp sticker spec: 512x512 WebP, transparent background, < 100 KB.
STICKER_CANVAS = 512
STICKER_MARGIN = 16
STICKER_MAX_BYTES = 100_000


def slugify(text: str) -> str:
    """Make a prompt safe for use inside an S3 key."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "prompt"


def mask_to_cutout(image_rgb: np.ndarray, mask: np.ndarray, padding: int = 8) -> np.ndarray:
    """Apply a boolean mask to an RGB image and crop to the masked region.

    Returns an RGBA array where everything outside the mask is fully
    transparent (unlike the legacy implementation, which produced opaque
    black backgrounds).
    """
    if mask.dtype != bool:
        mask = mask > 0.5
    if mask.shape != image_rgb.shape[:2]:
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (image_rgb.shape[1], image_rgb.shape[0]), Image.NEAREST
        )
        mask = np.asarray(mask_img) > 127

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError("Mask is empty")

    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding + 1, image_rgb.shape[0])
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding + 1, image_rgb.shape[1])

    rgba = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    rgba[..., :3] = image_rgb[y0:y1, x0:x1]
    rgba[..., 3] = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    return rgba


def encode_png(rgba: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def make_sticker(rgba: np.ndarray) -> bytes:
    """Fit a cutout onto a 512x512 transparent WebP canvas (WhatsApp spec).

    Quality is stepped down until the file fits under the 100 KB limit.
    """
    cutout = Image.fromarray(rgba, "RGBA")
    target = STICKER_CANVAS - 2 * STICKER_MARGIN
    scale = min(target / cutout.width, target / cutout.height)
    new_size = (max(int(cutout.width * scale), 1), max(int(cutout.height * scale), 1))
    cutout = cutout.resize(new_size, Image.LANCZOS)

    canvas = Image.new("RGBA", (STICKER_CANVAS, STICKER_CANVAS), (0, 0, 0, 0))
    offset = (
        (STICKER_CANVAS - cutout.width) // 2,
        (STICKER_CANVAS - cutout.height) // 2,
    )
    canvas.paste(cutout, offset, cutout)

    data = b""
    for quality in (90, 80, 70, 60, 50, 40, 30):
        buffer = io.BytesIO()
        canvas.save(buffer, format="WEBP", quality=quality, method=4)
        data = buffer.getvalue()
        if len(data) <= STICKER_MAX_BYTES:
            break
    return data
