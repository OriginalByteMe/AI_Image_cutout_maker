import io

import numpy as np
import pytest
from PIL import Image

from app.cutout.postprocess import (
    STICKER_CANVAS,
    STICKER_MAX_BYTES,
    encode_png,
    make_sticker,
    mask_to_cutout,
    slugify,
)


def _sample_image(width=64, height=48):
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def test_slugify():
    assert slugify("Red Hat!") == "red-hat"
    assert slugify("  a   dog  ") == "a-dog"
    assert slugify("???") == "prompt"


def test_mask_to_cutout_crops_and_sets_alpha():
    image = _sample_image()
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:20, 30:40] = True

    rgba = mask_to_cutout(image, mask, padding=2)

    assert rgba.shape == (14, 14, 4)  # 10px region + 2px padding each side
    center_alpha = rgba[7, 7, 3]
    corner_alpha = rgba[0, 0, 3]
    assert center_alpha == 255
    assert corner_alpha == 0


def test_mask_to_cutout_resizes_mismatched_mask():
    image = _sample_image(64, 48)
    mask = np.zeros((24, 32), dtype=bool)  # half resolution
    mask[5:10, 8:16] = True
    rgba = mask_to_cutout(image, mask)
    assert rgba.shape[2] == 4
    assert rgba[..., 3].max() == 255


def test_mask_to_cutout_rejects_empty_mask():
    with pytest.raises(ValueError):
        mask_to_cutout(_sample_image(), np.zeros((48, 64), dtype=bool))


def test_encode_png_roundtrip():
    image = _sample_image()
    mask = np.ones((48, 64), dtype=bool)
    data = encode_png(mask_to_cutout(image, mask, padding=0))
    decoded = Image.open(io.BytesIO(data))
    assert decoded.mode == "RGBA"
    assert decoded.size == (64, 48)


def test_make_sticker_meets_whatsapp_spec():
    image = _sample_image(200, 100)
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:80, 40:160] = True
    data = make_sticker(mask_to_cutout(image, mask))

    assert len(data) <= STICKER_MAX_BYTES
    decoded = Image.open(io.BytesIO(data))
    assert decoded.format == "WEBP"
    assert decoded.size == (STICKER_CANVAS, STICKER_CANVAS)
