"""Codificación compartida de imágenes (PNG base64 ↔ data URL JPEG)."""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from forge_img2prompt.image_encoding import jpeg_data_url, png_base64


def _img(size=(16, 16), color=(220, 40, 40)) -> Image.Image:
    return Image.new("RGB", size, color)


def _decode(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64)))


def test_png_base64_roundtrip():
    img = _decode(png_base64(_img()))
    assert img.format == "PNG"
    assert img.size == (16, 16)


def test_jpeg_data_url_prefix_and_payload():
    url = jpeg_data_url(_img())
    assert url.startswith("data:image/jpeg;base64,")
    img = _decode(url.split(",", 1)[1])
    assert img.format == "JPEG"


def test_downscales_above_max_side():
    url = jpeg_data_url(_img((4000, 2000)), max_side=1280)
    img = _decode(url.split(",", 1)[1])
    assert max(img.size) == 1280


def test_png_converts_non_rgb_mode():
    img = Image.new("L", (8, 8), 128)
    decoded = _decode(png_base64(img))
    assert decoded.mode == "RGB"
