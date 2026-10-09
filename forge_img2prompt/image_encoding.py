"""Codificación de imágenes PIL a base64 compartida por los backends VL.

Ollama espera PNG base64 en el campo ``images``; la API OpenAI-compatible
(NaN) espera una ``data:`` URL JPEG en ``image_url``. El escalado previo es
idéntico en ambos, así que se resuelve aquí una sola vez.
"""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

DEFAULT_MAX_SIDE = 1280


def _fit(image: Image.Image, max_side: int) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        image = image.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


def _encode(image: Image.Image, fmt: str, max_side: int, **save_kw: object) -> str:
    buf = BytesIO()
    _fit(image, max_side).save(buf, format=fmt, **save_kw)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def png_base64(image: Image.Image, *, max_side: int = DEFAULT_MAX_SIDE) -> str:
    """PNG base64 (campo ``images`` de Ollama ``/api/chat``)."""
    return _encode(image, "PNG", max_side, optimize=True)


def jpeg_data_url(image: Image.Image, *, max_side: int = DEFAULT_MAX_SIDE) -> str:
    """``data:image/jpeg;base64,…`` (campo ``image_url`` OpenAI-compatible)."""
    b64 = _encode(image, "JPEG", max_side, quality=90, optimize=True)
    return f"data:image/jpeg;base64,{b64}"
