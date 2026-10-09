"""Cliente HTTP NaN (nan.builders) con API OpenAI-compatible (/v1/chat/completions)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from PIL import Image

from forge_img2prompt.image_encoding import jpeg_data_url
from forge_img2prompt.log import log
from forge_img2prompt.nan_settings import (
    DEFAULT_NAN_MODEL,
    NanConfig,
    nan_vision_model_ids,
)

MAX_RESPONSE_BYTES = 8_000_000
MAX_ERROR_BYTES = 2000


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """No seguir 3xx: evita reenviar el Bearer a otro host en un redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            newurl,
            code,
            "redirect bloqueado por seguridad (posible fuga de credenciales)",
            headers,
            fp,
        )


_SAFE_OPENER = urllib.request.build_opener(_NoRedirect)


def _open(req, *, timeout):
    """Abre una petición sin seguir redirects (protege el token Bearer)."""
    return _SAFE_OPENER.open(req, timeout=timeout)


def _jpeg_data_url(image: Image.Image, *, max_side: int = 1280) -> str:
    """data:image/jpeg;base64,… para el campo OpenAI ``image_url``."""
    return jpeg_data_url(image, max_side=max_side)


def _vision_user_content(
    images: list[Image.Image], user_text: str
) -> str | list[dict[str, Any]]:
    """OpenAI content parts: texto + ``image_url`` por cada imagen; texto si no hay."""
    imgs = [im for im in images if im is not None]
    text = (user_text or "").strip()
    if not imgs:
        return text
    parts: list[dict[str, Any]] = []
    for im in imgs:
        parts.append({"type": "image_url", "image_url": {"url": _jpeg_data_url(im)}})
    if text:
        parts.append({"type": "text", "text": text})
    return parts


def chat_with_image(
    cfg: NanConfig,
    *,
    system: str,
    user_text: str,
    image: Image.Image | list[Image.Image] | None = None,
    max_tokens: int = 320,
) -> str:
    """
    POST {base}/chat/completions con system + user(texto + imágenes) opcionales.

    ``image`` puede ser None (solo texto / notas→prompt), una PIL o una lista
    (primaria + refs foto 1…N). NaN expone modelos de razonamiento: además de
    ``deepseek-v4-flash``, muchos anteponen una traza de pensamiento, así que
    pedimos ``reasoning_effort: none`` (ignorado sin error donde no aplique).
    """
    if image is None:
        images: list[Image.Image] = []
    elif isinstance(image, list):
        images = image
    else:
        images = [image]
    n_images = sum(1 for im in images if im is not None)

    messages: list[dict[str, Any]] = []
    if (system or "").strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append(
        {"role": "user", "content": _vision_user_content(images, user_text)}
    )
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "stream": False,
        "max_tokens": int(max_tokens),
        "temperature": 0.2,
        # Trazas de razonamiento fuera: menos latencia y sin fugas al prompt.
        "reasoning_effort": "none",
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    log(
        f"NaN POST {cfg.chat_url} · model={cfg.model} · "
        f"max_tokens={max_tokens} · images={n_images}"
    )
    req = urllib.request.Request(cfg.chat_url, data=body, headers=headers, method="POST")
    try:
        with _open(req, timeout=cfg.timeout) as resp:
            raw = resp.read(MAX_RESPONSE_BYTES).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read(MAX_ERROR_BYTES).decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc.reason)
        raise RuntimeError(
            f"NaN HTTP {exc.code} en {cfg.chat_url}: {detail or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"No se pudo conectar a NaN en `{cfg.base_url}` ({exc.reason}). "
            "Comprueba la URL y la API key en Settings → Image → Prompt / NaN "
            "(la key empieza por `sk-`)."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Timeout ({cfg.timeout:.0f}s) esperando a NaN ({cfg.model})."
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Respuesta NaN no-JSON: {raw[:400]}") from exc

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            f"NaN sin choices: {data.get('error') or raw[:400]}"
        )
    message = choices[0].get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise RuntimeError(f"NaN sin message.content: {raw[:400]}")
    text = " ".join(content.split()).strip()
    log(f"NaN OK · {len(text)} chars")
    return text


def _fetch_models_payload(cfg: NanConfig) -> list[str] | None:
    headers = {}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    req = urllib.request.Request(cfg.models_url, headers=headers, method="GET")
    try:
        with _open(req, timeout=min(10.0, cfg.timeout)) as resp:
            data = json.loads(resp.read(MAX_RESPONSE_BYTES).decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return None
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None
    out: list[str] = []
    for it in items:
        if isinstance(it, dict):
            name = str(it.get("id") or it.get("name") or "").strip()
            if name:
                out.append(name)
    return out


def list_vision_models(cfg: NanConfig | None = None) -> list[str]:
    """
    Modelos NaN con visión. ``GET /v1/models`` intersecado con el registro
    curado; si /models falla o no trae conocidos, usa el registro completo.
    Orden: el default primero, luego el resto en orden de registro.
    """
    from forge_img2prompt.nan_settings import get_nan_config

    cfg = cfg or get_nan_config()
    known = nan_vision_model_ids()
    known_set = set(known)
    fetched = _fetch_models_payload(cfg)
    if fetched:
        live = [m for m in fetched if m in known_set]
        if live:
            live_set = set(live)
            ordered = [m for m in known if m in live_set]
            if DEFAULT_NAN_MODEL in ordered:
                ordered = [DEFAULT_NAN_MODEL, *[m for m in ordered if m != DEFAULT_NAN_MODEL]]
            log(f"NaN /models: {len(fetched)} modelos, {len(ordered)} con visión")
            return ordered
    log(f"NaN /models no disponible en {cfg.base_url}; catálogo curado")
    ordered = list(known)
    if DEFAULT_NAN_MODEL in ordered:
        ordered = [DEFAULT_NAN_MODEL, *[m for m in ordered if m != DEFAULT_NAN_MODEL]]
    return ordered


def ping_models(cfg: NanConfig) -> tuple[bool, str]:
    """Comprueba /v1/models; devuelve (ok, mensaje)."""
    fetched = _fetch_models_payload(cfg)
    if fetched is None:
        return False, f"Sin conexión a `{cfg.base_url}` (revisa URL y API key)."
    if cfg.model in fetched:
        return True, f"NaN OK · modelo `{cfg.model}` disponible en `{cfg.base_url}`."
    preview = ", ".join(fetched[:8]) or "(ninguno)"
    return (
        False,
        f"NaN responde en `{cfg.base_url}` pero no lista `{cfg.model}`. "
        f"Vistos: {preview}.",
    )
