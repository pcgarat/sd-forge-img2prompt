"""Cliente HTTP Ollama nativo (/api/chat), alineado con chatBot."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from PIL import Image

from forge_img2prompt.image_encoding import png_base64
from forge_img2prompt.log import log
from forge_img2prompt.ollama_settings import (
    OLLAMA_VISION_FALLBACK,
    OllamaConfig,
    is_cloud_model,
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


def image_to_b64(image: Image.Image, *, max_side: int = 1280) -> str:
    """PNG base64 para el campo ``images`` de /api/chat."""
    return png_base64(image, max_side=max_side)


def chat_with_image(
    cfg: OllamaConfig,
    *,
    system: str,
    user_text: str,
    image: Image.Image | list[Image.Image] | None = None,
    num_predict: int = 320,
) -> str:
    """
    POST {base}/api/chat con system + user(text) + images opcionales.

    ``image`` puede ser None (solo texto / notas→prompt), una PIL o una lista
    (primaria + refs foto 1…N).
    """
    if image is None:
        images: list[Image.Image] = []
    elif isinstance(image, list):
        images = image
    else:
        images = [image]
    b64_list = [image_to_b64(im) for im in images if im is not None]

    messages: list[dict[str, Any]] = []
    if (system or "").strip():
        messages.append({"role": "system", "content": system.strip()})
    user_msg: dict[str, Any] = {
        "role": "user",
        "content": (user_text or "").strip(),
    }
    if b64_list:
        user_msg["images"] = b64_list
    messages.append(user_msg)
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "stream": False,
        # 0 = descargar el modelo de VRAM al terminar (Forge y Ollama comparten GPU).
        "keep_alive": 0,
        "options": {"num_predict": int(num_predict), "temperature": 0.2},
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    from forge_img2prompt import prompt_log

    prompt_log.record(
        transport="Ollama POST /api/chat",
        payload=prompt_log.redact_images(payload),
        params={"num_predict": int(num_predict), "temperature": 0.2, "stream": False},
        images=images,
    )

    log(
        f"Ollama POST {cfg.chat_url} · model={cfg.model} · "
        f"num_predict={num_predict} · images={len(b64_list)}"
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
            f"Ollama HTTP {exc.code} en {cfg.chat_url}: {detail or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"No se pudo conectar a Ollama en {cfg.base_url} ({exc.reason}). "
            "Comprueba Settings → Image → Prompt / Ollama. "
            "Si Forge va en Docker, el host debe exponer Ollama "
            "(OLLAMA_HOST=0.0.0.0:11434) y la URL suele ser http://172.17.0.1:11434."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Timeout ({cfg.timeout:.0f}s) esperando a Ollama ({cfg.model})."
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Respuesta Ollama no-JSON: {raw[:400]}") from exc

    msg = data.get("message") or {}
    content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, str):
        raise RuntimeError(f"Ollama sin message.content: {raw[:400]}")
    text = " ".join(content.split()).strip()
    log(f"Ollama OK · {len(text)} chars")
    return text


def _fetch_tags_payload(cfg: OllamaConfig) -> dict[str, Any] | None:
    headers = {}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    req = urllib.request.Request(cfg.tags_url, headers=headers, method="GET")
    try:
        with _open(req, timeout=min(10.0, cfg.timeout)) as resp:
            data = json.loads(resp.read(MAX_RESPONSE_BYTES).decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _prefer_name(a: str, b: str) -> str:
    """Entre alias del mismo digest, preferir el nombre más corto / limpio."""
    if len(a) != len(b):
        return a if len(a) < len(b) else b
    return a if a < b else b


def list_vision_models(cfg: OllamaConfig | None = None) -> list[str]:
    """
    Tags Ollama con capability ``vision`` (local + cloud pullados).

    Deduplica por digest. Si /api/tags falla, usa ``OLLAMA_VISION_FALLBACK``.
    Orden: locales primero, luego cloud; dentro de cada grupo alfabético,
    con ``DEFAULT`` preferido al inicio si está presente.
    """
    from forge_img2prompt.ollama_settings import DEFAULT_OLLAMA_MODEL, get_ollama_config

    cfg = cfg or get_ollama_config()
    data = _fetch_tags_payload(cfg)
    if data is None:
        log(f"Ollama /api/tags no disponible en {cfg.base_url}; fallback curado")
        return list(OLLAMA_VISION_FALLBACK)

    models = data.get("models")
    if not isinstance(models, list):
        return list(OLLAMA_VISION_FALLBACK)

    by_digest: dict[str, str] = {}
    no_digest: list[str] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        caps = m.get("capabilities")
        if not isinstance(caps, list) or "vision" not in caps:
            continue
        name = str(m.get("name") or m.get("model") or "").strip()
        if not name:
            continue
        digest = str(m.get("digest") or "").strip()
        if digest:
            prev = by_digest.get(digest)
            by_digest[digest] = name if prev is None else _prefer_name(prev, name)
        else:
            no_digest.append(name)

    names = list(dict.fromkeys([*by_digest.values(), *no_digest]))
    if not names:
        return list(OLLAMA_VISION_FALLBACK)

    local = sorted(n for n in names if not is_cloud_model(n))
    cloud = sorted(n for n in names if is_cloud_model(n))
    ordered = [*local, *cloud]
    if DEFAULT_OLLAMA_MODEL in ordered:
        ordered = [DEFAULT_OLLAMA_MODEL, *[n for n in ordered if n != DEFAULT_OLLAMA_MODEL]]
    return ordered


def ping_tags(cfg: OllamaConfig) -> tuple[bool, str]:
    """Comprueba /api/tags; devuelve (ok, mensaje)."""
    data = _fetch_tags_payload(cfg)
    if data is None:
        return False, f"Sin conexión a `{cfg.base_url}`."

    models = data.get("models") if isinstance(data, dict) else None
    names: list[str] = []
    if isinstance(models, list):
        for m in models:
            if isinstance(m, dict):
                n = m.get("name") or m.get("model") or ""
                if n:
                    names.append(str(n))
    if cfg.model in names:
        return True, f"Ollama OK · modelo `{cfg.model}` disponible en `{cfg.base_url}`."
    if cfg.is_cloud:
        key_note = (
            "API key presente."
            if cfg.api_key
            else "sin API key en Settings (vale `ollama signin` en el host local)."
        )
        return (
            True,
            f"Ollama responde en `{cfg.base_url}` · cloud `{cfg.model}` "
            f"(no hace falta en /api/tags local). {key_note}",
        )
    preview = ", ".join(names[:8]) or "(ninguno)"
    return (
        False,
        f"Ollama responde en `{cfg.base_url}` pero no tiene `{cfg.model}`. "
        f"Vistos: {preview}.",
    )
