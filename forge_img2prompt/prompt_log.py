"""Historial de requests crudos enviados al modelo (Ollama, NaN o transformers).

Guarda el *body* tal cual (payload con images en base64, o messages con
tensores resumidos) más los parámetros de generación, para mostrarlo en la
pestaña Image → Prompt (acordeón **Log**).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

_MAX_B64_CHARS = 160
_MAX_ENTRIES = 10

_lock = threading.Lock()
_entries: list["LogEntry"] = []


@dataclass(frozen=True)
class LogEntry:
    ts: str
    transport: str
    text: str


def _truncate_b64(data: str) -> str:
    if len(data) <= _MAX_B64_CHARS:
        return data
    head = data[: _MAX_B64_CHARS // 2]
    tail = data[-_MAX_B64_CHARS // 2 :]
    return f"{head}…({len(data)} b64 chars)…{tail}"


def redact_images(obj: Any) -> Any:
    """Copia del payload con las imágenes base64 recortadas (legible en UI)."""
    if isinstance(obj, dict):
        return {
            k: (redact_images(v) if k != "images" else _redact_image_list(v))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [redact_images(v) for v in obj]
    return obj


def _redact_image_list(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    return [_truncate_b64(v) if isinstance(v, str) else v for v in value]


def redact_data_urls(obj: Any) -> Any:
    """Copia del payload OpenAI con las ``data:image/…;base64,`` recortadas."""
    if isinstance(obj, dict):
        return {k: redact_data_urls(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_data_urls(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("data:image"):
        head, _, b64 = obj.partition(",")
        return f"{head},{_truncate_b64(b64)}" if b64 else obj
    return obj


def _shape(value: Any) -> str:
    if hasattr(value, "shape"):
        try:
            return "×".join(str(d) for d in value.shape)
        except Exception:  # noqa: BLE001
            return "tensor"
    return "?"


def summarize_vl_messages(messages: Any) -> Any:
    """Sustituye PIL/tensores por marcadores para serializar el chat template."""
    if isinstance(messages, dict):
        return {k: summarize_vl_messages(v) for k, v in messages.items()}
    if isinstance(messages, (list, tuple)):
        return [summarize_vl_messages(v) for v in messages]
    if hasattr(messages, "shape"):
        return f"<tensor {_shape(messages)}>"
    if type(messages).__module__.startswith("PIL"):
        try:
            return f"<image {messages.size[0]}×{messages.size[1]} {messages.mode}>"
        except Exception:  # noqa: BLE001
            return "<image>"
    if isinstance(messages, bytes):
        return f"<{len(messages)} bytes>"
    return messages


def _to_json(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception:  # noqa: BLE001
        return str(payload)


def record(
    *,
    transport: str,
    payload: Any,
    params: dict[str, Any] | None = None,
    images: list[Any] | None = None,
) -> None:
    """Añade un request al historial; ``images`` es una lista de PIL para resumir."""
    images = images or []
    image_lines: list[str] = []
    for i, im in enumerate(images, start=1):
        try:
            image_lines.append(f"  #{i}: {im.size[0]}×{im.size[1]} {im.mode}")
        except Exception:  # noqa: BLE001
            image_lines.append(f"  #{i}: <imagen>")

    body = _to_json(payload)
    blocks: list[str] = []
    if image_lines and isinstance(payload, (dict, list)):
        blocks.append("// imágenes incluidas en el body (base64 recortado)\n" + "\n".join(image_lines))
    blocks.append(body)
    header = f"// transport: {transport}\n"
    if params:
        header += f"// params: {_to_json(params)}\n"
    text = header + "\n".join(blocks)
    entry = LogEntry(
        ts=datetime.now().strftime("%H:%M:%S"),
        transport=transport,
        text=text,
    )
    with _lock:
        _entries.insert(0, entry)
        del _entries[_MAX_ENTRIES:]


def last_raw() -> str:
    with _lock:
        return _entries[0].text if _entries else ""


def history() -> list[LogEntry]:
    with _lock:
        return list(_entries)


def render_markdown() -> str:
    """Historial en Markdown (más reciente primero) para el acordeón Log."""
    entries = history()
    if not entries:
        return "_Sin peticiones todavía: pulsa **Generate** o **Añadir detalle**._"
    blocks: list[str] = []
    total = len(entries)
    for i, entry in enumerate(entries):
        n = total - i
        blocks.append(f"**[{n}] {entry.ts}** · `{entry.transport}`\n\n```\n{entry.text}\n```")
    return "\n\n---\n\n".join(blocks)


def clear() -> None:
    with _lock:
        _entries.clear()
