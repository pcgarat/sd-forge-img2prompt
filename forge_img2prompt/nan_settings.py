"""Forge Settings + lectura de config NaN (nan.builders, API OpenAI-compatible)."""

from __future__ import annotations

import os
from dataclasses import dataclass

# NaN (https://nan.builders): clúster comunitario de GPUs que sirve modelos
# abiertos tras una API compatible con OpenAI. Base /v1 + Bearer key (sk-...).
DEFAULT_NAN_BASE_URL = "https://api.nan.builders/v1"
DEFAULT_NAN_MODEL = "deepseek-v4-flash"
NAN_VALUE = "nan:"
DEFAULT_NAN_TIMEOUT = 120

# Modelos NaN con capacidad de visión (entrada de imagen). id, etiqueta, riesgo.
# El clúster publica la lista viva en GET /v1/models; esto es fuente de etiquetas
# y fallback si /models no responde.
NAN_VISION_MODELS: tuple[tuple[str, str, str], ...] = (
    ("deepseek-v4-flash", "DeepSeek V4 Flash · 305B MoE · 1M ctx · 3B tok/mes", ""),
    ("glm5.3-flash", "GLM 5.3 Flash · 320B-18B · 1M ctx · 2B tok/mes", ""),
    ("qwen3.8-flash", "Qwen 3.8 Flash · 125B-6B · 1M ctx · 500M tok/mes", ""),
    ("mimo-v2.6-flash", "MiMo V2.6 Flash · omnimodal · 1M ctx · 1B tok/mes", ""),
    ("gemma4", "Gemma 4 · 26B-A4B · 262K ctx · sin contador", ""),
    ("qwen3.6", "Qwen 3.6 · 35B-A3B · 262K ctx · sin contador", ""),
    ("glm5.3", "GLM 5.3 · 753B MoE · 1M ctx · 3B tok/periodo", "premium"),
)

OPT_BASE_URL = "img2prompt_nan_base_url"
OPT_API_KEY = "img2prompt_nan_api_key"
OPT_TIMEOUT = "img2prompt_nan_timeout"


def nan_vision_model_ids() -> tuple[str, ...]:
    return tuple(model_id for model_id, _label, _risk in NAN_VISION_MODELS)


def vision_label(model_id: str) -> str:
    for known_id, label, _risk in NAN_VISION_MODELS:
        if known_id == model_id:
            return label
    return model_id


def vision_risk(model_id: str) -> str:
    for known_id, _label, risk in NAN_VISION_MODELS:
        if known_id == model_id:
            return risk
    return ""


def _env_base_url() -> str:
    return (os.environ.get("NAN_BASE_URL") or "").strip()


def normalize_nan_base_url(raw: str | None) -> str:
    """Base OpenAI-compatible; tolera que peguen la URL completa del endpoint."""
    base = (raw or "").strip()
    if not base:
        base = _env_base_url() or DEFAULT_NAN_BASE_URL
    if not base.startswith("http://") and not base.startswith("https://"):
        base = f"https://{base}"
    # Quita barras y, repetidamente, sufijos de endpoint (/chat/completions, /models).
    while True:
        stripped = base.rstrip("/")
        if stripped != base:
            base = stripped
            continue
        for suffix in ("/chat/completions", "/models"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        else:
            break
    return base.rstrip("/")


def normalize_nan_model(model: str | None) -> str:
    """Model id no vacío; vacío → default."""
    return (model or "").strip() or DEFAULT_NAN_MODEL


def parse_nan_value(value: str | None) -> str:
    """`nan:` / `nan:<modelo>` → model id (default si vacío)."""
    v = (value or "").strip()
    if not v:
        return DEFAULT_NAN_MODEL
    if v == NAN_VALUE:
        return DEFAULT_NAN_MODEL
    if v.startswith(NAN_VALUE):
        return v[len(NAN_VALUE) :].strip() or DEFAULT_NAN_MODEL
    return v


def is_nan_value(value: str | None) -> bool:
    v = (value or "").strip()
    return v == NAN_VALUE or v.startswith(NAN_VALUE)


@dataclass(frozen=True)
class NanConfig:
    base_url: str
    model: str
    api_key: str
    timeout: float

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/models"


def _opt(name: str, default):
    try:
        from modules import shared

        opts = getattr(shared, "opts", None)
        if opts is None:
            return default
        return getattr(opts, name, default)
    except Exception:
        return default


def get_nan_config(*, model: str | None = None) -> NanConfig:
    """URL/key/timeout desde Settings (o env); modelo desde la pestaña."""
    # ``normalize_nan_base_url`` resuelve la env cuando el campo está vacío o es
    # el default registrado (así ``NAN_BASE_URL`` sigue vivo tras registrar Settings).
    base = normalize_nan_base_url(str(_opt(OPT_BASE_URL, "") or "").strip())
    tag = normalize_nan_model(model)
    key = str(_opt(OPT_API_KEY, "") or "").strip()
    if not key:
        key = (os.environ.get("NAN_API_KEY") or "").strip()
    try:
        timeout = float(_opt(OPT_TIMEOUT, DEFAULT_NAN_TIMEOUT) or DEFAULT_NAN_TIMEOUT)
    except (TypeError, ValueError):
        timeout = float(DEFAULT_NAN_TIMEOUT)
    timeout = max(5.0, min(600.0, timeout))
    return NanConfig(base_url=base, model=tag, api_key=key, timeout=timeout)


def register_nan_settings() -> None:
    """Registrar opciones de conexión NaN en Settings (sin selector de modelo)."""
    import gradio as gr
    from modules import shared

    section = ("img2prompt_nan", "Image → Prompt / NaN")
    shared.opts.add_option(
        OPT_BASE_URL,
        shared.OptionInfo(
            DEFAULT_NAN_BASE_URL,
            "NaN base URL",
            gr.Textbox,
            {"interactive": True},
            section=section,
        ).info(
            "Endpoint OpenAI-compatible. Por defecto https://api.nan.builders/v1 "
            "(también respeta la env NAN_BASE_URL si el campo está vacío). "
            "El modelo se elige en la pestaña Image → Prompt."
        ),
    )
    shared.opts.add_option(
        OPT_API_KEY,
        shared.OptionInfo(
            "",
            "NaN API key",
            gr.Textbox,
            {"interactive": True, "type": "password"},
            section=section,
        ).info(
            "Personal y no transferible (empieza por `sk-`). Se genera en "
            "cloud.nan.builders → API Keys. También respeta la env NAN_API_KEY."
        ),
    )
    shared.opts.add_option(
        OPT_TIMEOUT,
        shared.OptionInfo(
            DEFAULT_NAN_TIMEOUT,
            "NaN timeout (segundos)",
            gr.Number,
            {"precision": 0},
            section=section,
        ),
    )
