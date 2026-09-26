"""Forge Settings + lectura de config Ollama (patrón chatBot: OLLAMA_HOST)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3-vl:8b-instruct"
DEFAULT_OLLAMA_TIMEOUT = 120
OLLAMA_VALUE = "ollama:"

# Fallback si /api/tags no responde. Solo tags con visión conocidos.
OLLAMA_VISION_FALLBACK: tuple[str, ...] = (
    "qwen3-vl:8b-instruct",
    "huihui-qwen3vl-8b-abl:q6",
    "kimi-k3:cloud",
    "kimi-k2.6:cloud",
    "mistral-large-3:675b-cloud",
    "glm-5.3-flash:cloud",
    "gemma4:31b-cloud",
)

# Compat tests / imports antiguos
OLLAMA_MODEL_CHOICES = OLLAMA_VISION_FALLBACK

OPT_BASE_URL = "img2prompt_ollama_base_url"
OPT_API_KEY = "img2prompt_ollama_api_key"
OPT_TIMEOUT = "img2prompt_ollama_timeout"


def _in_docker() -> bool:
    if Path("/.dockerenv").is_file():
        return True
    try:
        return "docker" in Path("/proc/1/cgroup").read_text(encoding="utf-8")
    except OSError:
        return False


def default_base_url() -> str:
    """Host: localhost (como chatBot). Contenedor: gateway docker0 (Ollama en el host)."""
    env = (os.environ.get("OLLAMA_HOST") or "").strip()
    if env:
        if not env.startswith("http://") and not env.startswith("https://"):
            env = f"http://{env}"
        return env.rstrip("/")
    if _in_docker():
        return "http://172.17.0.1:11434"
    return DEFAULT_OLLAMA_HOST


def normalize_ollama_model(model: str | None) -> str:
    """Tag no vacío; vacío → default. Acepta cualquier tag (lista viva desde /api/tags)."""
    tag = (model or "").strip()
    return tag or DEFAULT_OLLAMA_MODEL


def parse_ollama_value(value: str | None) -> str:
    """`ollama:` / `ollama:tag` → tag Ollama (default si vacío)."""
    v = (value or "").strip()
    if not v:
        return DEFAULT_OLLAMA_MODEL
    if v == OLLAMA_VALUE:
        return DEFAULT_OLLAMA_MODEL
    if v.startswith("ollama:"):
        rest = v[len("ollama:") :].strip()
        return rest or DEFAULT_OLLAMA_MODEL
    return v


def is_cloud_model(model: str | None) -> bool:
    """Ollama Cloud: tags `*:cloud` o `*-cloud` (p.ej. gemma4:31b-cloud)."""
    tag = (model or "").strip().lower()
    if not tag:
        return False
    return tag.endswith(":cloud") or tag.endswith("-cloud")


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model: str
    api_key: str
    timeout: float

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/chat"

    @property
    def tags_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/tags"

    @property
    def is_cloud(self) -> bool:
        return is_cloud_model(self.model)


def _opt(name: str, default):
    try:
        from modules import shared

        opts = getattr(shared, "opts", None)
        if opts is None:
            return default
        return getattr(opts, name, default)
    except Exception:
        return default


def get_ollama_config(*, model: str | None = None) -> OllamaConfig:
    """URL/key/timeout desde Settings; modelo desde la pestaña (o default)."""
    base = str(_opt(OPT_BASE_URL, "") or "").strip() or default_base_url()
    tag = normalize_ollama_model(model)
    key = str(_opt(OPT_API_KEY, "") or "").strip()
    if not key:
        key = (os.environ.get("OLLAMA_API_KEY") or "").strip()
    try:
        timeout = float(_opt(OPT_TIMEOUT, DEFAULT_OLLAMA_TIMEOUT) or DEFAULT_OLLAMA_TIMEOUT)
    except (TypeError, ValueError):
        timeout = float(DEFAULT_OLLAMA_TIMEOUT)
    timeout = max(5.0, min(600.0, timeout))
    if not base.startswith("http://") and not base.startswith("https://"):
        base = f"http://{base}"
    return OllamaConfig(
        base_url=base.rstrip("/"),
        model=tag,
        api_key=key,
        timeout=timeout,
    )


def is_ollama_value(value: str) -> bool:
    v = (value or "").strip()
    return v == OLLAMA_VALUE or v.startswith("ollama:")


def register_ollama_settings() -> None:
    """Registrar opciones de conexión en Settings (sin selector de modelo)."""
    import gradio as gr
    from modules import shared

    section = ("img2prompt_ollama", "Image → Prompt / Ollama")
    shared.opts.add_option(
        OPT_BASE_URL,
        shared.OptionInfo(
            default_base_url(),
            "Ollama base URL",
            gr.Textbox,
            {"interactive": True},
            section=section,
        ).info(
            "Sin path /api. Host local: http://127.0.0.1:11434. "
            "Forge en Docker: http://172.17.0.1:11434 y OLLAMA_HOST=0.0.0.0:11434 en el host. "
            "Cloud directo: https://ollama.com (requiere API key). "
            "También respeta la env OLLAMA_HOST si el campo está vacío al crear la opción. "
            "El modelo se elige en la pestaña Image → Prompt."
        ),
    )
    shared.opts.add_option(
        OPT_API_KEY,
        shared.OptionInfo(
            "",
            "Ollama API key",
            gr.Textbox,
            {"interactive": True, "type": "password"},
            section=section,
        ).info(
            "Obligatoria si usas modelos `:cloud` vía https://ollama.com "
            "(Bearer). Vacía en local puro. Alternativa: `ollama signin` en el host "
            "con URL localhost / docker gateway. También respeta OLLAMA_API_KEY."
        ),
    )
    shared.opts.add_option(
        OPT_TIMEOUT,
        shared.OptionInfo(
            DEFAULT_OLLAMA_TIMEOUT,
            "Ollama timeout (segundos)",
            gr.Number,
            {"precision": 0},
            section=section,
        ),
    )
