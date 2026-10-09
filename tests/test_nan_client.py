"""Tests del cliente NaN (nan.builders, OpenAI-compatible; sin red real)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from PIL import Image

from forge_img2prompt.nan_client import (
    _jpeg_data_url,
    chat_with_image,
    list_vision_models,
    ping_models,
)
from forge_img2prompt.nan_settings import (
    DEFAULT_NAN_BASE_URL,
    DEFAULT_NAN_MODEL,
    NAN_VALUE,
    NanConfig,
    get_nan_config,
    is_nan_value,
    nan_vision_model_ids,
    normalize_nan_base_url,
    normalize_nan_model,
    parse_nan_value,
)


def _red_png() -> Image.Image:
    return Image.new("RGB", (16, 16), (220, 40, 40))


def _cfg(**over) -> NanConfig:
    base = dict(
        base_url="https://api.nan.builders/v1",
        model=DEFAULT_NAN_MODEL,
        api_key="sk-test",
        timeout=30,
    )
    base.update(over)
    return NanConfig(**base)


class _Resp:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


def test_defaults_and_parsing():
    assert DEFAULT_NAN_BASE_URL == "https://api.nan.builders/v1"
    assert DEFAULT_NAN_MODEL == "deepseek-v4-flash"
    assert DEFAULT_NAN_MODEL in nan_vision_model_ids()
    assert is_nan_value(NAN_VALUE)
    assert is_nan_value("nan:glm5.3-flash")
    assert not is_nan_value("ollama:qwen3-vl:8b-instruct")
    assert parse_nan_value("nan:qwen3.8-flash") == "qwen3.8-flash"
    assert parse_nan_value(NAN_VALUE) == DEFAULT_NAN_MODEL
    assert parse_nan_value("") == DEFAULT_NAN_MODEL
    assert normalize_nan_model("  glm5.3 ") == "glm5.3"
    assert normalize_nan_model("") == DEFAULT_NAN_MODEL


def test_normalize_base_url_strips_endpoint():
    assert normalize_nan_base_url("") == DEFAULT_NAN_BASE_URL
    assert (
        normalize_nan_base_url("https://api.nan.builders/v1/chat/completions")
        == "https://api.nan.builders/v1"
    )
    assert normalize_nan_base_url("api.nan.builders/v1/") == "https://api.nan.builders/v1"


def test_get_nan_config_model_override(monkeypatch):
    monkeypatch.setenv("NAN_API_KEY", "sk-env")
    cfg = get_nan_config(model="glm5.3-flash")
    assert cfg.model == "glm5.3-flash"
    assert cfg.api_key == "sk-env"
    assert cfg.chat_url.endswith("/v1/chat/completions")
    assert cfg.models_url.endswith("/v1/models")


def test_jpeg_data_url_and_content_parts():
    from forge_img2prompt.nan_client import _vision_user_content

    url = _jpeg_data_url(_red_png())
    assert url.startswith("data:image/jpeg;base64,")
    parts = _vision_user_content([_red_png()], "describe")
    assert parts[0]["type"] == "image_url"
    assert parts[-1] == {"type": "text", "text": "describe"}
    assert _vision_user_content([], "solo texto") == "solo texto"


def test_chat_with_image_builds_openai_payload():
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization") or req.get_header(
            "Authorization"
        )
        captured["body"] = json.loads(req.data.decode())
        return _Resp({"choices": [{"message": {"role": "assistant", "content": "  Un square rojo.  "}}]})

    with patch("forge_img2prompt.nan_client.urllib.request.urlopen", fake_urlopen):
        text = chat_with_image(
            _cfg(),
            system="sys",
            user_text="what color?",
            image=_red_png(),
            max_tokens=32,
        )
    assert text == "Un square rojo."
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["auth"] == "Bearer sk-test"
    body = captured["body"]
    assert body["model"] == "deepseek-v4-flash"
    assert body["stream"] is False
    assert body["max_tokens"] == 32
    assert body["reasoning_effort"] == "none"
    msgs = body["messages"]
    assert msgs[0] == {"role": "system", "content": "sys"}
    parts = msgs[1]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[1] == {"type": "text", "text": "what color?"}


def test_chat_with_multiple_images():
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _Resp({"choices": [{"message": {"content": "ok"}}]})

    imgs = [_red_png(), Image.new("RGB", (8, 8), (0, 255, 0))]
    with patch("forge_img2prompt.nan_client.urllib.request.urlopen", fake_urlopen):
        chat_with_image(_cfg(), system="", user_text="refs", image=imgs, max_tokens=16)
    parts = captured["body"]["messages"][0]["content"]
    assert sum(1 for p in parts if p["type"] == "image_url") == 2


def test_chat_connection_error_message():
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError("refused")

    with patch("forge_img2prompt.nan_client.urllib.request.urlopen", boom):
        with pytest.raises(RuntimeError, match="No se pudo conectar a NaN"):
            chat_with_image(_cfg(), system="", user_text="hi", image=_red_png())


def test_chat_http_error_reports_code():
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.HTTPError(
            url="https://api.nan.builders/v1/chat/completions",
            code=402,
            msg="quota",
            hdrs=None,
            fp=None,
        )

    with patch("forge_img2prompt.nan_client.urllib.request.urlopen", boom):
        with pytest.raises(RuntimeError, match="NaN HTTP 402"):
            chat_with_image(_cfg(), system="", user_text="hi")


def test_list_vision_models_filters_against_curated():
    payload = {
        "data": [
            {"id": "deepseek-v4-flash"},
            {"id": "glm5.3-flash"},
            {"id": "whisper"},  # no visión en el registro curado
            {"id": "kokoro"},
        ]
    }
    with patch(
        "forge_img2prompt.nan_client.urllib.request.urlopen",
        lambda *a, **k: _Resp(payload),
    ):
        models = list_vision_models(_cfg())
    assert models[0] == "deepseek-v4-flash"
    assert "glm5.3-flash" in models
    assert "whisper" not in models
    assert "kokoro" not in models


def test_list_vision_models_fallback_on_error():
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError("refused")

    with patch("forge_img2prompt.nan_client.urllib.request.urlopen", boom):
        models = list_vision_models(_cfg())
    assert models[0] == DEFAULT_NAN_MODEL
    assert set(models) == set(nan_vision_model_ids())


def test_ping_models_ok_and_missing():
    payload = {"data": [{"id": "deepseek-v4-flash"}, {"id": "glm5.3-flash"}]}
    with patch(
        "forge_img2prompt.nan_client.urllib.request.urlopen",
        lambda *a, **k: _Resp(payload),
    ):
        ok, msg = ping_models(_cfg(model="glm5.3-flash"))
    assert ok and "disponible" in msg

    with patch(
        "forge_img2prompt.nan_client.urllib.request.urlopen",
        lambda *a, **k: _Resp(payload),
    ):
        ok2, msg2 = ping_models(_cfg(model="no-existe"))
    assert not ok2 and "no lista" in msg2
