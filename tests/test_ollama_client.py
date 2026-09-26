"""Tests del cliente Ollama (sin red real)."""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image

from forge_img2prompt.ollama_client import (
    chat_with_image,
    image_to_b64,
    list_vision_models,
    ping_tags,
)
from forge_img2prompt.ollama_settings import (
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_VALUE,
    OLLAMA_VISION_FALLBACK,
    OllamaConfig,
    default_base_url,
    get_ollama_config,
    is_cloud_model,
    is_ollama_value,
    normalize_ollama_model,
    parse_ollama_value,
)
from forge_img2prompt.vl_catalog import choice_by_value, list_vl_models, ollama_choice


def _red_png() -> Image.Image:
    return Image.new("RGB", (16, 16), (220, 40, 40))


def test_image_to_b64_roundtrip():
    b64 = image_to_b64(_red_png())
    assert isinstance(b64, str) and len(b64) > 20
    raw = __import__("base64").b64decode(b64)
    img = Image.open(BytesIO(raw))
    assert img.size[0] >= 1


def test_is_ollama_value():
    assert is_ollama_value(OLLAMA_VALUE)
    assert is_ollama_value("ollama:qwen3-vl:8b-instruct")
    assert not is_ollama_value("/data/Models/TextEncoders/foo")


def test_parse_ollama_value():
    assert parse_ollama_value("ollama:kimi-k3:cloud") == "kimi-k3:cloud"
    assert parse_ollama_value("ollama:") == DEFAULT_OLLAMA_MODEL
    assert parse_ollama_value(OLLAMA_VALUE) == DEFAULT_OLLAMA_MODEL


def test_ollama_choice_in_catalog():
    payload = {
        "models": [
            {
                "name": "qwen3-vl:8b-instruct",
                "digest": "aaa",
                "capabilities": ["vision", "completion"],
            },
            {
                "name": "kimi-k3:cloud",
                "digest": "bbb",
                "capabilities": ["vision", "completion"],
            },
            {
                "name": "deepseek-v4-flash:cloud",
                "digest": "ccc",
                "capabilities": ["completion", "thinking"],
            },
        ]
    }

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()

    with patch(
        "forge_img2prompt.ollama_client.urllib.request.urlopen",
        lambda *a, **k: _Resp(),
    ):
        cats = list_vl_models()
    ollama = [c for c in cats if c.is_ollama]
    assert len(ollama) == 2
    assert ollama[0].value == "ollama:qwen3-vl:8b-instruct"
    assert ollama[1].value == "ollama:kimi-k3:cloud"
    got = choice_by_value("ollama:kimi-k3:cloud", cats)
    assert got is not None and got.is_ollama and got.hf_id == "kimi-k3:cloud"


def test_list_vision_models_filters_and_dedupes():
    payload = {
        "models": [
            {
                "name": "huihui-qwen3-vl-8b-abliterated:q6_k",
                "digest": "same",
                "capabilities": ["vision", "completion"],
            },
            {
                "name": "huihui-qwen3vl-8b-abl:q6",
                "digest": "same",
                "capabilities": ["vision", "completion"],
            },
            {
                "name": "gpt-oss:120b-cloud",
                "digest": "nox",
                "capabilities": ["completion", "thinking"],
            },
            {
                "name": "gemma4:31b-cloud",
                "digest": "gem",
                "capabilities": ["vision", "completion"],
            },
        ]
    }

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()

    cfg = OllamaConfig(
        base_url="http://example.invalid",
        model="x",
        api_key="",
        timeout=5,
    )
    with patch(
        "forge_img2prompt.ollama_client.urllib.request.urlopen",
        lambda *a, **k: _Resp(),
    ):
        names = list_vision_models(cfg)
    assert "gpt-oss:120b-cloud" not in names
    assert "gemma4:31b-cloud" in names
    # Alias más corto gana el digest duplicado
    assert "huihui-qwen3vl-8b-abl:q6" in names
    assert "huihui-qwen3-vl-8b-abliterated:q6_k" not in names


def test_list_vision_models_fallback_on_error():
    cfg = OllamaConfig(
        base_url="http://127.0.0.1:9",
        model="x",
        api_key="",
        timeout=1,
    )

    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError("refused")

    with patch("forge_img2prompt.ollama_client.urllib.request.urlopen", boom):
        names = list_vision_models(cfg)
    assert names == list(OLLAMA_VISION_FALLBACK)
    assert "deepseek-v4.1-flash:cloud" not in names


def test_chat_with_image_builds_native_payload(monkeypatch):
    cfg = OllamaConfig(
        base_url="http://127.0.0.1:11434",
        model="qwen3-vl:8b-instruct",
        api_key="",
        timeout=30,
    )
    captured: dict = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {"message": {"role": "assistant", "content": "  Red square.  "}}
            ).encode()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode())
        return _Resp()

    with patch("forge_img2prompt.ollama_client.urllib.request.urlopen", fake_urlopen):
        text = chat_with_image(
            cfg,
            system="sys",
            user_text="what color?",
            image=_red_png(),
            num_predict=32,
        )
    assert text == "Red square."
    assert captured["url"].endswith("/api/chat")
    assert captured["body"]["model"] == "qwen3-vl:8b-instruct"
    assert captured["body"]["stream"] is False
    msgs = captured["body"]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "images" in msgs[1] and len(msgs[1]["images"]) == 1


def test_chat_connection_error_message():
    cfg = OllamaConfig(
        base_url="http://127.0.0.1:9",
        model="x",
        api_key="",
        timeout=1,
    )

    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError("refused")

    with patch("forge_img2prompt.ollama_client.urllib.request.urlopen", boom):
        with pytest.raises(RuntimeError, match="No se pudo conectar a Ollama"):
            chat_with_image(cfg, system="", user_text="hi", image=_red_png())


def test_ping_tags_ok():
    cfg = OllamaConfig(
        base_url="http://example.invalid",
        model="qwen3-vl:8b-instruct",
        api_key="",
        timeout=5,
    )

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {"models": [{"name": "qwen3-vl:8b-instruct"}]}
            ).encode()

    with patch(
        "forge_img2prompt.ollama_client.urllib.request.urlopen",
        lambda *a, **k: _Resp(),
    ):
        ok, msg = ping_tags(cfg)
    assert ok
    assert "disponible" in msg


def test_default_base_url_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://10.0.0.5:11434")
    assert default_base_url() == "http://10.0.0.5:11434"
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.5:11434")
    assert default_base_url() == "http://10.0.0.5:11434"


def test_ollama_choice_label():
    c = ollama_choice("kimi-k3:cloud")
    assert "Ollama" in c.label
    assert "cloud" in c.label
    assert c.kind == "ollama"
    assert c.value == "ollama:kimi-k3:cloud"


def test_vision_fallback_has_no_deepseek():
    assert DEFAULT_OLLAMA_MODEL == OLLAMA_VISION_FALLBACK[0]
    assert all("deepseek" not in t for t in OLLAMA_VISION_FALLBACK)
    assert "kimi-k3:cloud" in OLLAMA_VISION_FALLBACK


def test_normalize_and_config_model_override():
    assert normalize_ollama_model("  kimi-k3:cloud ") == "kimi-k3:cloud"
    assert normalize_ollama_model("") == DEFAULT_OLLAMA_MODEL
    assert is_cloud_model("kimi-k3:cloud")
    assert is_cloud_model("gemma4:31b-cloud")
    assert is_cloud_model("mistral-large-3:675b-cloud")
    assert not is_cloud_model("qwen3-vl:8b-instruct")
    cfg = get_ollama_config(model="gemma4:31b-cloud")
    assert cfg.model == "gemma4:31b-cloud"


def test_ping_tags_cloud_ok_without_local_tag():
    cfg = OllamaConfig(
        base_url="http://example.invalid",
        model="kimi-k3:cloud",
        api_key="sk-test",
        timeout=5,
    )

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"models": [{"name": "qwen3-vl:8b-instruct"}]}).encode()

    with patch(
        "forge_img2prompt.ollama_client.urllib.request.urlopen",
        lambda *a, **k: _Resp(),
    ):
        ok, msg = ping_tags(cfg)
    assert ok
    assert "cloud" in msg.lower()
    assert "API key presente" in msg


def test_chat_sends_bearer_when_api_key(monkeypatch):
    cfg = OllamaConfig(
        base_url="https://ollama.com",
        model="kimi-k3:cloud",
        api_key="sk-secret",
        timeout=30,
    )
    captured: dict = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {"message": {"role": "assistant", "content": "ok"}}
            ).encode()

    def fake_urlopen(req, timeout=None):
        captured["auth"] = req.headers.get("Authorization") or req.get_header(
            "Authorization"
        )
        return _Resp()

    with patch("forge_img2prompt.ollama_client.urllib.request.urlopen", fake_urlopen):
        text = chat_with_image(
            cfg,
            system="",
            user_text="hi",
            image=_red_png(),
            num_predict=16,
        )
    assert text == "ok"
    assert captured["auth"] == "Bearer sk-secret"
