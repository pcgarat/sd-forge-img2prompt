"""Enrutado Ollama/NaN, ramas de error y salvaguardas de transporte."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from forge_img2prompt.nan_settings import NanConfig
from forge_img2prompt.ollama_settings import OllamaConfig
from forge_img2prompt.provider import PromptRequest
from forge_img2prompt.stack import detect_stack
from forge_img2prompt.vl_catalog import _nan_choice_for, _ollama_choice_for


def _krea():
    return detect_stack("krea2_turbo.safetensors", "qwen3vl_4b.safetensors")


def _request() -> PromptRequest:
    return PromptRequest(None, "un gato en un tejado", _krea(), "es")


# --- Enrutado CompositeProvider ---------------------------------------------


def test_provider_for_routes_nan_and_ollama():
    from forge_img2prompt.ollama_vl import NanVLProvider, OllamaVLProvider
    from forge_img2prompt.vl_provider import CompositeProvider

    comp = CompositeProvider()
    assert isinstance(comp._provider_for(_nan_choice_for("glm5.3-flash")), NanVLProvider)
    assert isinstance(
        comp._provider_for(_ollama_choice_for("qwen3-vl:8b-instruct")), OllamaVLProvider
    )


def test_composite_generate_routes_to_nan(monkeypatch):
    from forge_img2prompt.vl_provider import CompositeProvider

    comp = CompositeProvider()
    choice = _nan_choice_for("glm5.3-flash")
    seen: dict = {}

    class _Recorder:
        def generate(self, request, choice, *, progress=None):
            seen["choice"] = choice
            return "NAN-RESULT"

    comp.nan = _Recorder()
    monkeypatch.setattr(
        "forge_img2prompt.vl_provider.choice_by_value", lambda *a, **k: choice
    )
    out = comp.generate(_request(), vl_value="nan:glm5.3-flash")
    assert out == "NAN-RESULT"
    assert seen["choice"].is_nan and seen["choice"].hf_id == "glm5.3-flash"


def test_composite_generate_routes_to_ollama(monkeypatch):
    from forge_img2prompt.vl_provider import CompositeProvider

    comp = CompositeProvider()
    choice = _ollama_choice_for("qwen3-vl:8b-instruct")
    seen: dict = {}

    class _Recorder:
        def generate(self, request, choice, *, progress=None):
            seen["choice"] = choice
            return "OLLAMA-RESULT"

    comp.ollama = _Recorder()
    monkeypatch.setattr(
        "forge_img2prompt.vl_provider.choice_by_value", lambda *a, **k: choice
    )
    out = comp.generate(_request(), vl_value="ollama:qwen3-vl:8b-instruct")
    assert out == "OLLAMA-RESULT"
    assert seen["choice"].is_ollama


# --- Transporte del proveedor NaN -------------------------------------------


def test_nan_runner_passes_max_tokens_and_config():
    from forge_img2prompt.ollama_vl import NanVLProvider

    captured: dict = {}

    def fake_chat(cfg, *, system, user_text, image, max_tokens):
        captured.update(
            cfg=cfg, system=system, user_text=user_text, image=image, max_tokens=max_tokens
        )
        return "ok"

    import forge_img2prompt.nan_client as nan_client

    original = nan_client.chat_with_image
    nan_client.chat_with_image = fake_chat
    try:
        out = NanVLProvider()._run(
            None, "sys", "brief", max_new_tokens=7, model="glm5.3-flash"
        )
    finally:
        nan_client.chat_with_image = original
    assert out == "ok"
    assert captured["max_tokens"] == 7
    assert captured["system"] == "sys"
    assert isinstance(captured["cfg"], NanConfig)
    assert captured["cfg"].model == "glm5.3-flash"


def test_nan_error_hint_marks_premium():
    from forge_img2prompt.ollama_vl import NanVLProvider

    cfg = NanConfig(
        base_url="https://api.nan.builders/v1",
        model="glm5.3",
        api_key="sk-x",
        timeout=30,
    )
    hint = NanVLProvider()._error_hint(RuntimeError("boom"), cfg, _nan_choice_for("glm5.3"))
    assert "premium" in hint
    assert "glm5.3" in hint and "api.nan.builders" in hint


def test_ollama_error_hint_keeps_model_and_url_context():
    from forge_img2prompt.ollama_vl import OllamaVLProvider

    cfg = OllamaConfig(
        base_url="http://127.0.0.1:11434",
        model="qwen3-vl:8b-instruct",
        api_key="",
        timeout=30,
    )
    hint = OllamaVLProvider()._error_hint(RuntimeError("boom"), cfg, _ollama_choice_for("x"))
    assert "qwen3-vl:8b-instruct" in hint
    assert "127.0.0.1:11434" in hint
    assert "boom" in hint


# --- Salvaguarda de redirects (Bearer) --------------------------------------


def test_nan_redirect_handler_blocks_instead_of_forwarding_token():
    from forge_img2prompt.nan_client import _NoRedirect

    req = urllib.request.Request(
        "https://api.nan.builders/v1/chat/completions",
        headers={"Authorization": "Bearer sk-secret"},
    )
    with pytest.raises(urllib.error.HTTPError) as err:
        _NoRedirect().redirect_request(req, None, 302, "Found", {}, "https://evil.example/x")
    assert err.value.code == 302


def test_ollama_redirect_handler_blocks_instead_of_forwarding_token():
    from forge_img2prompt.ollama_client import _NoRedirect

    req = urllib.request.Request(
        "https://ollama.com/api/chat", headers={"Authorization": "Bearer sk-secret"}
    )
    with pytest.raises(urllib.error.HTTPError):
        _NoRedirect().redirect_request(req, None, 307, "Temporary Redirect", {}, "https://evil/x")
