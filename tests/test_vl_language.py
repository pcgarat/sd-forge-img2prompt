from __future__ import annotations

import json
from unittest.mock import patch

from forge_img2prompt.provider import LANG_EN, LANG_ES
from forge_img2prompt.vl_provider import (
    _build_detail_user_text,
    _build_identify_user_text,
    _build_user_text,
    _language_instruction,
    _word_range_instruction,
)


def test_language_instruction_spanish():
    text = _language_instruction(LANG_ES)
    assert "Spanish" in text


def test_language_instruction_english():
    text = _language_instruction(LANG_EN)
    assert "English" in text


def test_user_text_includes_language_and_word_range():
    es = _build_user_text("prioridad chaqueta", "krea2", LANG_ES, word_min=40, word_max=80)
    en = _build_user_text("jacket focus", "krea2", LANG_EN, word_min=40, word_max=80)
    assert "Spanish" in es
    assert "English" in en
    assert "prioridad chaqueta" in es
    assert "jacket focus" in en
    assert "40" in es and "80" in es
    assert "40" in en and "80" in en


def test_detail_user_text_uses_anchor_not_full_base():
    text = _build_detail_user_text(
        "",
        LANG_EN,
        anchor="the man on the left",
        word_min=8,
        word_max=20,
    )
    low = text.lower()
    assert "subject anchor" in low
    assert "the man on the left" in low
    assert "comma-separated tags" in low
    assert "gray" in low
    assert "a man and a woman sitting" not in low
    assert "8" in text and "20" in text
    assert "Hard limit" in text or "at most 20" in text


def test_identify_user_text_uses_scene_snippet_only():
    text = _build_identify_user_text(
        "A man and a woman sit in a studio.",
        LANG_ES,
    )
    assert "Scene context for identity only" in text
    assert "A man and a woman sit in a studio." in text
    assert "Spanish" in text


def test_detail_user_text_includes_zone_notes():
    text = _build_detail_user_text(
        "costura deshilachada",
        LANG_ES,
        anchor="la chaqueta",
        word_min=5,
        word_max=15,
    )
    assert "costura deshilachada" in text
    assert "la chaqueta" in text
    assert "Spanish" in text


def test_word_range_instruction_exact():
    assert "exactly 12" in _word_range_instruction(12, 12)
    assert "LENGTH" in _word_range_instruction(12, 12)
    assert "UI sliders" in _word_range_instruction(40, 80)


def test_caption_system_style_rules():
    from forge_img2prompt.vl_provider import _CAPTION_SYSTEM

    low = _CAPTION_SYSTEM.lower()
    assert "one paragraph" not in low
    assert "word-count" in low or "word count" in low.replace("-", " ")
    assert "natural language" in low
    assert "prose" not in low
    assert "no hay" in low or "there is no" in low
    assert "concrete" in low and "concise" in low
    assert "only if" in low or "actually visible" in low


def test_inside_out_strategy_prompts():
    from forge_img2prompt.provider import STRATEGY_INSIDE_OUT, STRATEGY_PROSE
    from forge_img2prompt.vl_provider import (
        _build_notes_only_user_text,
        _build_user_text,
        _caption_system_for,
        _notes_system_for,
    )

    prose_sys = _caption_system_for(STRATEGY_PROSE).lower()
    inside_sys = _caption_system_for(STRATEGY_INSIDE_OUT).lower()
    assert "subject, action/pose" in prose_sys or "subject, action" in prose_sys
    assert "inside out" in inside_sys or "inside-out" in inside_sys
    assert "hyperrealistic" in inside_sys
    assert "micro-textures" in inside_sys or "micro textures" in inside_sys.replace("-", " ")
    assert "technical wrap" in inside_sys

    user = _build_user_text(
        "prioridad chaqueta",
        "krea2",
        LANG_ES,
        strategy=STRATEGY_INSIDE_OUT,
        word_min=40,
        word_max=80,
    ).lower()
    assert "inside-out" in user or "inside out" in user
    assert "clothing" in user or "accessories" in user

    notes_sys = _notes_system_for(STRATEGY_INSIDE_OUT).lower()
    assert "inside out" in notes_sys or "inside-out" in notes_sys
    notes_user = _build_notes_only_user_text(
        "un gato naranja",
        "krea2",
        LANG_ES,
        strategy=STRATEGY_INSIDE_OUT,
        word_min=40,
        word_max=80,
    ).lower()
    assert "inside-out" in notes_user or "inside out" in notes_user


def test_normalize_strategy_aliases():
    from forge_img2prompt.provider import (
        STRATEGY_INSIDE_OUT,
        STRATEGY_PROSE,
        normalize_strategy,
        strategy_label,
    )

    assert normalize_strategy("prose") == STRATEGY_PROSE
    assert normalize_strategy("Prosa") == STRATEGY_PROSE
    assert normalize_strategy("inside_out") == STRATEGY_INSIDE_OUT
    assert normalize_strategy("Dentro → fuera") == STRATEGY_INSIDE_OUT
    assert normalize_strategy("dentro-fuera") == STRATEGY_INSIDE_OUT
    assert normalize_strategy("weird") == STRATEGY_PROSE
    assert strategy_label(STRATEGY_INSIDE_OUT) == "Dentro → fuera"


def test_family_hint_has_no_length():
    from forge_img2prompt.vl_provider import _family_hint

    for fam in ("krea2", "klein9b"):
        low = _family_hint(fam).lower()
        assert "word" not in low
        assert "paragraph" not in low
        assert "short" not in low
        assert "long" not in low
        assert "length" not in low
        assert "prose" not in low


def test_notes_only_user_text_expands_brief():
    from forge_img2prompt.vl_provider import _NOTES_SYSTEM, _build_notes_only_user_text

    text = _build_notes_only_user_text(
        "un gato naranja en un tejado",
        "krea2",
        LANG_ES,
        word_min=40,
        word_max=80,
    )
    low = text.lower()
    assert "no main photograph" in low
    assert "gato naranja" in low
    assert "user brief" in low
    assert "40" in text and "80" in text
    assert "natural language" in _NOTES_SYSTEM.lower()
    assert "prose" not in _NOTES_SYSTEM.lower()


def test_chat_text_only_omits_images(monkeypatch):
    from forge_img2prompt.ollama_client import chat_with_image
    from forge_img2prompt.ollama_settings import OllamaConfig

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
                {"message": {"role": "assistant", "content": "A cat on a roof."}}
            ).encode()

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _Resp()

    with patch("forge_img2prompt.ollama_client.urllib.request.urlopen", fake_urlopen):
        text = chat_with_image(
            cfg, system="sys", user_text="brief", image=None, num_predict=32
        )
    assert text == "A cat on a roof."
    assert "images" not in captured["body"]["messages"][-1]
    assert captured["body"].get("keep_alive") == 0
