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
