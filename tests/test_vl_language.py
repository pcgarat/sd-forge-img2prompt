from forge_img2prompt.provider import LANG_EN, LANG_ES
from forge_img2prompt.vl_provider import _build_user_text, _language_instruction


def test_language_instruction_spanish():
    text = _language_instruction(LANG_ES)
    assert "Spanish" in text


def test_language_instruction_english():
    text = _language_instruction(LANG_EN)
    assert "English" in text


def test_user_text_includes_language():
    es = _build_user_text("prioridad chaqueta", "krea2", LANG_ES)
    en = _build_user_text("jacket focus", "krea2", LANG_EN)
    assert "Spanish" in es
    assert "English" in en
    assert "prioridad chaqueta" in es
    assert "jacket focus" in en
