from forge_img2prompt.provider import (
    DETAIL_WORDS_BOUNDS,
    DETAIL_WORDS_DEFAULT,
    GEN_WORDS_BOUNDS,
    GEN_WORDS_DEFAULT,
    OVERLAP_DISCARD_BOUNDS,
    OVERLAP_DISCARD_DEFAULT,
    clamp_overlap_discard,
    clamp_text_to_words,
    clamp_word_range,
    count_words,
    max_tokens_for_words,
    shared_content_count,
)


def test_clamp_defaults_on_none():
    assert clamp_word_range(
        None, None, bounds=GEN_WORDS_BOUNDS, default=GEN_WORDS_DEFAULT
    ) == GEN_WORDS_DEFAULT


def test_clamp_detail_bounds():
    lo, hi = clamp_word_range(
        0, 1000, bounds=DETAIL_WORDS_BOUNDS, default=DETAIL_WORDS_DEFAULT
    )
    assert (lo, hi) == DETAIL_WORDS_BOUNDS


def test_count_words_empty():
    assert count_words("") == 0
    assert count_words("  a   b  ") == 2


def test_max_tokens_for_words_clamped():
    assert max_tokens_for_words(1, floor=32, ceil=512) == 32
    assert max_tokens_for_words(1000, floor=32, ceil=200) == 200
    # Detail: budget stays near word_max, not 2×+
    assert max_tokens_for_words(20, floor=24, ceil=320) <= 50
    assert max_tokens_for_words(200, floor=24, ceil=320) == 282


def test_clamp_text_to_words_hard_cap():
    text = "uno dos tres cuatro cinco seis siete ocho nueve diez"
    assert clamp_text_to_words(text, 5) == "uno dos tres cuatro cinco"
    assert count_words(clamp_text_to_words(text, 5)) == 5


def test_clamp_text_to_words_prefers_sentence():
    text = "Primera frase corta. Luego viene mucho más texto de relleno aquí."
    out = clamp_text_to_words(text, 8)
    assert out.endswith(".")
    assert "Primera frase corta." in out
    assert count_words(out) <= 8


def test_clamp_overlap_discard():
    assert clamp_overlap_discard(0) == 0
    assert clamp_overlap_discard(-3) == OVERLAP_DISCARD_BOUNDS[0]
    assert clamp_overlap_discard(999) == OVERLAP_DISCARD_BOUNDS[1]
    assert clamp_overlap_discard(None) == OVERLAP_DISCARD_DEFAULT


def test_shared_content_count_ignores_stopwords():
    n = shared_content_count(
        "El hombre con una camisa blanca",
        "El hombre de la izquierda con camisa oscura",
    )
    assert n >= 2  # hombre, camisa at least
