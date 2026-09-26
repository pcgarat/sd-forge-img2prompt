from forge_img2prompt.provider import (
    append_detail,
    detail_is_redundant,
    detail_looks_like_full_scene,
    detail_looks_like_tag_soup,
    normalize_zone_anchor,
    scene_context_snippet,
)


def test_append_detail_basic():
    assert (
        append_detail("A cat on a windowsill.", "soft afternoon light on the fur")
        == "A cat on a windowsill. Soft afternoon light on the fur."
    )


def test_append_detail_adds_period_to_base():
    assert append_detail("A red jacket", "leather texture with scuffs") == (
        "A red jacket. Leather texture with scuffs."
    )


def test_append_detail_empty_detail():
    assert append_detail("Keep me.", "") == "Keep me."


def test_append_detail_empty_base():
    assert append_detail("", "only the crop detail") == "Only the crop detail."


def test_append_detail_normalizes_whitespace():
    assert append_detail("  Hello   world  ", "  more   detail ") == (
        "Hello world. More detail."
    )


def test_detail_is_redundant_paraphrase():
    base = (
        "A young woman wearing a red leather jacket stands by a rainy window, "
        "soft daylight on her face, cinematic atmosphere."
    )
    echo = (
        "A young woman wearing a red leather jacket stands by a rainy window "
        "with soft daylight and cinematic atmosphere"
    )
    assert detail_is_redundant(base, echo, max_shared=6)


def test_detail_is_redundant_specific_ok():
    base = (
        "A young woman wearing a red leather jacket stands by a rainy window, "
        "soft daylight on her face, cinematic atmosphere."
    )
    zone = "Worn red leather jacket with scuffed elbows and a brass zipper."
    assert not detail_is_redundant(base, zone, max_shared=10)


def test_detail_is_redundant_allows_contextual_anchor():
    base = (
        "Un hombre y una mujer sentados en un estudio con paredes de madera, "
        "mirándose con una sonrisa. La mujer, con cabello castaño y vestida con "
        "un vestido oscuro, sonríe ampliamente mientras el hombre, con camisa "
        "blanca y mangas enrolladas, la observa con una expresión amable."
    )
    zone = (
        "El hombre de la izquierda muestra barba canosa corta, cabello oscuro "
        "con canas en las sienes y piel marcada alrededor de los ojos."
    )
    assert not detail_is_redundant(base, zone, max_shared=10)


def test_detail_is_redundant_rejects_near_copy():
    base = (
        "Un hombre con camisa blanca y mangas enrolladas observa con una "
        "expresión amable a una mujer."
    )
    zone = (
        "El hombre con camisa blanca y mangas enrolladas observa con una "
        "expresión amable."
    )
    assert detail_is_redundant(base, zone, max_shared=6)


def test_detail_is_redundant_zero_never_discards():
    base = "Un hombre con camisa blanca observa a una mujer."
    zone = "Un hombre con camisa blanca observa a una mujer con expresión amable."
    assert not detail_is_redundant(base, zone, max_shared=0)


def test_detail_looks_like_full_scene_rejects_couple():
    text = (
        "un hombre con cabello oscuro y barba corta, vestido con una camisa blanca, "
        "y una mujer con cabello castaño claro, sentados en una mesa con pared de madera"
    )
    assert detail_looks_like_full_scene(text, word_max=25)


def test_detail_looks_like_full_scene_accepts_face():
    text = (
        "El hombre de la izquierda muestra barba canosa corta, cabello oscuro "
        "con canas en las sienes y piel marcada alrededor de los ojos."
    )
    assert not detail_looks_like_full_scene(text, word_max=30)


def test_detail_looks_like_tag_soup():
    soup = (
        "barba, cabello canoso, piel arrugada, camisa blanca, ojos marrones, "
        "expresión serena, rostro adulto"
    )
    assert detail_looks_like_tag_soup(soup)
    assert not detail_looks_like_tag_soup(
        "El hombre de la izquierda tiene barba canosa y ojos marrones."
    )


def test_normalize_zone_anchor():
    assert normalize_zone_anchor("el hombre de la izquierda.") == (
        "el hombre de la izquierda"
    )
    assert normalize_zone_anchor('"the woman on the right"') == "the woman on the right"


def test_scene_context_snippet_truncates():
    long = "Primera frase. " + ("palabra " * 80)
    snip = scene_context_snippet(long, max_chars=80)
    assert len(snip) <= 90
    assert "Primera frase" in snip
