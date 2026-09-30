"""Referencias opcionales foto 1–3 para notas Generate / zona."""

from __future__ import annotations

from PIL import Image

from forge_img2prompt.provider import (
    REF_IMAGES_MAX,
    active_ref_pairs,
    normalize_ref_images,
    normalize_ref_slots,
    ref_images_instruction,
    ref_label,
)
from forge_img2prompt.vl_provider import (
    _build_detail_user_text,
    _build_user_text,
    _pack_primary_and_refs,
    _vl_user_content,
)


def _rgb(color=(10, 20, 30)) -> Image.Image:
    return Image.new("RGB", (8, 8), color)


def test_ref_label_is_one_based():
    assert ref_label(1) == "foto 1"
    assert ref_label(3) == "foto 3"


def test_normalize_slots_preserve_index():
    """Solo Foto 2 rellena → etiqueta foto 2 (no se renumera a foto 1)."""
    b = _rgb((0, 1, 0))
    slots = normalize_ref_slots(None, b, None)
    assert slots[0] is None
    assert slots[1] is not None
    assert slots[2] is None
    pairs = active_ref_pairs(slots)
    assert pairs == [(2, slots[1])]


def test_normalize_images_dense_list():
    a, b = _rgb((1, 0, 0)), _rgb((0, 1, 0))
    out = normalize_ref_images(None, b, a)
    assert len(out) == 2
    assert out[0].getpixel((0, 0)) == (0, 1, 0)


def test_normalize_caps_at_max():
    imgs = [_rgb((i, i, i)) for i in range(5)]
    out = normalize_ref_images(*imgs)
    assert len(out) == REF_IMAGES_MAX
    assert len(normalize_ref_slots(*imgs)) == REF_IMAGES_MAX


def test_normalize_gallery_tuple():
    img = _rgb((9, 9, 9))
    out = normalize_ref_images((img, "caption"))
    assert len(out) == 1
    assert out[0].mode == "RGB"


def test_instruction_empty_when_no_refs():
    assert ref_images_instruction(0) == ""
    assert ref_images_instruction(indices=[]) == ""


def test_instruction_uses_slot_indices():
    text = ref_images_instruction(indices=[2, 3])
    assert "labeled foto 2, foto 3" in text
    assert "labeled foto 1" not in text


def test_build_user_text_includes_ref_block():
    text = _build_user_text(
        "el sombrero de la foto 2", "krea2", ref_indices=[2]
    )
    assert "foto 2" in text
    assert "el sombrero de la foto 2" in text


def test_build_detail_user_text_includes_ref_block():
    text = _build_detail_user_text(
        "usa la textura de la foto 2", ref_indices=[1, 2]
    )
    assert "foto 1" in text and "foto 2" in text
    assert "textura de la foto 2" in text


def test_pack_primary_preserves_slot_labels():
    primary = _rgb((1, 1, 1))
    slots = (None, _rgb((2, 2, 2)), None)
    packed, indices = _pack_primary_and_refs(primary, ref_slots=slots)
    assert len(packed) == 2
    assert indices == [2]


def test_vl_user_content_labels_by_slot():
    images = [_rgb((1, 1, 1)), _rgb((2, 2, 2))]
    content = _vl_user_content(images, "user notes here", ref_indices=[2])
    texts = [c["text"] for c in content if c.get("type") == "text"]
    assert "Primary image." in texts
    assert "foto 2" in texts
    assert "foto 1" not in texts
    assert texts[-1] == "user notes here"
