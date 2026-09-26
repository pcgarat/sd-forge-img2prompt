from PIL import Image

from forge_img2prompt.mask_crop import (
    bbox_from_mask,
    crop_from_editor,
    crop_masked_content,
    crop_masked_region,
    editor_to_rgb,
    mask_from_composite_diff,
    mask_from_layers,
)


def _rgb(w: int = 200, h: int = 200, color=(10, 20, 30)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def _rgba_brush(w: int, h: int, box: tuple[int, int, int, int]) -> Image.Image:
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    paint = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (255, 0, 0, 255))
    layer.paste(paint, (box[0], box[1]))
    return layer


def test_editor_to_rgb_from_dict():
    bg = _rgb()
    assert editor_to_rgb({"background": bg, "layers": [], "composite": bg}).size == (200, 200)


def test_editor_to_rgb_plain_image():
    assert editor_to_rgb(_rgb(50, 40)).size == (50, 40)


def test_editor_to_rgb_none():
    assert editor_to_rgb(None) is None
    assert editor_to_rgb({}) is None


def test_mask_from_layers_empty():
    assert mask_from_layers(None) is None
    assert mask_from_layers([]) is None


def test_mask_and_bbox():
    layer = _rgba_brush(100, 100, (20, 30, 40, 50))
    mask = mask_from_layers([layer])
    assert mask is not None
    assert mask.mode == "L"
    box = bbox_from_mask(mask, pad=0)
    assert box == (20, 30, 40, 50)


def test_bbox_empty_mask():
    mask = Image.new("L", (50, 50), 0)
    assert bbox_from_mask(mask) is None


def test_crop_masked_region():
    rgb = _rgb(100, 100, (1, 2, 3))
    for x in range(20, 40):
        for y in range(30, 50):
            rgb.putpixel((x, y), (200, 100, 50))
    layer = _rgba_brush(100, 100, (20, 30, 40, 50))
    mask = mask_from_layers([layer])
    assert mask is not None
    crop = crop_masked_region(rgb, mask, pad=0, min_side=1)
    assert crop is not None
    assert crop.size == (20, 20)
    assert crop.getpixel((0, 0)) == (200, 100, 50)


def test_crop_masked_content_blacks_outside():
    rgb = _rgb(100, 100, (1, 2, 3))
    for x in range(20, 40):
        for y in range(30, 50):
            rgb.putpixel((x, y), (200, 100, 50))
    mask = Image.new("L", (100, 100), 0)
    for x in range(20, 40):
        for y in range(30, 50):
            mask.putpixel((x, y), 255)
    crop = crop_masked_content(rgb, mask, pad=0, min_side=1)
    assert crop is not None
    assert crop.size == (20, 20)
    assert crop.getpixel((0, 0)) == (200, 100, 50)


def test_crop_empty_mask_returns_none():
    rgb = _rgb(80, 80)
    mask = Image.new("L", (80, 80), 0)
    assert crop_masked_region(rgb, mask) is None
    assert crop_masked_content(rgb, mask) is None


def test_mask_from_composite_diff():
    bg = _rgb(80, 80, (10, 10, 10))
    composite = bg.copy()
    for x in range(10, 40):
        for y in range(10, 40):
            composite.putpixel((x, y), (200, 0, 200))
    mask = mask_from_composite_diff(bg, composite)
    assert mask is not None
    box = bbox_from_mask(mask, pad=0)
    assert box is not None
    assert box[0] <= 10 and box[2] >= 40


def test_crop_from_editor_layers():
    bg = _rgb(120, 120)
    layer = _rgba_brush(120, 120, (40, 40, 60, 60))
    crop = crop_from_editor(
        {"background": bg, "layers": [layer], "composite": bg},
        pad=0,
        min_side=1,
        blackout=True,
    )
    assert crop is not None
    assert crop.size == (20, 20)


def test_crop_from_editor_vl_gray_mask():
    bg = _rgb(100, 100, (1, 2, 3))
    for x in range(20, 40):
        for y in range(30, 50):
            bg.putpixel((x, y), (200, 100, 50))
    # Paint only left half of the colored square; right half stays in bbox
    layer = _rgba_brush(100, 100, (20, 30, 30, 50))
    editor = {"background": bg, "layers": [layer], "composite": bg}
    from forge_img2prompt.mask_crop import VL_OUTSIDE_RGB

    vl = crop_from_editor(
        editor, pad=0, min_side=1, blackout=True, outside=VL_OUTSIDE_RGB
    )
    clean = crop_from_editor(editor, pad=0, min_side=1, blackout=False)
    assert vl is not None and clean is not None
    assert vl.size == clean.size == (10, 20)
    # Painted pixel kept; with gray mask the crop is only the painted bbox
    assert vl.getpixel((0, 0)) == (200, 100, 50)


def test_crop_from_editor_vl_hides_unpainted_in_bbox():
    """Unpainted neighbors inside a loose bbox must not reach the VL."""
    from forge_img2prompt.mask_crop import VL_OUTSIDE_RGB

    bg = _rgb(80, 80, (9, 9, 9))
    # Two subjects side by side
    for x in range(10, 30):
        for y in range(10, 40):
            bg.putpixel((x, y), (200, 50, 50))  # left subject
    for x in range(40, 60):
        for y in range(10, 40):
            bg.putpixel((x, y), (50, 50, 200))  # right subject
    # Paint only the left subject
    layer = _rgba_brush(80, 80, (10, 10, 30, 40))
    editor = {"background": bg, "layers": [layer], "composite": bg}
    # Force a wide crop via pad so the bbox could include the right subject
    vl = crop_from_editor(
        editor, pad=25, min_side=1, blackout=True, outside=VL_OUTSIDE_RGB
    )
    assert vl is not None
    # Right subject blue must not appear; those pixels are grayed out
    blues = [
        vl.getpixel((x, y))
        for x in range(vl.size[0])
        for y in range(vl.size[1])
        if vl.getpixel((x, y)) == (50, 50, 200)
    ]
    assert blues == []
    assert VL_OUTSIDE_RGB in {
        vl.getpixel((x, y)) for x in range(vl.size[0]) for y in range(vl.size[1])
    }


def test_crop_from_editor_composite_fallback():
    bg = _rgb(100, 100, (5, 5, 5))
    composite = bg.copy()
    for x in range(30, 70):
        for y in range(30, 70):
            composite.putpixel((x, y), (255, 0, 200))
    crop = crop_from_editor(
        {"background": bg, "layers": [], "composite": composite},
        pad=0,
        min_side=1,
    )
    assert crop is not None
    assert crop.size[0] >= 40


def test_crop_from_editor_no_paint():
    bg = _rgb(80, 80)
    assert crop_from_editor({"background": bg, "layers": [], "composite": bg}) is None
