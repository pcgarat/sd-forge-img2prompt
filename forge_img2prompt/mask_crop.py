"""Crop a region from an ImageEditor value using the brush layer as mask."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageChops

EditorValue = dict[str, Any]

# Ignore tiny brush noise / JPEG dither when comparing composite vs background
_DIFF_THRESHOLD = 18
_MIN_MASK_PIXELS = 32
# Mid-gray fill for VL crops: pure black reads as an inpaint hole and makes
# Qwen invent the rest of the scene; a flat gray + prompt says "ignore this".
VL_OUTSIDE_RGB: tuple[int, int, int] = (110, 110, 110)


def editor_to_rgb(editor: EditorValue | Image.Image | None) -> Image.Image | None:
    """Return RGB background from an ImageEditor dict or a plain PIL image."""
    if editor is None:
        return None
    if isinstance(editor, Image.Image):
        img = editor
    elif isinstance(editor, dict):
        img = editor.get("background")
        if img is None:
            img = editor.get("composite")
        if not isinstance(img, Image.Image):
            return None
    else:
        return None
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _layer_alpha(layer: Image.Image) -> Image.Image:
    if layer.mode == "RGBA":
        return layer.split()[-1]
    if layer.mode == "L":
        return layer
    return layer.convert("L")


def mask_from_layers(layers: list[Any] | None) -> Image.Image | None:
    """Build an L-mode mask from brush layer(s). Non-zero alpha / luminance = painted."""
    if not layers:
        return None
    out: Image.Image | None = None
    for layer in layers:
        if not isinstance(layer, Image.Image):
            continue
        alpha = _layer_alpha(layer)
        if out is None:
            out = alpha
            continue
        if alpha.size != out.size:
            alpha = alpha.resize(out.size, Image.Resampling.NEAREST)
        out = ImageChops.lighter(out, alpha)
    if out is None:
        return None
    binary = out.point(lambda p: 255 if p > 0 else 0)
    if _nonzero_count(binary) < _MIN_MASK_PIXELS:
        return None
    return binary


def _nonzero_count(mask: Image.Image) -> int:
    hist = mask.histogram()
    return sum(hist[1:]) if hist else 0


def mask_from_composite_diff(
    background: Image.Image,
    composite: Image.Image | None,
    *,
    threshold: int = _DIFF_THRESHOLD,
) -> Image.Image | None:
    """Fallback when brush is baked into composite (no usable layer alpha)."""
    if composite is None or not isinstance(composite, Image.Image):
        return None
    bg = background.convert("RGB") if background.mode != "RGB" else background
    cp = composite.convert("RGB") if composite.mode != "RGB" else composite
    if cp.size != bg.size:
        cp = cp.resize(bg.size, Image.Resampling.NEAREST)
    diff = ImageChops.difference(bg, cp).convert("L")
    binary = diff.point(lambda p: 255 if p >= threshold else 0)
    if _nonzero_count(binary) < _MIN_MASK_PIXELS:
        return None
    return binary


def resolve_mask(editor: EditorValue) -> Image.Image | None:
    """Prefer brush-layer alpha; fall back to composite−background diff."""
    mask = mask_from_layers(editor.get("layers"))
    if mask is not None:
        return mask
    bg = editor.get("background")
    if not isinstance(bg, Image.Image):
        return None
    return mask_from_composite_diff(bg, editor.get("composite"))


def bbox_from_mask(
    mask: Image.Image,
    *,
    pad: int = 8,
) -> tuple[int, int, int, int] | None:
    """Return (left, top, right, bottom) padded bbox of non-zero mask pixels."""
    if mask.mode != "L":
        mask = mask.convert("L")
    box = mask.getbbox()
    if box is None:
        return None
    left, top, right, bottom = box
    w, h = mask.size
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(w, right + pad)
    bottom = min(h, bottom + pad)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _expand_box(
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    *,
    min_side: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    cw, ch = right - left, bottom - top
    if cw >= min_side and ch >= min_side:
        return box
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    w, h = size
    half_w = max(cw, min_side) // 2
    half_h = max(ch, min_side) // 2
    left = max(0, cx - half_w)
    top = max(0, cy - half_h)
    right = min(w, left + max(cw, min_side))
    bottom = min(h, top + max(ch, min_side))
    left = max(0, right - max(cw, min_side))
    top = max(0, bottom - max(ch, min_side))
    return left, top, right, bottom


def crop_masked_region(
    rgb: Image.Image,
    mask: Image.Image,
    *,
    pad: int = 8,
    min_side: int = 64,
) -> Image.Image | None:
    """Crop RGB to the padded mask bbox; expand to at least min_side when possible."""
    if mask.size != rgb.size:
        mask = mask.resize(rgb.size, Image.Resampling.NEAREST)
    box = bbox_from_mask(mask, pad=pad)
    if box is None:
        return None
    left, top, right, bottom = _expand_box(box, rgb.size, min_side=min_side)
    if rgb.mode != "RGB":
        rgb = rgb.convert("RGB")
    return rgb.crop((left, top, right, bottom))


def crop_masked_content(
    rgb: Image.Image,
    mask: Image.Image,
    *,
    pad: int = 8,
    min_side: int = 64,
    outside: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image | None:
    """Black-out pixels outside the mask, then crop to the mask bbox.

    The VL only sees the painted region; surroundings become solid ``outside``.
    """
    if mask.size != rgb.size:
        mask = mask.resize(rgb.size, Image.Resampling.NEAREST)
    box = bbox_from_mask(mask, pad=pad)
    if box is None:
        return None
    left, top, right, bottom = _expand_box(box, rgb.size, min_side=min_side)
    if rgb.mode != "RGB":
        rgb = rgb.convert("RGB")
    if mask.mode != "L":
        mask = mask.convert("L")
    base = Image.new("RGB", rgb.size, outside)
    masked = Image.composite(rgb, base, mask)
    return masked.crop((left, top, right, bottom))


def crop_from_editor(
    editor: EditorValue | None,
    *,
    pad: int = 8,
    min_side: int = 64,
    blackout: bool = True,
    outside: tuple[int, int, int] | None = None,
) -> Image.Image | None:
    """ImageEditor value → RGB crop of the painted region.

    ``blackout=True`` (default, used for preview **and** VL): pixels outside the
    brush become ``outside`` (UI preview: black; VL: mid-gray). A plain bbox
    crop without masking leaks surrounding subjects into the rectangle and the
    model retells the whole photo.

    ``blackout=False``: tight bbox of original pixels only (tests / debug).
    """
    rgb = editor_to_rgb(editor)
    if rgb is None or not isinstance(editor, dict):
        return None
    mask = resolve_mask(editor)
    if mask is None:
        return None
    if blackout:
        fill = VL_OUTSIDE_RGB if outside is None else outside
        return crop_masked_content(
            rgb, mask, pad=pad, min_side=min_side, outside=fill
        )
    return crop_masked_region(rgb, mask, pad=pad, min_side=min_side)
