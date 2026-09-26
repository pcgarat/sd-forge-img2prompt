"""Persist Image → Prompt UI prefs across Forge restarts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forge_img2prompt.log import log
from forge_img2prompt.provider import (
    DEFAULT_LANG,
    DETAIL_WORDS_BOUNDS,
    DETAIL_WORDS_DEFAULT,
    GEN_WORDS_BOUNDS,
    GEN_WORDS_DEFAULT,
    OVERLAP_DISCARD_BOUNDS,
    OVERLAP_DISCARD_DEFAULT,
    clamp_overlap_discard,
    clamp_word_range,
)

_PREFS_NAME = "ui_prefs.json"

_DEFAULTS: dict[str, Any] = {
    "gen_wmin": GEN_WORDS_DEFAULT[0],
    "gen_wmax": GEN_WORDS_DEFAULT[1],
    "det_wmin": DETAIL_WORDS_DEFAULT[0],
    "det_wmax": DETAIL_WORDS_DEFAULT[1],
    "det_overlap": OVERLAP_DISCARD_DEFAULT,
    "language": DEFAULT_LANG,
    "vl_value": "",
}


def prefs_path(ext_dir: str | Path) -> Path:
    return Path(ext_dir) / _PREFS_NAME


def load_prefs(ext_dir: str | Path) -> dict[str, Any]:
    path = prefs_path(ext_dir)
    data = dict(_DEFAULTS)
    if not path.is_file():
        return data
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data.update({k: raw[k] for k in _DEFAULTS if k in raw})
    except Exception as exc:  # noqa: BLE001
        log(f"prefs: no se pudo leer `{path}` ({exc}); defaults")
        return dict(_DEFAULTS)

    g_lo, g_hi = clamp_word_range(
        data.get("gen_wmin"),
        data.get("gen_wmax"),
        bounds=GEN_WORDS_BOUNDS,
        default=GEN_WORDS_DEFAULT,
    )
    d_lo, d_hi = clamp_word_range(
        data.get("det_wmin"),
        data.get("det_wmax"),
        bounds=DETAIL_WORDS_BOUNDS,
        default=DETAIL_WORDS_DEFAULT,
    )
    data["gen_wmin"], data["gen_wmax"] = g_lo, g_hi
    data["det_wmin"], data["det_wmax"] = d_lo, d_hi
    data["det_overlap"] = clamp_overlap_discard(data.get("det_overlap"))
    lang = str(data.get("language") or DEFAULT_LANG).strip().lower()
    data["language"] = lang if lang in ("es", "en") else DEFAULT_LANG
    data["vl_value"] = str(data.get("vl_value") or "")
    return data


def save_prefs(ext_dir: str | Path, updates: dict[str, Any]) -> dict[str, Any]:
    current = load_prefs(ext_dir)
    current.update({k: updates[k] for k in _DEFAULTS if k in updates})
    g_lo, g_hi = clamp_word_range(
        current.get("gen_wmin"),
        current.get("gen_wmax"),
        bounds=GEN_WORDS_BOUNDS,
        default=GEN_WORDS_DEFAULT,
    )
    d_lo, d_hi = clamp_word_range(
        current.get("det_wmin"),
        current.get("det_wmax"),
        bounds=DETAIL_WORDS_BOUNDS,
        default=DETAIL_WORDS_DEFAULT,
    )
    current["gen_wmin"], current["gen_wmax"] = g_lo, g_hi
    current["det_wmin"], current["det_wmax"] = d_lo, d_hi
    current["det_overlap"] = clamp_overlap_discard(current.get("det_overlap"))
    path = prefs_path(ext_dir)
    try:
        path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        log(f"prefs: no se pudo guardar `{path}` ({exc})")
    return current


def format_range(lo: int, hi: int) -> str:
    return f"{int(lo)},{int(hi)}"


def parse_range_text(
    raw: str | None,
    *,
    bounds: tuple[int, int],
    default: tuple[int, int],
) -> tuple[int, int]:
    text = (raw or "").strip().replace("–", "-").replace("—", "-")
    if not text:
        return clamp_word_range(None, None, bounds=bounds, default=default)
    sep = "," if "," in text else ("-" if "-" in text else None)
    if sep is None:
        try:
            n = int(round(float(text)))
            return clamp_word_range(n, n, bounds=bounds, default=default)
        except (TypeError, ValueError):
            return clamp_word_range(None, None, bounds=bounds, default=default)
    left, right = text.split(sep, 1)
    return clamp_word_range(left.strip(), right.strip(), bounds=bounds, default=default)


def dual_range_html(
    *,
    widget_id: str,
    bridge_id: str,
    label: str,
    lo: int,
    hi: int,
    minimum: int,
    maximum: int,
    step: int,
) -> str:
    """Markup for a dual-thumb range; JS syncs to a hidden Gradio textbox."""
    return (
        f'<div class="img2prompt-dual" id="{widget_id}" '
        f'data-bridge="{bridge_id}" data-min="{minimum}" data-max="{maximum}" '
        f'data-step="{step}" data-lo="{lo}" data-hi="{hi}">'
        f'<div class="img2prompt-dual-head">'
        f'<span class="img2prompt-dual-title">{label}</span>'
        f'<span class="img2prompt-dual-values">'
        f'<strong class="lo-val">{lo}</strong>–<strong class="hi-val">{hi}</strong>'
        f"</span></div>"
        f'<div class="img2prompt-dual-sliders">'
        f'<input type="range" class="img2prompt-dual-lo" min="{minimum}" '
        f'max="{maximum}" step="{step}" value="{lo}" aria-label="{label} mínimo">'
        f'<input type="range" class="img2prompt-dual-hi" min="{minimum}" '
        f'max="{maximum}" step="{step}" value="{hi}" aria-label="{label} máximo">'
        f"</div></div>"
    )
