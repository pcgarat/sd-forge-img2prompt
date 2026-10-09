"""Registro persistente de estrategias de redacción del prompt global."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from forge_img2prompt.log import log

STRATEGY_PROSE = "prose"
STRATEGY_INSIDE_OUT = "inside_out"
STRATEGY_LAYERED = "layered"
DEFAULT_STRATEGY = STRATEGY_PROSE

_STORE_NAME = "strategies.json"
_ext_dir: Path | None = None

_COMMON_CAPTION_RULES = (
    "Reply in natural language only — no bullet lists, no booru tags, no preamble. "
    "Never use negative phrasing (forbidden: 'there is no…', 'no hay…', 'without any…', "
    "'absence of…', listing what is missing). Describe only what is present. "
    "Mention readable on-image text in \"quotes\" only if such text is actually visible; "
    "if the main image has no text, do not mention text, captions, signs, or quotes at all. "
    "Do not choose your own length; obey only the word-count range in the user message."
)

_COMMON_NOTES_RULES = (
    "There is no main photograph: the user supplies a brief in notes. "
    "Turn that brief into a ready-to-paste prompt in natural language. "
    "Enrich it with concrete visual details that fit the brief; do not contradict or "
    "replace the user's intent. "
    "Never use negative phrasing (forbidden: 'there is no…', 'no hay…', 'without…'). "
    "Mention text in \"quotes\" only if the brief asks for readable on-image text; "
    "otherwise do not mention text at all. "
    "No bullet lists, no booru tags, no preamble. "
    "Do not choose your own length; obey only the word-count range in the user message."
)

_INSIDE_OUT_STRUCTURE = (
    "Write from the inside out (subject first → surroundings last); early tokens "
    "carry more weight for the generator. Mandatory order in one continuous prompt: "
    "(1) Core — who/what, facial expression or gaze, micro-textures "
    "(pores, wrinkles, seams); "
    "(2) Mid layer — clothing, exact materials, garment colors, held objects/accessories; "
    "(3) Immediate surroundings — what the subject sits/leans on, nearby interacting objects; "
    "(4) Background — landscape/architecture, weather, general atmosphere; "
    "(5) Technical wrap — camera/lens, lighting, color palette, artistic style. "
    "Prefer concrete physical/technical descriptors over vague adjectives. "
    "Forbidden empty praise: 'hyperrealistic', 'beautiful', 'photorealistic', "
    "'high quality', 'masterpiece', and similar fillers. "
    "When describing materials, lighting, or framing, be specific "
    "(e.g. emerald velvet with oxidized brass buttons; golden rim light, soft shadows, "
    "blue hour; medium close-up, 85mm, f/1.8, shallow DOF/bokeh)."
)

# Capas prioritarias (guía plantillas SD: subject → scene → composition → quality).
# Redactado corto y numerado para VLM pequeños (p. ej. Qwen3-VL 4B).
_LAYERED_STRUCTURE = (
    "Write ONE continuous natural-language prompt using FOUR layers in this exact order. "
    "Early words matter more — never put style/quality first. "
    "LAYER 1 SUBJECT (start here): core person/object + action or pose; "
    "add clothing, expression, materials only if visible. "
    "LAYER 2 SCENE: place/space; then lighting, weather, time, atmosphere if visible. "
    "LAYER 3 COMPOSITION: shot type (close-up, medium shot, full body, wide shot), "
    "camera angle, framing. "
    "LAYER 4 STYLE/QUALITY (end only): a few concrete style words "
    "(photography, illustration, anime, soft color palette, studio lighting). "
    "FORBIDDEN at the start or as filler spam: masterpiece, best quality, 8k, "
    "ultra detailed, award winning, cinematic lighting used alone without scene facts. "
    "Do NOT use Stable Diffusion weight syntax like (word:1.5), ((word)), or [a|b] blends — "
    "write plain prose. "
    "Good shape: 'a woman sitting in a cozy library, warm afternoon light, "
    "medium shot from the side, digital illustration'. "
    "Bad shape: 'masterpiece, best quality, 8k, beautiful woman, library, cinematic'."
)

_LAYERED_USER_HINT = (
    "Use four layers in order: "
    "(1) subject + pose first, "
    "(2) scene/lighting, "
    "(3) shot type and angle, "
    "(4) short style words last. "
    "Never lead with masterpiece / best quality / 8k. Plain prose only — no (word:1.5) weights."
)


@dataclass(frozen=True)
class PromptStrategy:
    id: str
    label: str
    caption_system: str
    notes_system: str
    user_hint: str
    builtin: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def configure(ext_dir: str | Path | None) -> None:
    """Fija el directorio de la extensión (donde se guarda strategies.json)."""
    global _ext_dir
    _ext_dir = Path(ext_dir) if ext_dir else None


def strategies_path(ext_dir: str | Path | None = None) -> Path | None:
    base = Path(ext_dir) if ext_dir is not None else _ext_dir
    if base is None:
        return None
    return Path(base) / _STORE_NAME


def factory_defaults() -> dict[str, PromptStrategy]:
    prose = PromptStrategy(
        id=STRATEGY_PROSE,
        label="Prosa",
        caption_system=(
            "You write image prompts for FLUX / Krea 2 / FLUX.2 Klein style generators. "
            "Be concrete, precise, and concise: state only observable facts that add information; "
            "omit filler, mood adjectives, and decorative flourishes that do not change the scene. "
            "Order: subject, action/pose, environment, composition, lighting, materials. "
            f"{_COMMON_CAPTION_RULES}"
        ),
        notes_system=(
            "You write image-generation prompts for FLUX / Krea 2 / FLUX.2 Klein. "
            f"{_COMMON_NOTES_RULES} "
            "Enrich with subject, pose/action, environment, composition, lighting, materials. "
            "Be concrete, precise, and concise; no decorative filler."
        ),
        user_hint="Be concrete, precise, and concise; no ornamental wording.",
        builtin=True,
    )
    inside = PromptStrategy(
        id=STRATEGY_INSIDE_OUT,
        label="Dentro → fuera",
        caption_system=(
            "You write image prompts for FLUX / Krea 2 / FLUX.2 Klein style generators. "
            f"{_INSIDE_OUT_STRUCTURE} "
            "Be concrete, precise, and concise; invent nothing that is not visible. "
            f"{_COMMON_CAPTION_RULES}"
        ),
        notes_system=(
            "You write image-generation prompts for FLUX / Krea 2 / FLUX.2 Klein. "
            f"{_COMMON_NOTES_RULES} "
            f"{_INSIDE_OUT_STRUCTURE} "
            "Be concrete, precise, and concise; no decorative filler."
        ),
        user_hint=(
            "Structure the prompt inside-out in this exact order: "
            "core subject/micro-details → clothing/accessories → immediate surroundings "
            "→ background → technical wrap (camera, lighting, palette, style). "
            "Use concrete materials, lighting and optics; avoid vague quality adjectives."
        ),
        builtin=True,
    )
    layered = PromptStrategy(
        id=STRATEGY_LAYERED,
        label="Capas (sujeto→calidad)",
        caption_system=(
            "You write image prompts for FLUX / Krea 2 / FLUX.2 Klein style generators. "
            f"{_LAYERED_STRUCTURE} "
            "Be concrete, precise, and concise; invent nothing that is not visible. "
            f"{_COMMON_CAPTION_RULES}"
        ),
        notes_system=(
            "You write image-generation prompts for FLUX / Krea 2 / FLUX.2 Klein. "
            f"{_COMMON_NOTES_RULES} "
            f"{_LAYERED_STRUCTURE} "
            "Expand the brief into the four layers; stay faithful to the user's intent. "
            "Be concrete, precise, and concise; no decorative filler."
        ),
        user_hint=_LAYERED_USER_HINT,
        builtin=True,
    )
    return {prose.id: prose, inside.id: inside, layered.id: layered}


STRATEGY_LABELS: dict[str, str] = {
    sid: s.label for sid, s in factory_defaults().items()
}
STRATEGY_CHOICES: tuple[tuple[str, str], ...] = tuple(
    (s.label, s.id) for s in factory_defaults().values()
)


def slugify_id(raw: str | None) -> str:
    text = (raw or "").strip().lower()
    for ch in (" ", "-", "→", "—", "/", "\\", "."):
        text = text.replace(ch, "_")
    text = re.sub(r"[^a-z0-9_]", "", text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _parse_item(raw: dict[str, Any], *, builtin_fallback: bool) -> PromptStrategy | None:
    sid = slugify_id(str(raw.get("id") or ""))
    label = str(raw.get("label") or "").strip()
    caption = str(raw.get("caption_system") or "").strip()
    notes = str(raw.get("notes_system") or "").strip()
    hint = str(raw.get("user_hint") or "").strip()
    if not sid or not label or not caption or not notes:
        return None
    if not hint:
        hint = "Be concrete, precise, and concise; no ornamental wording."
    builtin = bool(raw.get("builtin", builtin_fallback))
    return PromptStrategy(
        id=sid,
        label=label,
        caption_system=caption,
        notes_system=notes,
        user_hint=hint,
        builtin=builtin,
    )


def load_all(ext_dir: str | Path | None = None) -> dict[str, PromptStrategy]:
    out = dict(factory_defaults())
    path = strategies_path(ext_dir)
    if path is None or not path.is_file():
        return out
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log(f"strategies: no se pudo leer `{path}` ({exc}); defaults")
        return out
    items = raw.get("strategies") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        sid = slugify_id(str(item.get("id") or ""))
        if not sid:
            continue
        if sid in out and out[sid].builtin:
            parsed = _parse_item({**item, "id": sid, "builtin": True}, builtin_fallback=True)
            if parsed is not None:
                out[sid] = parsed
            continue
        parsed = _parse_item({**item, "id": sid, "builtin": False}, builtin_fallback=False)
        if parsed is not None:
            out[sid] = parsed
    return out


def save_all(
    strategies: dict[str, PromptStrategy],
    ext_dir: str | Path | None = None,
) -> None:
    path = strategies_path(ext_dir)
    if path is None:
        log("strategies: sin ext_dir; no se guarda")
        return
    payload = {
        "strategies": [s.to_dict() for s in strategies.values()],
    }
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        log(f"strategies: no se pudo guardar `{path}` ({exc})")


def get_strategy(
    strategy_id: str | None,
    ext_dir: str | Path | None = None,
) -> PromptStrategy:
    key = normalize_strategy(strategy_id, ext_dir=ext_dir)
    all_s = load_all(ext_dir)
    return all_s.get(key) or all_s[DEFAULT_STRATEGY]


def strategy_choices(ext_dir: str | Path | None = None) -> list[tuple[str, str]]:
    items = sorted(load_all(ext_dir).values(), key=lambda s: (not s.builtin, s.label.lower()))
    return [(s.label, s.id) for s in items]


def normalize_strategy(
    strategy: str | None,
    *,
    ext_dir: str | Path | None = None,
) -> str:
    raw = (strategy or "").strip()
    if not raw:
        return DEFAULT_STRATEGY
    all_s = load_all(ext_dir)
    if raw in all_s:
        return raw
    low = raw.lower()
    for sid, strat in all_s.items():
        if sid.lower() == low or strat.label.lower() == low:
            return sid
    slug = slugify_id(raw)
    if slug in all_s:
        return slug
    if slug in (
        STRATEGY_INSIDE_OUT,
        "dentro_fuera",
        "dentrofuera",
        "insideout",
        "outward",
    ):
        return STRATEGY_INSIDE_OUT
    if slug in (
        STRATEGY_LAYERED,
        "capas",
        "capas_sujeto_calidad",
        "sujeto_calidad",
        "sd_layers",
        "template_layers",
        "layered_prompt",
    ):
        return STRATEGY_LAYERED
    if slug in (STRATEGY_PROSE, "prosa", "natural"):
        return STRATEGY_PROSE
    return DEFAULT_STRATEGY


def strategy_label(
    strategy: str | None,
    *,
    ext_dir: str | Path | None = None,
) -> str:
    key = normalize_strategy(strategy, ext_dir=ext_dir)
    return get_strategy(key, ext_dir).label


def upsert_strategy(
    *,
    strategy_id: str,
    label: str,
    caption_system: str,
    notes_system: str,
    user_hint: str,
    ext_dir: str | Path | None = None,
    as_new: bool = False,
) -> tuple[PromptStrategy | None, str]:
    sid = slugify_id(strategy_id if not as_new else (strategy_id or label))
    label_clean = (label or "").strip()
    caption = (caption_system or "").strip()
    notes = (notes_system or "").strip()
    hint = (user_hint or "").strip()
    if not sid:
        return None, "ID inválido: usa letras, números o `_`."
    if not label_clean:
        return None, "El nombre (label) es obligatorio."
    if not caption:
        return None, "El system prompt de caption es obligatorio."
    if not notes:
        return None, "El system prompt de notas es obligatorio."
    if not hint:
        hint = "Be concrete, precise, and concise; no ornamental wording."

    all_s = load_all(ext_dir)
    existing = all_s.get(sid)
    if as_new and existing is not None:
        return None, f"Ya existe una estrategia con id `{sid}`."
    if existing is None and not as_new and sid not in factory_defaults():
        # editar id nuevo sin as_new = crear
        pass

    builtin = bool(existing.builtin) if existing is not None else False
    if sid in factory_defaults():
        builtin = True

    strat = PromptStrategy(
        id=sid,
        label=label_clean,
        caption_system=caption,
        notes_system=notes,
        user_hint=hint,
        builtin=builtin,
    )
    all_s[sid] = strat
    save_all(all_s, ext_dir)
    return strat, f"Estrategia `{strat.label}` guardada."


def delete_strategy(
    strategy_id: str,
    ext_dir: str | Path | None = None,
) -> tuple[bool, str]:
    sid = slugify_id(strategy_id)
    all_s = load_all(ext_dir)
    existing = all_s.get(sid)
    if existing is None:
        return False, f"No existe `{sid}`."
    if existing.builtin or sid in factory_defaults():
        return False, "Las estrategias integradas no se pueden eliminar (usa Restaurar)."
    del all_s[sid]
    save_all(all_s, ext_dir)
    return True, f"Estrategia `{existing.label}` eliminada."


def reset_strategy(
    strategy_id: str,
    ext_dir: str | Path | None = None,
) -> tuple[PromptStrategy | None, str]:
    sid = slugify_id(strategy_id)
    defaults = factory_defaults()
    if sid not in defaults:
        return None, "Solo se pueden restaurar estrategias integradas."
    all_s = load_all(ext_dir)
    all_s[sid] = defaults[sid]
    save_all(all_s, ext_dir)
    return defaults[sid], f"Estrategia `{defaults[sid].label}` restaurada a defaults."


def blank_strategy_template() -> PromptStrategy:
    prose = factory_defaults()[STRATEGY_PROSE]
    return PromptStrategy(
        id="",
        label="",
        caption_system=prose.caption_system,
        notes_system=prose.notes_system,
        user_hint=prose.user_hint,
        builtin=False,
    )
