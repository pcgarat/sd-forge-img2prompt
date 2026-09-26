from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from forge_img2prompt.log import log
from forge_img2prompt.stack import StackInfo

_TAG_SOUP_HINTS = (", masterpiece", "1girl,", "best quality,", "ultra detailed,")


# Valores canónicos del selector de idioma (UI y providers).
LANG_ES = "es"
LANG_EN = "en"
LANG_CHOICES: tuple[tuple[str, str], ...] = (
    ("Español", LANG_ES),
    ("English", LANG_EN),
)
DEFAULT_LANG = LANG_ES

# Rangos de palabras (UI + providers). Bounds = límites absolutos del slider.
GEN_WORDS_BOUNDS: tuple[int, int] = (20, 180)
GEN_WORDS_DEFAULT: tuple[int, int] = (45, 90)
DETAIL_WORDS_BOUNDS: tuple[int, int] = (5, 200)
DETAIL_WORDS_DEFAULT: tuple[int, int] = (8, 25)
# 0 = no descartar por solapamiento; N = descartar si hay ≥ N tokens de contenido compartidos
OVERLAP_DISCARD_BOUNDS: tuple[int, int] = (0, 40)
OVERLAP_DISCARD_DEFAULT: int = 10


def clamp_word_range(
    lo: int | float | None,
    hi: int | float | None,
    *,
    bounds: tuple[int, int],
    default: tuple[int, int],
) -> tuple[int, int]:
    """Normalize (min, max) inside absolute bounds; swap if inverted."""
    floor, ceil = bounds
    d_lo, d_hi = default
    try:
        a = int(round(float(lo))) if lo is not None else d_lo
    except (TypeError, ValueError):
        a = d_lo
    try:
        b = int(round(float(hi))) if hi is not None else d_hi
    except (TypeError, ValueError):
        b = d_hi
    a = max(floor, min(ceil, a))
    b = max(floor, min(ceil, b))
    if a > b:
        a, b = b, a
    return a, b


def count_words(text: str) -> int:
    return len((text or "").split())


def max_tokens_for_words(word_max: int, *, floor: int = 32, ceil: int = 512) -> int:
    """Token budget capped near word_max so the model cannot ramble forever."""
    # ~1.35 tokens/word + small cushion; lower than before so max is harder to blow past
    return max(floor, min(ceil, int(word_max * 1.35) + 12))


def clamp_text_to_words(text: str, word_max: int) -> str:
    """Hard-cap prose to ``word_max`` words, preferring a sentence end when possible."""
    words = (text or "").split()
    if not words:
        return ""
    if word_max <= 0 or len(words) <= word_max:
        return " ".join(words)
    truncated = words[:word_max]
    # Prefer ending on .!? anywhere in the allowed window (keep ≥1 word)
    for i in range(len(truncated) - 1, 0, -1):
        if truncated[i][-1] in ".!?":
            return " ".join(truncated[: i + 1])
    return " ".join(truncated)


def clamp_overlap_discard(value: int | float | None) -> int:
    floor, ceil = OVERLAP_DISCARD_BOUNDS
    try:
        n = int(round(float(value))) if value is not None else OVERLAP_DISCARD_DEFAULT
    except (TypeError, ValueError):
        n = OVERLAP_DISCARD_DEFAULT
    return max(floor, min(ceil, n))


@dataclass(frozen=True)
class PromptRequest:
    image: Image.Image | None
    user_notes: str
    stack: StackInfo
    language: str = DEFAULT_LANG
    word_min: int = GEN_WORDS_DEFAULT[0]
    word_max: int = GEN_WORDS_DEFAULT[1]


@dataclass(frozen=True)
class DetailRequest:
    """Crop of a painted region; base_prompt is scene context only (not mutated)."""

    crop: Image.Image
    base_prompt: str
    stack: StackInfo
    language: str = DEFAULT_LANG
    user_notes: str = ""
    word_min: int = DETAIL_WORDS_DEFAULT[0]
    word_max: int = DETAIL_WORDS_DEFAULT[1]
    overlap_discard: int = OVERLAP_DISCARD_DEFAULT


def append_detail(base: str, detail: str) -> str:
    """Join base prompt and a detail fragment into one prose string."""
    base = " ".join((base or "").split()).strip()
    detail = " ".join((detail or "").split()).strip()
    if not detail:
        return base
    if detail[0].islower():
        detail = detail[0].upper() + detail[1:]
    if detail[-1] not in ".!?":
        detail += "."
    if not base:
        return detail
    if base[-1] not in ".!?":
        base += "."
    return f"{base} {detail}"


def scene_context_snippet(base: str, *, max_chars: int = 280) -> str:
    """Short scene text for zone identity only (not a full prompt to retell)."""
    text = " ".join((base or "").split()).strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Prefer breaking at a sentence end
    for sep in (". ", "! ", "? ", "; "):
        idx = cut.rfind(sep)
        if idx >= max_chars // 3:
            return cut[: idx + 1].strip()
    idx = cut.rfind(" ")
    if idx > 0:
        cut = cut[:idx]
    return cut.rstrip(",;:") + "…"


def normalize_zone_anchor(text: str, *, max_words: int = 12) -> str:
    """Clean identify-step output into a short noun phrase."""
    raw = " ".join((text or "").split()).strip().strip("\"'`")
    if not raw:
        return ""
    # Take first line / clause
    for sep in (".", "\n", ";", ":"):
        if sep in raw:
            raw = raw.split(sep, 1)[0].strip()
            break
    words = raw.split()
    if len(words) > max_words:
        raw = " ".join(words[:max_words])
    return raw.strip(" ,;-")


def detail_looks_like_tag_soup(detail: str) -> bool:
    """True when the fragment is a comma-separated attribute list, not prose."""
    text = " ".join((detail or "").split()).strip()
    if not text:
        return True
    commas = text.count(",")
    if commas >= 4:
        return True
    # Many short comma chunks: "barba, cabello, piel, ojos"
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) >= 4 and sum(1 for p in parts if len(p.split()) <= 3) >= 3:
        return True
    return False


_STOPWORDS = frozenset(
    {
        "a",
        "al",
        "con",
        "de",
        "del",
        "el",
        "en",
        "la",
        "las",
        "los",
        "un",
        "una",
        "unos",
        "unas",
        "y",
        "e",
        "o",
        "u",
        "que",
        "se",
        "su",
        "sus",
        "por",
        "para",
        "como",
        "más",
        "mas",
        "muy",
        "the",
        "and",
        "with",
        "from",
        "for",
        "into",
        "onto",
        "his",
        "her",
        "their",
        "this",
        "that",
        "of",
        "in",
        "on",
        "at",
        "to",
        "an",
    }
)


def _token_set(text: str) -> set[str]:
    return {
        t
        for t in "".join(c.lower() if c.isalnum() else " " for c in text).split()
        if len(t) > 2 and t not in _STOPWORDS
    }


def shared_content_count(base: str, detail: str) -> int:
    """How many content tokens (no stopwords) appear in both texts."""
    return len(_token_set(base) & _token_set(detail))


def detail_is_redundant(
    base: str,
    detail: str,
    *,
    max_shared: int = OVERLAP_DISCARD_DEFAULT,
) -> bool:
    """True when detail shares too many content words with the base prompt.

    ``max_shared <= 0`` disables this check (never discard for overlap).
    Otherwise discard when shared content tokens >= ``max_shared``, or when the
    whole detail is a literal substring of the base.
    """
    if max_shared <= 0:
        return False
    detail_t = _token_set(detail)
    if not detail_t:
        return True
    shared = shared_content_count(base, detail)
    if shared >= max_shared:
        return True
    b = " ".join((base or "").lower().split())
    d = " ".join((detail or "").lower().split())
    if len(d) >= 40 and d in b:
        return True
    return False


_SCENE_LEAK_HINTS = (
    "mujer",
    "hombre y",
    "y una mujer",
    "y un hombre",
    "sentados",
    "sentadas",
    "pareja",
    "mesa",
    "pared",
    "fondo",
    "habitación",
    "habitacion",
    "escenario",
    "woman",
    "man and",
    "and a woman",
    "and a man",
    "sitting",
    "couple",
    "table",
    "background",
    "wooden wall",
    "room",
)


def detail_looks_like_full_scene(detail: str, *, word_max: int | None = None) -> bool:
    """Heuristic: fragment reads like a full-scene caption, not a zone label."""
    text = " ".join((detail or "").lower().split())
    if not text:
        return True
    words = text.split()
    if word_max is not None and len(words) > max(word_max * 2, word_max + 15):
        return True
    hits = sum(1 for h in _SCENE_LEAK_HINTS if h in text)
    if hits >= 2:
        return True
    # Multiple subjects joined with "y una/un" is a strong full-scene signal
    if " y una " in text or " y un " in text or " and a " in text:
        return True
    return False


def normalize_language(language: str | None) -> str:
    raw = (language or "").strip().lower()
    if raw in (LANG_ES, "español", "spanish", "spa"):
        return LANG_ES
    if raw in (LANG_EN, "english", "eng", "en-us", "en-gb"):
        return LANG_EN
    return DEFAULT_LANG


@dataclass(frozen=True)
class PromptResult:
    prompt: str
    negative_hint: str
    sampler_hints: str
    status: str = ""
    fragment: str = ""  # solo el texto de detalle de zona (detail())


class PromptProvider(Protocol):
    def generate(self, request: PromptRequest) -> PromptResult: ...


def _sampler_hints(family: str, variant: str) -> str:
    if family == "krea2":
        if variant == "turbo":
            return "Krea 2 Turbo: ~8 steps, CFG ≈ 0–1, Euler + Simple, clip skip 1."
        if variant == "raw":
            return "Krea 2 RAW: ~28 steps, CFG ≈ 4.5, Euler + Simple, clip skip 1."
        return "Krea 2: Turbo ≈ 8 steps / CFG bajo; RAW ≈ 28 steps / CFG ~4.5 (Euler + Simple)."
    if family == "klein9b":
        if variant == "base":
            return "FLUX.2 Klein 9B Base: ~20–50 steps, CFG ≈ 3.5–5, Euler + Simple."
        return "FLUX.2 Klein 9B distilled: 4 steps, CFG = 1 (no subas steps: overcook)."
    return ""


def _negative_hint(family: str, variant: str) -> str:
    if family == "krea2" and variant == "turbo":
        return "Turbo/distilled: el negativo suele aportar poco; describe exclusiones en el prompt positivo."
    if family == "klein9b" and variant == "distilled":
        return "Klein distilled (CFG=1): el negativo casi no aplica; escribe restricciones en el prompt positivo."
    return ""


def _looks_like_tag_soup(text: str) -> bool:
    low = text.lower()
    if low.count(",") >= 6 and " " in low:
        return True
    return any(h in low for h in _TAG_SOUP_HINTS)


def _normalize_notes(notes: str) -> str:
    return " ".join((notes or "").split()).strip().strip('"')


def _notes_to_prose(notes: str, family: str, language: str = DEFAULT_LANG) -> str:
    """Turn user notes into natural-language prose. Empty notes → empty string."""
    text = _normalize_notes(notes)
    if not text:
        return ""

    lang = normalize_language(language)
    if _looks_like_tag_soup(text):
        core = text.replace(",", ", ")
        if lang == LANG_ES:
            return (
                f"Una escena detallada: {core}. "
                "Describe primero el sujeto, luego entorno, composición, iluminación, materiales "
                "y ambiente en prosa conectada — no una lista de keywords."
            )
        return (
            f"A detailed scene: {core}. "
            "Describe the subject first, then environment, composition, lighting, materials and mood "
            "in connected prose — not a keyword list."
        )

    if text[0].islower():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."

    if lang == LANG_ES:
        if family == "klein9b":
            return (
                f"{text} "
                "Mantén relaciones espaciales explícitas; prioriza sustantivos concretos, "
                "dirección de la luz y materiales. Pon el texto legible de la imagen entre «comillas»."
            )
        return (
            f"{text} "
            "Enfatiza composición, iluminación, materiales y atmósfera en lenguaje natural. "
            "Pon el texto legible de la imagen entre «comillas»."
        )

    if family == "klein9b":
        return (
            f"{text} "
            "Keep spatial relationships explicit; prefer concrete nouns, lighting direction, "
            'and materials. Put any on-image text in "quotes".'
        )
    return (
        f"{text} "
        "Emphasize composition, lighting, materials and atmosphere in natural language. "
        'Put any on-image text in "quotes".'
    )


def _image_fingerprint(image: Image.Image | None) -> str:
    if image is None:
        return "sin imagen"
    try:
        w, h = image.size
        mode = image.mode
        return f"imagen {w}×{h} {mode}"
    except Exception:
        return "imagen (metadatos no legibles)"


class StubProvider:
    """No vision backend: prompt comes only from user notes (+ stack profile)."""

    def generate(self, request: PromptRequest) -> PromptResult:
        stack = request.stack
        log(f"StubProvider · stack={stack.summary}")
        if not stack.is_supported:
            log("abort stub: stack no soportado")
            return PromptResult(
                prompt="",
                negative_hint="",
                sampler_hints="",
                status=(
                    "Stack no reconocido como Krea 2 o FLUX.2 Klein 9B (o TE incompatible). "
                    f"Detectado: {stack.summary}. No se genera prompt «optimizado»."
                ),
            )

        notes = _normalize_notes(request.user_notes)
        label = "Krea 2" if stack.family == "krea2" else "Klein 9B"
        img_info = _image_fingerprint(request.image)

        if not notes:
            log(f"stub sin notas · {img_info}")
            return PromptResult(
                prompt="",
                negative_hint="",
                sampler_hints=_sampler_hints(stack.family, stack.variant),
                status=(
                    f"Stub {label} ({stack.variant}) · {img_info}. "
                    "v1 **no analiza la imagen**: escribe en Notas qué ves / quieres reproducir "
                    "y pulsa Generate de nuevo. El caption automático llegará con el backend."
                ),
            )

        lang = normalize_language(request.language)
        prompt = _notes_to_prose(notes, stack.family, lang)
        lang_label = "español" if lang == LANG_ES else "English"
        status = (
            f"Stub {label} ({stack.variant}) · {img_info} · idioma={lang_label}. "
            f"Prompt derivado de tus notas ({len(notes)} caracteres). "
            "La imagen aún no se captiona."
        )
        log(f"stub OK · {len(prompt)} chars de prompt desde notas")

        return PromptResult(
            prompt=prompt,
            negative_hint=_negative_hint(stack.family, stack.variant),
            sampler_hints=_sampler_hints(stack.family, stack.variant),
            status=status,
        )
