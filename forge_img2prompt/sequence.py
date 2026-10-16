"""Secuencia de n prompts coherentes: biblia de continuidad + deltas por escena.

El problema: n imágenes distintas que deben **parecer la misma secuencia**. La
solución tiene dos fases de LLM y un ensamblado determinista:

1. **Biblia** — una llamada convierte el objetivo de la escena 1 en una ficha de
   continuidad con campos concretos (sujeto, ropa, lugar, luz…). El LLM inventa
   ahí todo el detalle una sola vez.
2. **Deltas** — una llamada por escena extra que describe **solo lo que cambia**
   (acción, encuadre, luz, props nuevos).
3. **Ensamblado** — ``compose_prompt`` pega la biblia **literal** más el delta
   con un orden fijo (dentro → fuera). La biblia se repite palabra por palabra
   en los n prompts, de modo que la consistencia no depende de que el modelo se
   acuerde de parafrasear igual en cada escena.

Ese último punto es la decisión de diseño central: **no** se pide al LLM que
reescriba los detalles continuos en cada prompt (deriva garantizada: «pelo
castaño» acaba siendo «cabello marrón»), se le pide una vez y se reinyecta.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from forge_img2prompt.log import log
from forge_img2prompt.progress import Progress, ProgressLike
from forge_img2prompt.provider import (
    DEFAULT_LANG,
    DEFAULT_STRATEGY,
    LANG_ES,
    STRATEGY_REGISTRY,
    StackInfo,
    _negative_hint,
    _sampler_hints,
    _token_set,
    clamp_word_range,
    count_words,
    family_hint,
    language_instruction,
    max_tokens_for_words,
    normalize_language,
    normalize_strategy,
    shared_content_count,
)

# --- Límites de la UI -------------------------------------------------------

SEQ_MIN_SCENES = 1
SEQ_MAX_SCENES = 12
SEQ_DEFAULT_SCENES = 2

ANCHOR_WORDS_BOUNDS: tuple[int, int] = (20, 500)
ANCHOR_WORDS_DEFAULT: tuple[int, int] = (50, 110)
DELTA_WORDS_BOUNDS: tuple[int, int] = (4, 500)
DELTA_WORDS_DEFAULT: tuple[int, int] = (10, 45)

# Campos de la ficha de continuidad, en el orden en que se ensamblan.
ANCHOR_FIELDS: tuple[str, ...] = (
    "subject",
    "wardrobe",
    "props",
    "setting",
    "lighting",
    "palette",
    "camera",
)
# La acción se cuelga del sujeto; el resto de campos son la capa continua.
COMPOSE_ORDER: tuple[str, ...] = (
    "subject",
    "wardrobe",
    "action",
    "props",
    "setting",
    "lighting",
    "palette",
    "camera",
)
# Campos que una escena extra puede sobrescribir (lo demás es continuo).
DELTA_OVERRIDES: tuple[str, ...] = ("props", "lighting", "palette", "camera")
DELTA_FIELDS: tuple[str, ...] = ("action", *DELTA_OVERRIDES)

ANCHOR_LABELS: dict[str, str] = {
    "subject": "Sujeto",
    "wardrobe": "Ropa / vestuario",
    "props": "Objetos fijos",
    "setting": "Lugar / decorado",
    "lighting": "Luz",
    "palette": "Paleta",
    "camera": "Cámara / estilo",
}
DELTA_LABELS: dict[str, str] = {
    "action": "Acción",
    "props": "Objetos que cambian",
    "lighting": "Luz",
    "palette": "Paleta",
    "camera": "Encuadre / cámara",
}


def clamp_scene_count(value: int | float | None) -> int:
    try:
        n = int(round(float(value))) if value is not None else SEQ_DEFAULT_SCENES
    except (TypeError, ValueError):
        n = SEQ_DEFAULT_SCENES
    return max(SEQ_MIN_SCENES, min(SEQ_MAX_SCENES, n))


def anchor_label(key: str) -> str:
    return ANCHOR_LABELS.get(key, key)


def delta_label(key: str) -> str:
    return DELTA_LABELS.get(key, key)


# --- Alias de claves del LLM ------------------------------------------------
# El modelo devuelve las claves en el idioma del prompt (o mezcladas): se
# resuelven por fragmento para no depender de un JSON exacto.
_FIELD_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("subject", ("subject", "sujeto", "personaje", "character", "protagonis")),
    ("wardrobe", ("wardrobe", "ropa", "vestuario", "clothing", "outfit", "attire")),
    ("action", ("action", "accion", "pose", "gesto", "doing")),
    ("props", ("prop", "objeto", "utensilio", "item", "handheld")),
    ("setting", ("setting", "escenario", "lugar", "entorno", "location",
                 "environment", "habitacion", "room", "decorado")),
    ("lighting", ("light", "luz", "ilumin", "ligh")),
    ("palette", ("palet", "color", "colour", "cromatic", "chroma")),
    ("camera", ("camera", "camara", "optica", "optics", "lens", "lente",
                "encuadre", "framing", "style", "estilo")),
)


def _squash_key(key: str) -> str:
    raw = unicodedata.normalize("NFKD", (key or "").strip().lower())
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    for ch in " -_/.→":
        raw = raw.replace(ch, "")
    return raw


def field_for_key(key: str) -> str | None:
    """Mapa tolerante clave→campo («subject», «sujeto», «anchor_subject»…)."""
    flat = _squash_key(key)
    if not flat:
        return None
    for field, tokens in _FIELD_TOKENS:
        if any(tok in flat for tok in tokens):
            return field
    return None


def _field_map(raw: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in (raw or {}).items():
        field = field_for_key(str(key))
        if field is None or field in out:
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value if v)
        if value is None:
            continue
        text = _clean_phrase(str(value))
        if text:
            out[field] = text
    return out


def _pick(mapping: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """Subconjunto de ``mapping`` restringido a ``fields`` (descarta ruido)."""
    return {k: v for k, v in mapping.items() if k in fields}


# --- DTOs -------------------------------------------------------------------


@dataclass(frozen=True)
class ConsistencyAnchor:
    """Ficha de continuidad: lo que debe ser idéntico en las n imágenes."""

    subject: str = ""
    wardrobe: str = ""
    props: str = ""
    setting: str = ""
    lighting: str = ""
    palette: str = ""
    camera: str = ""

    def field(self, key: str) -> str:
        return str(getattr(self, key, "") or "")

    def with_fields(self, updates: Mapping[str, Any]) -> ConsistencyAnchor:
        clean = {k: v for k, v in updates.items() if k in ANCHOR_FIELDS}
        return replace(self, **clean)

    def is_empty(self) -> bool:
        return not any(self.field(k) for k in ANCHOR_FIELDS)

    def to_dict(self) -> dict[str, str]:
        return {k: self.field(k) for k in ANCHOR_FIELDS}

    def as_prose(self) -> str:
        return _join_segments(self.field(k) for k in ANCHOR_FIELDS)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ConsistencyAnchor:
        return cls(**_pick(_field_map(raw), ANCHOR_FIELDS))

    @classmethod
    def from_fields(cls, *values: str) -> ConsistencyAnchor:
        """Posicional, en orden ``ANCHOR_FIELDS`` (formularios de la UI)."""
        padded = [*(str(v or "") for v in values)][: len(ANCHOR_FIELDS)]
        while len(padded) < len(ANCHOR_FIELDS):
            padded.append("")
        return cls(**dict(zip(ANCHOR_FIELDS, padded)))


@dataclass(frozen=True)
class SceneDelta:
    """Lo que cambia en una escena; los overrides vacíos heredan de la biblia."""

    action: str = ""
    props: str = ""
    lighting: str = ""
    palette: str = ""
    camera: str = ""

    def field(self, key: str) -> str:
        return str(getattr(self, key, "") or "")

    def is_empty(self) -> bool:
        return not any(self.field(k) for k in DELTA_FIELDS)

    def to_dict(self) -> dict[str, str]:
        return {k: self.field(k) for k in DELTA_FIELDS}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SceneDelta:
        return cls(**_pick(_field_map(raw), DELTA_FIELDS))


@dataclass(frozen=True)
class SequenceRequest:
    """Objetivo de la escena 1 + cambios de las escenas 2..n."""

    goal: str
    deltas: tuple[str, ...] = ()
    stack: StackInfo | None = None
    language: str = DEFAULT_LANG
    strategy: str = DEFAULT_STRATEGY
    anchor: ConsistencyAnchor | None = None
    anchor_word_min: int = ANCHOR_WORDS_DEFAULT[0]
    anchor_word_max: int = ANCHOR_WORDS_DEFAULT[1]
    delta_word_min: int = DELTA_WORDS_DEFAULT[0]
    delta_word_max: int = DELTA_WORDS_DEFAULT[1]

    @property
    def scene_count(self) -> int:
        return 1 + len(self.deltas)


@dataclass(frozen=True)
class SequenceResult:
    prompts: tuple[str, ...]
    anchor: ConsistencyAnchor
    deltas: tuple[SceneDelta, ...]
    negative_hint: str = ""
    sampler_hints: str = ""
    status: str = ""

    def prompt_for(self, index: int) -> str:
        """Texto de la escena ``index`` (1-based); vacío si no existe."""
        try:
            i = int(index)
        except (TypeError, ValueError):
            return ""
        if 1 <= i <= len(self.prompts):
            return self.prompts[i - 1]
        return ""


# --- Limpieza y ensamblado --------------------------------------------------

_PUNCT_EDGE = " \t\n\r,;.:·-–—\"'«»"


def _clean_phrase(text: str) -> str:
    return " ".join((text or "").split()).strip().strip(_PUNCT_EDGE)


_MIN_DEDUPE_CHARS = 8


def _dedupe_against(segments: list[str]) -> None:
    """Vacía un segmento que ya está contenido en otro (biblia ↔ delta).

    El solape va en las dos direcciones: el modelo re-describe el sujeto en la
    acción, y el delta repite un objeto que ya está en el decorado. Se aplica
    solo al segmento más corto de cada pareja para no amputar un texto rico (la
    acción) cuando la biblia es un sustantivo suelto.
    """
    lowered = [seg.casefold() for seg in segments]
    for i, low in enumerate(lowered):
        if not low or len(segments[i]) < _MIN_DEDUPE_CHARS:
            continue
        for j in range(len(segments)):
            if i == j or not lowered[j]:
                continue
            other = lowered[j]
            if low not in other and other not in low:
                continue
            # El más corto cede; a igualdad, cede el posterior (el delta).
            mine, theirs = len(segments[i]), len(segments[j])
            if mine < theirs:
                loser = i
            elif theirs < mine:
                loser = j
            else:
                loser = max(i, j)
            segments[loser] = ""
            lowered[loser] = ""
            if loser == i:
                break


def _join_segments(segments: Any) -> str:
    parts = [_clean_phrase(str(seg or "")) for seg in segments]
    parts = [p for p in parts if p]
    if not parts:
        return ""
    _dedupe_against(parts)
    kept: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part:
            continue
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        kept.append(part)
    if not kept:
        return ""
    text = ", ".join(kept)
    return text[0].upper() + text[1:] + "."


def compose_prompt(
    anchor: ConsistencyAnchor,
    delta: SceneDelta,
    *,
    fallback_action: str = "",
) -> str:
    """Prompt autocontenido: biblia literal + delta, en orden dentro → fuera."""
    segments: list[str] = []
    for key in COMPOSE_ORDER:
        if key == "action":
            text = delta.field("action") or fallback_action
        elif key in DELTA_OVERRIDES:
            text = delta.field(key) or anchor.field(key)
        else:
            text = anchor.field(key)
        segments.append(text)
    return _join_segments(segments)


def compose_sequence(
    anchor: ConsistencyAnchor,
    deltas: tuple[SceneDelta, ...] | list[SceneDelta],
    *,
    fallback_action: str = "",
) -> tuple[str, ...]:
    return tuple(
        compose_prompt(anchor, delta, fallback_action=fallback_action if i == 0 else "")
        for i, delta in enumerate(deltas)
    )


def degraded_prompts(goal: str, deltas: tuple[SceneDelta, ...] | list[SceneDelta]) -> tuple[str, ...]:
    """Sin biblia: «objetivo + cambio». Suelo honesto cuando el backend cae.

    La continuidad de sujeto, lugar y luz se mantiene porque el objetivo se
    repite entero; la pega es que la acción de la escena 1 viaja en todos los
    prompts, así que conviene editar la biblia a mano y pulsar «Recomponer».

    Si el delta de la escena 1 ya es el objetivo, ``_join_segments`` lo dedupe:
    la escena 1 queda con el objetivo solo.
    """
    out: list[str] = []
    for delta in deltas:
        out.append(_join_segments([goal, delta.field("action"), *(
            delta.field(key) for key in DELTA_OVERRIDES if delta.field(key)
        )]))
    return tuple(out)


# --- Parseo de la respuesta del LLM -----------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_KV_LINE_RE = re.compile(r"^\s*[-*•]?\s*([^:=\-]{2,40})\s*[:=]\s*(.+?)\s*$")


def _strip_trailing_commas(text: str) -> str:
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def _loads_object(text: str) -> dict[str, Any] | None:
    for candidate in (text, _strip_trailing_commas(text)):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Primer objeto JSON del texto, tolerando fences, prosa alrededor y comas finales."""
    raw = (text or "").strip()
    if not raw:
        return None
    candidates: list[str] = []
    fence = _FENCE_RE.search(raw)
    if fence:
        candidates.append(fence.group(1))
    candidates.append(raw)
    for candidate in candidates:
        data = _loads_object(candidate)
        if data is not None:
            return data
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        return _loads_object(raw[start : end + 1])
    return None


def _parse_key_value_lines(text: str) -> dict[str, str]:
    """Segunda oportunidad: la respuesta llegó como lista «campo: valor»."""
    out: dict[str, str] = {}
    for line in (text or "").splitlines():
        match = _KV_LINE_RE.match(line)
        if not match:
            continue
        field = field_for_key(match.group(1))
        if field is None or field in out:
            continue
        value = _clean_phrase(match.group(2))
        if value:
            out[field] = value
    return out


def _unwrap(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Baja un nivel si el modelo envolvió el objeto («anchor», «change»…)."""
    if any(field_for_key(k) for k in data):
        return data
    nested = [v for v in data.values() if isinstance(v, dict)]
    if len(nested) == 1:
        return nested[0]
    return data


def parse_anchor(raw: str) -> ConsistencyAnchor:
    data = extract_json_object(raw)
    if data is not None:
        return ConsistencyAnchor.from_mapping(_unwrap(data))
    return ConsistencyAnchor.from_mapping(_parse_key_value_lines(raw))


def parse_delta(raw: str) -> SceneDelta:
    data = extract_json_object(raw)
    if data is not None:
        return SceneDelta.from_mapping(_unwrap(data))
    return SceneDelta.from_mapping(_parse_key_value_lines(raw))


def parse_bible(raw: str) -> tuple[ConsistencyAnchor, SceneDelta]:
    """Respuesta de la fase 1 → (biblia, delta de la escena 1)."""
    data = extract_json_object(raw)
    if data is None:
        return parse_anchor(raw), SceneDelta()
    anchor_raw: Mapping[str, Any] = data
    scene_raw: Mapping[str, Any] = {}
    for key, value in data.items():
        if not isinstance(value, Mapping):
            continue
        flat = _squash_key(key)
        if any(tok in flat for tok in ("scene1", "escena1", "first", "primera", "action", "accion")):
            scene_raw = value
        elif any(tok in flat for tok in ("anchor", "biblia", "continuity", "continuidad", "sheet")):
            anchor_raw = value
    anchor = ConsistencyAnchor.from_mapping(_unwrap(anchor_raw))
    if anchor.is_empty():
        anchor = ConsistencyAnchor.from_mapping(data)
    scene = SceneDelta.from_mapping(scene_raw) if scene_raw else SceneDelta()
    if not scene.action:
        for key in ("action", "accion"):
            for candidate, value in data.items():
                if _squash_key(candidate) == key and isinstance(value, str):
                    scene = replace(scene, action=_clean_phrase(value))
    return anchor, scene


# --- Prompts de las dos fases ----------------------------------------------

_ANCHOR_RULES = (
    "You build the continuity sheet for a SEQUENCE of images that must look like "
    "the same scene shot at different moments. "
    "Reply with ONE JSON object and nothing else, no prose, no fences. "
    'Shape: {"anchor": {"subject": "", "wardrobe": "", "props": "", "setting": "", '
    '"lighting": "", "palette": "", "camera": ""}, '
    '"scene_1": {"action": "", "props": "", "lighting": "", "palette": "", "camera": ""}}. '
    "anchor = what must stay IDENTICAL in every image. Invent concrete detail that "
    "fits the goal and never contradicts it. "
    "Every value is a short phrase of observable specifics: colours, materials, "
    "textures, finishes, quantities, geometry, direction of light, focal length. "
    "Forbidden vague words: beautiful, realistic, cinematic, high quality, detailed, "
    "stunning, epic. Name ONE single referent for the subject and reuse that exact noun. "
    "Never describe what is absent (no 'there is no', 'no hay', 'without'). "
    "Do not invent extra people, animals or objects that the goal does not imply; "
    "leave a field as an empty string if there is no basis for it. "
    "scene_1.action = what the subject is doing in the FIRST image, as a short clause. "
    "scene_1 overrides (props/lighting/palette/camera) only when the first image "
    "differs from the anchor defaults, else empty string. "
    "Mention readable on-image text in quotes only if the goal asks for it."
)

_DELTA_RULES = (
    "You describe ONLY what changes between two consecutive images of a sequence. "
    "Reply with ONE JSON object and nothing else, no prose, no fences. "
    'Shape: {"action": "", "props": "", "lighting": "", "palette": "", "camera": ""}. '
    "action (required) = what the subject does now, as a short clause in the present. "
    "The other keys are OPTIONAL overrides: fill them ONLY if that aspect actually "
    "changes in this scene (time of day, new object, new framing), otherwise leave "
    'them as an empty string. Anything you leave empty keeps the continuity sheet. '
    "NEVER re-describe the continuity sheet (subject, wardrobe, setting, palette, "
    "style): it is already repeated word for word in every prompt. "
    "If you must name the subject, use the SAME noun as the continuity sheet. "
    "Concrete and physical; forbidden vague words: beautiful, realistic, cinematic, "
    "high quality, detailed. Never describe what is absent."
)


def anchor_system(language: str = DEFAULT_LANG) -> str:
    return f"{_ANCHOR_RULES} {language_instruction(language)}"


def delta_system(language: str = DEFAULT_LANG) -> str:
    return f"{_DELTA_RULES} {language_instruction(language)}"


def anchor_user_text(
    goal: str,
    *,
    language: str = DEFAULT_LANG,
    strategy: str = DEFAULT_STRATEGY,
    family: str = "krea2",
    word_min: int = ANCHOR_WORDS_DEFAULT[0],
    word_max: int = ANCHOR_WORDS_DEFAULT[1],
) -> str:
    parts = [
        "User goal for the FIRST image of the sequence:",
        _clean_phrase(goal) or "(empty)",
        _strategy_hint(strategy),
        family_hint(family),
        "Invent all the concrete detail the user did not give: the images must look "
        "like the same place, the same light and the same person throughout.",
        "Keep the whole anchor (all fields together) between "
        f"{word_min} and {word_max} words.",
        "Json only.",
    ]
    return " ".join(p for p in parts if p)


def delta_user_text(
    anchor: ConsistencyAnchor,
    *,
    goal: str,
    change: str,
    index: int,
    total: int,
    language: str = DEFAULT_LANG,
    word_min: int = DELTA_WORDS_DEFAULT[0],
    word_max: int = DELTA_WORDS_DEFAULT[1],
) -> str:
    parts = [
        f"Scene {index} of {total}.",
        "Continuity sheet already added to every prompt (do NOT rewrite it): "
        + json.dumps(anchor.to_dict(), ensure_ascii=False),
        f"Sequence goal (scene 1): {_clean_phrase(goal) or '(empty)'}",
        "What changes in this scene, as the user wrote it: "
        f"{_clean_phrase(change) or '(empty — keep everything identical, change only the pose slightly)'}",
        f"Write the action in {word_min}–{word_max} words. Assume everything you "
        "are not told to change stays exactly the same.",
        "Json only.",
    ]
    return " ".join(p for p in parts if p)


def _strategy_hint(strategy: str | None) -> str:
    spec = STRATEGY_REGISTRY.get(strategy)
    return spec.user_hint or ""


# --- Motor ------------------------------------------------------------------

RunFn = Callable[[str, str, int], str]


def _bible_tokens(word_max: int) -> int:
    # Margen extra sobre el presupuesto de prosa: claves JSON + scene_1.
    return max_tokens_for_words(word_max, floor=240, ceil=900) + 160


def _delta_tokens(word_max: int) -> int:
    return max_tokens_for_words(word_max, floor=120, ceil=600) + 80


def handle(
    request: SequenceRequest,
    run: RunFn,
    *,
    progress: ProgressLike = None,
    service: str = "",
) -> SequenceResult:
    """Ejecuta las dos fases sobre un ``run(system, user, max_tokens)`` cualquiera.

    ``run`` es el único punto de contacto con el backend, así que la lógica es
    agnóstica (Ollama, NaN, transformers o un doble de test). Un fallo del
    backend no aborta la secuencia: se degrada a «escena 1 + cambio» y se avisa.
    """
    sink = Progress.wrap(progress)
    lang = normalize_language(request.language)
    strategy = normalize_strategy(request.strategy)
    goal = _clean_phrase(request.goal)
    family = request.stack.family if request.stack else "krea2"
    supported = bool(request.stack and request.stack.is_supported)

    a_lo, a_hi = clamp_word_range(
        request.anchor_word_min,
        request.anchor_word_max,
        bounds=ANCHOR_WORDS_BOUNDS,
        default=ANCHOR_WORDS_DEFAULT,
    )
    d_lo, d_hi = clamp_word_range(
        request.delta_word_min,
        request.delta_word_max,
        bounds=DELTA_WORDS_BOUNDS,
        default=DELTA_WORDS_DEFAULT,
    )

    anchor = request.anchor or ConsistencyAnchor()
    needs_bible = request.anchor is None
    scene1 = SceneDelta()
    notes: list[str] = []
    backend_down = False
    total_steps = (1 if needs_bible else 0) + len(request.deltas)
    done = 0

    def report(desc: str) -> None:
        sink.notify(done / max(total_steps, 1), desc)

    def invoke(system: str, user: str, tokens: int) -> str | None:
        if backend_down:
            return None
        try:
            return run(system, user, tokens) or ""
        except Exception as exc:  # noqa: BLE001
            log(f"secuencia: fallo del backend ({type(exc).__name__}: {exc})")
            return None

    if needs_bible:
        report("Biblia de continuidad…")
        raw = invoke(
            anchor_system(lang),
            anchor_user_text(
                goal,
                language=lang,
                strategy=strategy,
                family=family,
                word_min=a_lo,
                word_max=a_hi,
            ),
            _bible_tokens(a_hi),
        )
        done += 1
        if raw is None:
            backend_down = True
        else:
            parsed, scene1 = parse_bible(raw)
            if not parsed.is_empty():
                anchor = parsed
            else:
                notes.append("La biblia llegó vacía: prompts degradados a «objetivo + cambio».")

    deltas: list[SceneDelta] = []
    for i, change in enumerate(request.deltas, start=2):
        report(f"Escena {i}…")
        delta = SceneDelta()
        if not backend_down:
            raw = invoke(
                delta_system(lang),
                delta_user_text(
                    anchor,
                    goal=goal,
                    change=change,
                    index=i,
                    total=request.scene_count,
                    language=lang,
                    word_min=d_lo,
                    word_max=d_hi,
                ),
                _delta_tokens(d_hi),
            )
            done += 1
            if raw is None:
                backend_down = True
            else:
                delta = parse_delta(raw)
        if not delta.action:
            delta = replace(delta, action=_clean_phrase(change))
        deltas.append(delta)

    if anchor.is_empty():
        # Sin biblia no hay campos continuos: la escena 1 es el objetivo y los
        # deltas del bucle (escenas 2..n) se mantienen tal cual.
        deltas = [SceneDelta(action=goal), *deltas]
        prompts = degraded_prompts(goal, deltas)
        notes.append(
            f"Sin biblia ({service or 'backend'}): cada prompt repite el objetivo "
            "entero y puede mezclar acciones. Edita la biblia a mano y pulsa "
            "«Recomponer» para arreglarlo sin volver a llamar al modelo."
        )
    else:
        first = scene1 if scene1.action else SceneDelta(action=goal)
        deltas = [first, *deltas]
        prompts = compose_sequence(anchor, deltas)

    notes.extend(
        continuity_notes(
            anchor,
            tuple(deltas),
            prompts,
            anchor_word_max=a_hi,
        )
    )

    hints = _sampler_hints(family, request.stack.variant if request.stack else "") if supported else ""
    negative = _negative_hint(family, request.stack.variant if request.stack else "") if supported else ""
    status_parts = [
        f"{service or 'Secuencia'} · {len(prompts)} prompts",
        f"biblia {'lista' if not anchor.is_empty() else 'no disponible'}",
        f"idioma={'español' if lang == LANG_ES else 'English'}",
        f"estrategia={strategy}",
    ]
    return SequenceResult(
        prompts=prompts,
        anchor=anchor,
        deltas=tuple(deltas),
        negative_hint=negative,
        sampler_hints=hints,
        status=" · ".join(status_parts) + ("\n" + "\n".join(f"· {n}" for n in notes) if notes else ""),
    )


def continuity_notes(
    anchor: ConsistencyAnchor,
    deltas: tuple[SceneDelta, ...],
    prompts: tuple[str, ...],
    *,
    anchor_word_max: int = ANCHOR_WORDS_DEFAULT[1],
) -> list[str]:
    """Avisos accionables de coherencia; no corrige, informa.

    Dos señales distintas que es fácil confundir:

    - **Cambio corto que cita la biblia** (el usuario escribió «abre la
      persiana»): el delta solo reescribe el canvas, que es exactamente lo que
      queremos. No se avisa.
    - **Cambio largo que re-describe la escena** (el modelo se fue de la mano):
      eso sí arriesga deriva y se avisa por palabra neta, no por solape.
    """
    notes: list[str] = []
    anchor_text = anchor.as_prose()
    anchor_words = count_words(anchor_text)
    if anchor_text and anchor_words > anchor_word_max:
        notes.append(
            f"La biblia ocupa {anchor_words} palabras (máx {anchor_word_max}); "
            "recorta campos para que quepa el cambio de cada escena."
        )
    for i, delta in enumerate(deltas, start=1):
        if not delta.action:
            notes.append(f"Escena {i} sin acción: quedará casi idéntica a la anterior.")
            continue
        # La señal es cuánta biblia reescribe el cambio, no cuánto ocupa: un
        # cambio corto que cita «cama» o «moqueta» es precisamente lo esperado.
        shared = shared_content_count(anchor_text, delta.action)
        if anchor_text and shared >= max(anchor_words // 2, 6):
            notes.append(
                f"Escena {i}: el cambio ({count_words(delta.action)} palabras) reescribe "
                f"la biblia ({shared} palabras coincidentes); describe solo lo que cambia."
            )
    for i in range(1, len(prompts)):
        prev, cur = prompts[i - 1], prompts[i]
        if not cur:
            notes.append(f"Escena {i + 1} vacía.")
            continue
        if score_duplicate(prev, cur) >= 0.9:
            notes.append(f"Escenas {i} y {i + 1} casi idénticas: revisa el cambio de la {i + 1}.")
    if len(deltas) <= 1:
        notes.append("Solo hay una escena; añade cambios con «+» para obtener una secuencia.")
    return notes


def score_duplicate(a: str, b: str) -> float:
    """Solape Jaccard de palabras de contenido, para detectar escenas repetidas."""
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# --- Estado UI ↔ recomposición sin LLM --------------------------------------
# El script guarda esto en un ``gr.State``: permite recomponer y mover escenas
# sin volver a llamar al modelo, que es donde está el coste.


def state_payload(result: SequenceResult, scene_sources: tuple[str, ...]) -> dict[str, Any]:
    """Estado persistible: biblia + deltas estructurados con su texto de origen.

    ``scene_sources`` es lo que el usuario escribió en cada caja (escena 1 =
    objetivo, extras = cambio). Guardarlo permite detectar ediciones: si la caja
    cambió, el delta estructurado ya no es válido y se re-deriva del texto.
    """
    scenes: list[dict[str, Any]] = []
    for i, delta in enumerate(result.deltas):
        source = scene_sources[i] if i < len(scene_sources) else ""
        scenes.append({"source": source, "delta": delta.to_dict()})
    return {"anchor": result.anchor.to_dict(), "scenes": scenes}


def _state_scenes(state: Mapping[str, Any] | None) -> list[Any]:
    raw = (state or {}).get("scenes")
    return list(raw) if isinstance(raw, list) else []


def resolve_deltas(
    state: Mapping[str, Any] | None,
    goal: str,
    change_texts: tuple[str, ...] | list[str],
) -> tuple[SceneDelta, ...]:
    """Deltas por escena: ``[objetivo, *cambios]`` contra el estado guardado.

    ``change_texts`` son **solo** las cajas de escena extra (la escena 1 es
    ``goal``). Al editar a mano una caja se pierden sus overrides estructurados
    (props/luz/cámara) porque ya no hay forma de saber a qué campo pertenecía el
    texto nuevo; se trata como acción. Es el comportamiento predecible: lo que
    escribes es lo que sale.
    """
    stored = _state_scenes(state)
    texts = [goal, *[str(t or "") for t in change_texts]]
    out: list[SceneDelta] = []
    for i, text in enumerate(texts):
        cleaned = _clean_phrase(text)
        entry = stored[i] if i < len(stored) else None
        delta_raw = entry.get("delta") if isinstance(entry, Mapping) else None
        source = entry.get("source") if isinstance(entry, Mapping) else None
        if (
            isinstance(delta_raw, Mapping)
            and isinstance(source, str)
            and source == text
        ):
            out.append(SceneDelta.from_mapping(delta_raw))
            continue
        out.append(SceneDelta(action=cleaned))
    return tuple(out)


def recompose(
    anchor: ConsistencyAnchor,
    goal: str,
    change_texts: tuple[str, ...] | list[str],
    *,
    state: Mapping[str, Any] | None = None,
    stack: StackInfo | None = None,
) -> SequenceResult:
    """Reensambla los n prompts con la biblia editada a mano (sin backend)."""
    deltas = resolve_deltas(state, goal, change_texts)
    supported = bool(stack and stack.is_supported)
    variant = stack.variant if stack else ""
    family = stack.family if stack else "krea2"
    prompts = compose_sequence(anchor, deltas)
    notes = continuity_notes(
        anchor,
        deltas,
        prompts,
    )
    return SequenceResult(
        prompts=prompts,
        anchor=anchor,
        deltas=deltas,
        negative_hint=_negative_hint(family, variant) if supported else "",
        sampler_hints=_sampler_hints(family, variant) if supported else "",
        status=(
            f"Recompuesto · {len(prompts)} prompts · biblia manual"
            + ("\n" + "\n".join(f"· {n}" for n in notes) if notes else "")
        ),
    )

