"""Registro de estrategias de redacción cargado desde ``strategies.json``.

El JSON (raíz del repo de extensión) define cada estrategia con ``id``,
``label``, ``caption_system``, ``notes_system``, ``user_hint`` y ``builtin``.
Aquí se valida y se expone un fallback embebido para que la extensión siga
funcionando si el fichero falta o está corrupto.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from forge_img2prompt.log import log

STRATEGY_PROSE = "prose"
STRATEGY_INSIDE_OUT = "inside_out"
DEFAULT_STRATEGY = STRATEGY_PROSE


@dataclass(frozen=True)
class StrategySpec:
    id: str
    label: str
    caption_system: str
    notes_system: str
    user_hint: str
    builtin: bool = True


_BUILTIN_STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec(
        id=STRATEGY_PROSE,
        label="Prosa",
        caption_system=(
            "You write image prompts for FLUX / Krea 2 / FLUX.2 Klein style generators. "
            "Be concrete, precise, and concise; state only observable facts and omit filler."
        ),
        notes_system=(
            "Turn the user's brief into a ready-to-paste image-generation prompt in "
            "natural language; enrich with concrete visual detail without contradicting it."
        ),
        user_hint="Be concrete, precise, and concise; no ornamental wording.",
    ),
    StrategySpec(
        id=STRATEGY_INSIDE_OUT,
        label="Dentro → fuera",
        caption_system=(
            "You write image prompts for FLUX / Krea 2 / FLUX.2 Klein style generators. "
            "Write from the inside out: core subject first, surroundings and technical "
            "wrap last; prefer concrete physical descriptors over vague adjectives."
        ),
        notes_system=(
            "Turn the user's brief into a ready-to-paste image-generation prompt written "
            "from the inside out: subject first, background and technical wrap last."
        ),
        user_hint=(
            "Structure the prompt inside-out: core subject → clothing/accessories → "
            "surroundings → background → technical wrap (camera, lighting, style)."
        ),
    ),
)


def _strategies_path() -> Path:
    # forge_img2prompt/strategy_registry.py → repo root
    return Path(__file__).resolve().parent.parent / "strategies.json"


def _parse(raw: object) -> tuple[StrategySpec, ...]:
    if not isinstance(raw, dict):
        return ()
    items = raw.get("strategies")
    if not isinstance(items, list):
        return ()
    specs: list[StrategySpec] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or "").strip()
        caption_system = str(item.get("caption_system") or "").strip()
        notes_system = str(item.get("notes_system") or "").strip()
        user_hint = str(item.get("user_hint") or "").strip()
        if not sid or sid in seen or not caption_system or not notes_system:
            continue
        seen.add(sid)
        specs.append(
            StrategySpec(
                id=sid,
                label=str(item.get("label") or sid).strip() or sid,
                caption_system=caption_system,
                notes_system=notes_system,
                user_hint=user_hint,
                builtin=bool(item.get("builtin", True)),
            )
        )
    return tuple(specs)


def load_strategies() -> tuple[StrategySpec, ...]:
    """Estrategias del JSON; fallback embebido si falta o no es válido."""
    path = _strategies_path()
    try:
        specs = _parse(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        log(f"strategies.json no disponible ({exc}); usando estrategias embebidas")
        return _BUILTIN_STRATEGIES
    if not specs:
        log("strategies.json sin estrategias válidas; usando embebidas")
        return _BUILTIN_STRATEGIES
    return specs


_ALIASES: dict[str, str] = {
    "prose": STRATEGY_PROSE,
    "prosa": STRATEGY_PROSE,
    "natural": STRATEGY_PROSE,
    "inside_out": STRATEGY_INSIDE_OUT,
    "insideout": STRATEGY_INSIDE_OUT,
    "dentro_fuera": STRATEGY_INSIDE_OUT,
    "dentrofuera": STRATEGY_INSIDE_OUT,
    "outward": STRATEGY_INSIDE_OUT,
}


def _norm_key(strategy: str | None) -> str:
    raw = (strategy or "").strip().lower()
    for ch in (" ", "-", "→", "—", "/", "\\"):
        raw = raw.replace(ch, "_")
    while "__" in raw:
        raw = raw.replace("__", "_")
    return raw.strip("_")


class StrategyRegistry:
    """Lookup/canonicalización de estrategias de forma table-driven."""

    def __init__(self, specs: tuple[StrategySpec, ...]) -> None:
        self._specs = specs
        self._by_id = {s.id: s for s in specs}
        self.default = (
            self._by_id[STRATEGY_PROSE] if STRATEGY_PROSE in self._by_id else specs[0]
        )

    @property
    def specs(self) -> tuple[StrategySpec, ...]:
        return self._specs

    def normalize(self, strategy: str | None) -> str:
        raw = _norm_key(strategy)
        if raw in self._by_id:
            return raw
        if raw in _ALIASES and _ALIASES[raw] in self._by_id:
            return _ALIASES[raw]
        return self.default.id

    def get(self, strategy: str | None) -> StrategySpec:
        return self._by_id[self.normalize(strategy)]

    def label(self, strategy: str | None) -> str:
        return self.get(strategy).label

    def choices(self) -> tuple[tuple[str, str], ...]:
        return tuple((s.label, s.id) for s in self._specs)


def make_registry(specs: tuple[StrategySpec, ...] | None = None) -> StrategyRegistry:
    return StrategyRegistry(specs or load_strategies())
