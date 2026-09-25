from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from forge_img2prompt.log import log
from forge_img2prompt.stack import StackInfo

_TAG_SOUP_HINTS = (", masterpiece", "1girl,", "best quality,", "ultra detailed,")


@dataclass(frozen=True)
class PromptRequest:
    image: Image.Image | None
    user_notes: str
    stack: StackInfo


@dataclass(frozen=True)
class PromptResult:
    prompt: str
    negative_hint: str
    sampler_hints: str
    status: str = ""


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


def _notes_to_prose(notes: str, family: str) -> str:
    """Turn user notes into natural-language prose. Empty notes → empty string."""
    text = _normalize_notes(notes)
    if not text:
        return ""

    if _looks_like_tag_soup(text):
        core = text.replace(",", ", ")
        return (
            f"A detailed scene: {core}. "
            "Describe the subject first, then environment, composition, lighting, materials and mood "
            "in connected prose — not a keyword list."
        )

    if text[0].islower():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."

    # Light family-specific framing so the same notes are not a bare echo.
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

        prompt = _notes_to_prose(notes, stack.family)
        status = (
            f"Stub {label} ({stack.variant}) · {img_info}. "
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
