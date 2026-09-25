from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

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


def _sampler_hints(variant: str) -> str:
    if variant == "turbo":
        return "Krea 2 Turbo: ~8 steps, CFG ≈ 0–1, Euler + Simple, clip skip 1."
    if variant == "raw":
        return "Krea 2 RAW: ~28 steps, CFG ≈ 4.5, Euler + Simple, clip skip 1."
    return "Krea 2: Turbo ≈ 8 steps / CFG bajo; RAW ≈ 28 steps / CFG ~4.5 (Euler + Simple)."


def _negative_hint(variant: str) -> str:
    if variant == "turbo":
        return "Turbo/distilled: el negativo suele aportar poco; describe exclusiones en el prompt positivo."
    return ""


def _looks_like_tag_soup(text: str) -> bool:
    low = text.lower()
    if low.count(",") >= 6 and " " in low:
        return True
    return any(h in low for h in _TAG_SOUP_HINTS)


def _notes_to_prose(notes: str) -> str:
    text = " ".join(notes.split()).strip().strip('"')
    if not text:
        return (
            "A clear photograph of the main subject centered in frame, natural lighting, "
            'realistic materials and colors. If text appears in the image, put exact words in "quotes".'
        )
    if _looks_like_tag_soup(text):
        return (
            f"A detailed scene described as: {text.replace(',', ', ')}. "
            "Render as coherent natural-language imagery with explicit composition, lighting, "
            "and materials rather than a keyword list."
        )
    if text[0].islower():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


class StubProvider:
    """v1 provider: no vision backend; reshapes user notes into Krea 2 prose."""

    def generate(self, request: PromptRequest) -> PromptResult:
        stack = request.stack
        if not stack.is_supported:
            return PromptResult(
                prompt="",
                negative_hint="",
                sampler_hints="",
                status=(
                    "Stack no reconocido como Krea 2 (o TE incompatible). "
                    f"Detectado: {stack.summary}. No se genera prompt «optimizado»."
                ),
            )

        prompt = _notes_to_prose(request.user_notes)
        has_image = request.image is not None
        status = f"Stub Krea 2 ({stack.variant})."
        if has_image:
            status += " Imagen recibida (caption real pendiente de backend)."
        if not request.user_notes.strip():
            status += " Sin notas: plantilla mínima; edítala antes de generar."

        return PromptResult(
            prompt=prompt,
            negative_hint=_negative_hint(stack.variant),
            sampler_hints=_sampler_hints(stack.variant),
            status=status,
        )
