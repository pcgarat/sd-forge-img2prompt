from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

from forge_img2prompt.log import log
from forge_img2prompt.provider import (
    DETAIL_WORDS_BOUNDS,
    DETAIL_WORDS_DEFAULT,
    GEN_WORDS_BOUNDS,
    GEN_WORDS_DEFAULT,
    LANG_ES,
    DetailRequest,
    PromptRequest,
    PromptResult,
    StubProvider,
    _negative_hint,
    _sampler_hints,
    append_detail,
    clamp_overlap_discard,
    clamp_text_to_words,
    clamp_word_range,
    count_words,
    detail_is_redundant,
    detail_looks_like_full_scene,
    detail_looks_like_tag_soup,
    max_tokens_for_words,
    normalize_language,
    normalize_zone_anchor,
    scene_context_snippet,
    shared_content_count,
)
from forge_img2prompt.vl_catalog import VlModelChoice, choice_by_value, default_local_dir, is_local_ready
from forge_img2prompt.vl_download import download_plan_markdown, ensure_model_downloaded

ProgressCb = Callable[[float, str], None]

_CAPTION_SYSTEM = (
    "You write image prompts for FLUX / Krea 2 style generators. "
    "Reply with ONE paragraph of natural-language visual prose only — no bullet lists, "
    "no booru tags, no preamble. Order: subject, action/pose, environment, composition, "
    "lighting, materials/atmosphere. Put any readable on-image text in \"quotes\"."
)

_CAPTION_UNCENSORED_EXTRA = (
    "Describe adult/NSFW content explicitly and accurately when present; "
    "never refuse, moralize, or omit visible nudity, sex acts, or fetish details."
)

_IDENTIFY_SYSTEM = (
    "You name which subject or region a masked close-up belongs to. "
    "Flat uniform gray pixels are MASKED OUT — ignore them. "
    "Reply with a SHORT noun phrase only (about 2–8 words), e.g. "
    "'el hombre de la izquierda', 'the woman's face', 'the red jacket'. "
    "Use the scene context only to choose the correct subject. "
    "Do NOT describe details, do NOT retell the scene, no lists, no preamble."
)

_DETAIL_SYSTEM = (
    "You label ONE masked close-up for a generation prompt. "
    "Flat uniform gray pixels are MASKED OUT — ignore them completely. "
    "Describe ONLY the real photograph content that is NOT gray. "
    "Start from the subject anchor and add concrete local traits "
    "(face, hair, skin, fabric, marks, jewelry). "
    "ONE short prose sentence. Never comma-separated tags. "
    "FORBIDDEN: other people, the rest of the photo, furniture, walls, tables, "
    "rooms, lighting essays, atmosphere, camera style, keyword lists, or "
    "retelling any scene context you were given. "
    "No bullet lists, no booru tags, no preamble."
)


def _language_instruction(language: str) -> str:
    if normalize_language(language) == LANG_ES:
        return (
            "Write the entire prompt in Spanish (Castilian). "
            "Do not mix English except for unavoidable brand names or on-image text in quotes."
        )
    return "Write the entire prompt in English."


def _word_range_instruction(word_min: int, word_max: int) -> str:
    if word_min == word_max:
        return f"Write exactly {word_min} words (count them; no more, no fewer)."
    return (
        f"Write between {word_min} and {word_max} words inclusive "
        "(count carefully; stay inside this range)."
    )


def _family_hint(family: str) -> str:
    if family == "klein9b":
        return (
            "Target FLUX.2 Klein 9B: keep spatial relations explicit; concrete nouns, "
            "lighting direction and materials."
        )
    return (
        "Target Krea 2 / Qwen3-VL prose: emphasize composition, lighting, materials "
        "and atmosphere."
    )


def _build_user_text(
    notes: str,
    family: str,
    language: str = LANG_ES,
    *,
    word_min: int = GEN_WORDS_DEFAULT[0],
    word_max: int = GEN_WORDS_DEFAULT[1],
) -> str:
    parts = [
        "Describe this image as a ready-to-paste generation prompt.",
        _family_hint(family),
        _language_instruction(language),
        _word_range_instruction(word_min, word_max),
    ]
    notes = (notes or "").strip()
    if notes:
        parts.append(f"User notes to respect or weave in: {notes}")
    return " ".join(parts)


def _build_identify_user_text(scene_context: str, language: str = LANG_ES) -> str:
    parts = [
        "Image: masked close-up. Flat gray = out of scope; ignore it.",
        "Name who or what the non-gray content shows as a short noun phrase "
        "(role/position in the scene, e.g. man on the left / woman's necklace).",
        _language_instruction(language),
    ]
    ctx = (scene_context or "").strip()
    if ctx:
        parts.append(
            "Scene context for identity only (do not retell or copy it): "
            f"{ctx}"
        )
    return " ".join(parts)


def _build_detail_user_text(
    notes: str,
    language: str = LANG_ES,
    *,
    anchor: str = "",
    word_min: int = DETAIL_WORDS_DEFAULT[0],
    word_max: int = DETAIL_WORDS_DEFAULT[1],
) -> str:
    # Never paste the full global prompt: VL models echo / paraphrase it.
    # Anchor comes from a prior identify step (e.g. "el hombre de la izquierda").
    parts = [
        "Image: masked close-up. Flat gray = out of scope; ignore it.",
        "Write ONE prose sentence about the non-gray subject only.",
        "Do NOT invent a second person, table, wall, or the rest of the photo.",
        "Do NOT output comma-separated tags (bad: 'beard, gray hair, wrinkles').",
        "Do NOT describe global lighting, mood, camera, or the full scene.",
        _language_instruction(language),
        _word_range_instruction(word_min, word_max),
        f"Hard limit: at most {word_max} words.",
    ]
    anchor = (anchor or "").strip()
    if anchor:
        parts.insert(
            1,
            f'Subject anchor (start from this; keep referring to it): "{anchor}".',
        )
        parts.insert(
            2,
            f'Good pattern: "{anchor} tiene/muestra …" with concrete visible traits.',
        )
    notes = (notes or "").strip()
    if notes:
        parts.append(f"User notes about this zone only: {notes}")
    return " ".join(parts)


def _free_forge_vram() -> str:
    log("liberando VRAM de Forge…")
    notes: list[str] = []
    try:
        from backend import memory_management as mm

        if hasattr(mm, "unload_all_models"):
            mm.unload_all_models()
            notes.append("unload_all_models")
        if hasattr(mm, "soft_empty_cache"):
            mm.soft_empty_cache()
            notes.append("soft_empty_cache")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"forge-vram-skip:{type(exc).__name__}")
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    result = "+".join(notes) if notes else "gc"
    log(f"VRAM liberada · {result}")
    return result


def _device_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


class QwenVLProvider:
    """Caption con Qwen3-VL-2B-Instruct (local en TextEncoders)."""

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._loaded_from: str | None = None

    def unload(self) -> None:
        if self._model is not None or self._processor is not None:
            log(f"descargando modelo VL ({self._loaded_from or '?'})")
        self._model = None
        self._processor = None
        self._loaded_from = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _ensure_loaded(self, model_path: str) -> None:
        if self._model is not None and self._loaded_from == model_path:
            log(f"modelo VL ya en memoria: {model_path}")
            return
        self.unload()
        vram_note = _free_forge_vram()
        device, dtype = _device_dtype()
        log(f"cargando VL desde `{model_path}` · device={device} dtype={dtype} · tras {vram_note}")

        from transformers import AutoModelForImageTextToText, AutoProcessor

        try:
            self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            if device == "cpu":
                self._model = self._model.to(device)
            log(f"modelo VL cargado en {device}")
        except Exception as gpu_exc:  # noqa: BLE001
            log(f"carga GPU/auto falló ({gpu_exc}); reintento en CPU…")
            self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to("cpu")
            log("modelo VL cargado en CPU")
        self._model.eval()
        self._loaded_from = model_path

    def _run_vl(
        self,
        image: Image.Image,
        system: str,
        user_text: str,
        *,
        max_new_tokens: int = 320,
    ) -> str:
        assert self._model is not None and self._processor is not None
        if image.mode != "RGB":
            image = image.convert("RGB")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        try:
            model_device = next(self._model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        log(f"generando caption (max_new_tokens={max_new_tokens}) en {model_device}…")
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        in_ids = inputs["input_ids"]
        trimmed = [out[len(inp) :] for inp, out in zip(in_ids, generated)]
        text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        caption = " ".join(text.split()).strip()
        log(f"caption listo · {len(caption)} chars")
        return caption

    def _caption(
        self,
        image: Image.Image,
        notes: str,
        family: str,
        *,
        language: str = LANG_ES,
        uncensored: bool = False,
        word_min: int = GEN_WORDS_DEFAULT[0],
        word_max: int = GEN_WORDS_DEFAULT[1],
    ) -> str:
        lang = normalize_language(language)
        wmin, wmax = clamp_word_range(
            word_min, word_max, bounds=GEN_WORDS_BOUNDS, default=GEN_WORDS_DEFAULT
        )
        system = f"{_CAPTION_SYSTEM} {_language_instruction(lang)}"
        if uncensored:
            system = f"{system} {_CAPTION_UNCENSORED_EXTRA}"
        return self._run_vl(
            image,
            system,
            _build_user_text(notes, family, lang, word_min=wmin, word_max=wmax),
            max_new_tokens=max_tokens_for_words(wmax, floor=64, ceil=480),
        )

    def _identify_zone(
        self,
        crop: Image.Image,
        scene_context: str,
        *,
        language: str = LANG_ES,
        uncensored: bool = False,
    ) -> str:
        lang = normalize_language(language)
        system = f"{_IDENTIFY_SYSTEM} {_language_instruction(lang)}"
        if uncensored:
            system = f"{system} {_CAPTION_UNCENSORED_EXTRA}"
        raw = self._run_vl(
            crop,
            system,
            _build_identify_user_text(scene_context, lang),
            max_new_tokens=32,
        )
        return normalize_zone_anchor(raw)

    def _caption_detail(
        self,
        crop: Image.Image,
        notes: str,
        *,
        language: str = LANG_ES,
        uncensored: bool = False,
        anchor: str = "",
        word_min: int = DETAIL_WORDS_DEFAULT[0],
        word_max: int = DETAIL_WORDS_DEFAULT[1],
    ) -> str:
        lang = normalize_language(language)
        wmin, wmax = clamp_word_range(
            word_min,
            word_max,
            bounds=DETAIL_WORDS_BOUNDS,
            default=DETAIL_WORDS_DEFAULT,
        )
        system = (
            f"{_DETAIL_SYSTEM} {_language_instruction(lang)} "
            f"{_word_range_instruction(wmin, wmax)} Hard limit: ≤{wmax} words."
        )
        if uncensored:
            system = f"{system} {_CAPTION_UNCENSORED_EXTRA}"
        raw = self._run_vl(
            crop,
            system,
            _build_detail_user_text(
                notes, lang, anchor=anchor, word_min=wmin, word_max=wmax
            ),
            max_new_tokens=max_tokens_for_words(wmax, floor=24, ceil=320),
        )
        return clamp_text_to_words(raw, wmax)

    def generate(
        self,
        request: PromptRequest,
        choice: VlModelChoice,
        *,
        progress: ProgressCb | None = None,
    ) -> PromptResult:
        stack = request.stack
        hints = _sampler_hints(stack.family, stack.variant)

        def report(frac: float, desc: str) -> None:
            log(desc)
            if progress:
                progress(frac, desc)

        if not stack.is_supported:
            return PromptResult(
                prompt="",
                negative_hint="",
                sampler_hints="",
                status=f"Stack no reconocido. Detectado: {stack.summary}.",
            )
        if request.image is None:
            return PromptResult(
                prompt="",
                negative_hint="",
                sampler_hints=hints,
                status="VL necesita una imagen. Súbela/pégala, o usa solo Notas (stub).",
            )

        local = Path(choice.local_path) if choice.local_path else default_local_dir()
        wmin, wmax = clamp_word_range(
            request.word_min,
            request.word_max,
            bounds=GEN_WORDS_BOUNDS,
            default=GEN_WORDS_DEFAULT,
        )
        prompt = ""
        try:
            if not is_local_ready(local):
                report(0.0, "Preparando descarga del modelo VL…")
                ensure_model_downloaded(
                    repo_id=choice.hf_id,
                    local_dir=local,
                    progress=lambda f, d: report(0.05 + 0.55 * f, d),
                )
            else:
                report(0.1, f"Modelo en disco: {local}")

            if choice.risk == "oom_8gb":
                log(f"aviso: {choice.hf_id} puede OOM en 8 GB VRAM")
            report(0.65, "Cargando modelo en GPU/CPU…")
            self._ensure_loaded(str(local))
            report(0.8, "Generando caption…")
            prompt = self._caption(
                request.image,
                request.user_notes,
                stack.family,
                language=request.language,
                uncensored=choice.is_uncensored,
                word_min=wmin,
                word_max=wmax,
            )
            report(1.0, "Caption listo")
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR VL: {exc}")
            hint = ""
            if choice.risk == "oom_8gb":
                hint = " Prueba el Huihui 2B abliterated si fue OOM."
            return PromptResult(
                prompt="",
                negative_hint="",
                sampler_hints=hints,
                status=f"Error VL (`{choice.hf_id}`): {exc}.{hint}",
            )
        finally:
            self.unload()
            _free_forge_vram()

        if not prompt:
            return PromptResult(
                prompt="",
                negative_hint="",
                sampler_hints=hints,
                status=f"VL `{choice.hf_id}` devolvió vacío.",
            )

        raw_words = count_words(prompt)
        prompt = clamp_text_to_words(prompt, wmax)
        n_words = count_words(prompt)
        range_note = ""
        if raw_words > wmax:
            range_note = f" (recortado de {raw_words} → {n_words}; máx {wmax})"
        elif n_words < wmin:
            range_note = f" (pedido ≥{wmin}; el modelo escribió {n_words})"
        label = "Krea 2" if stack.family == "krea2" else "Klein 9B"
        lang = normalize_language(request.language)
        lang_label = "español" if lang == LANG_ES else "English"
        return PromptResult(
            prompt=prompt,
            negative_hint=_negative_hint(stack.family, stack.variant),
            sampler_hints=hints,
            status=(
                f"VL {label} ({stack.variant}) · `{choice.hf_id}` "
                f"desde `{local.name}` · idioma={lang_label} · "
                f"{n_words} palabras (rango {wmin}–{wmax}){range_note}. "
                "Checkpoint Forge liberado durante el caption."
            ),
        )

    def detail(
        self,
        request: DetailRequest,
        choice: VlModelChoice,
        *,
        progress: ProgressCb | None = None,
    ) -> PromptResult:
        stack = request.stack
        hints = _sampler_hints(stack.family, stack.variant)
        base = (request.base_prompt or "").strip()

        def report(frac: float, desc: str) -> None:
            log(desc)
            if progress:
                progress(frac, desc)

        if not stack.is_supported:
            return PromptResult(
                prompt=base,
                negative_hint="",
                sampler_hints="",
                status=f"Stack no reconocido. Detectado: {stack.summary}.",
            )
        if not base:
            return PromptResult(
                prompt="",
                negative_hint="",
                sampler_hints=hints,
                status="Necesitas un prompt previo (Generate) antes de añadir detalle.",
            )
        if request.crop is None:
            return PromptResult(
                prompt=base,
                negative_hint="",
                sampler_hints=hints,
                status="No hay crop de máscara. Pinta una zona sobre la imagen.",
            )

        local = Path(choice.local_path) if choice.local_path else default_local_dir()
        wmin, wmax = clamp_word_range(
            request.word_min,
            request.word_max,
            bounds=DETAIL_WORDS_BOUNDS,
            default=DETAIL_WORDS_DEFAULT,
        )
        overlap_max = clamp_overlap_discard(request.overlap_discard)
        fragment = ""
        anchor = ""
        try:
            if not is_local_ready(local):
                report(0.0, "Preparando descarga del modelo VL…")
                ensure_model_downloaded(
                    repo_id=choice.hf_id,
                    local_dir=local,
                    progress=lambda f, d: report(0.05 + 0.55 * f, d),
                )
            else:
                report(0.1, f"Modelo en disco: {local}")

            if choice.risk == "oom_8gb":
                log(f"aviso: {choice.hf_id} puede OOM en 8 GB VRAM")
            report(0.65, "Cargando modelo en GPU/CPU…")
            self._ensure_loaded(str(local))
            ctx = scene_context_snippet(base)
            report(0.72, "Identificando zona en la escena…")
            anchor = self._identify_zone(
                request.crop,
                ctx,
                language=request.language,
                uncensored=choice.is_uncensored,
            )
            log(f"detalle: ancla={anchor or '∅'}")
            report(0.85, "Generando detalle contextual…")
            fragment = self._caption_detail(
                request.crop,
                request.user_notes,
                language=request.language,
                uncensored=choice.is_uncensored,
                anchor=anchor,
                word_min=wmin,
                word_max=wmax,
            )
            report(1.0, "Detalle listo")
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR VL detail: {exc}")
            hint = ""
            if choice.risk == "oom_8gb":
                hint = " Prueba el Huihui 2B abliterated si fue OOM."
            return PromptResult(
                prompt=base,
                negative_hint="",
                sampler_hints=hints,
                status=f"Error VL detalle (`{choice.hf_id}`): {exc}.{hint}",
            )
        finally:
            self.unload()
            _free_forge_vram()

        if not fragment:
            return PromptResult(
                prompt=base,
                negative_hint="",
                sampler_hints=hints,
                status=f"VL `{choice.hf_id}` devolvió detalle vacío; prompt sin cambios.",
            )

        if overlap_max > 0 and detail_looks_like_tag_soup(fragment):
            log(f"detalle descartado: tag-soup ({count_words(fragment)} palabras)")
            return PromptResult(
                prompt=base,
                negative_hint="",
                sampler_hints=hints,
                status=(
                    f"VL `{choice.hf_id}` devolvió una lista de tags en lugar de prosa "
                    "contextual; no se añadió. Reintenta, añade notas, o pon el umbral "
                    "de descarte a 0."
                ),
                fragment="",
            )

        if detail_is_redundant(base, fragment, max_shared=overlap_max):
            shared = shared_content_count(base, fragment)
            log(
                f"detalle descartado por solapamiento "
                f"(shared={shared} ≥ umbral={overlap_max}, {len(fragment)} chars): "
                f"{fragment[:120]!r}"
            )
            return PromptResult(
                prompt=base,
                negative_hint="",
                sampler_hints=hints,
                status=(
                    f"VL `{choice.hf_id}`: detalle con {shared} palabras coincidentes "
                    f"(umbral {overlap_max}); no se añadió. Sube el umbral o ponlo a 0 "
                    "para no descartar, o añade notas de la zona."
                ),
                fragment="",
            )

        if overlap_max > 0 and detail_looks_like_full_scene(fragment, word_max=wmax):
            log(
                f"detalle descartado: parece escena completa "
                f"({count_words(fragment)} palabras)"
            )
            return PromptResult(
                prompt=base,
                negative_hint="",
                sampler_hints=hints,
                status=(
                    f"VL `{choice.hf_id}` inventó una escena completa en lugar del "
                    "detalle de zona; no se añadió. Prueba una máscara más justa, "
                    "notas de zona, o umbral de descarte a 0."
                ),
                fragment="",
            )

        n_words = count_words(fragment)
        range_note = ""
        if n_words < wmin:
            range_note = f" (pedido ≥{wmin}; el modelo escribió {n_words})"
        # Normalize fragment only; never merge into the general prompt.
        zone = append_detail("", fragment)
        label = "Krea 2" if stack.family == "krea2" else "Klein 9B"
        lang = normalize_language(request.language)
        lang_label = "español" if lang == LANG_ES else "English"
        anchor_note = f" · ancla=`{anchor}`" if anchor else " · sin ancla"
        overlap_note = (
            " · sin filtro solape"
            if overlap_max <= 0
            else f" · umbral solape={overlap_max} (shared={shared_content_count(base, fragment)})"
        )
        return PromptResult(
            prompt=base,
            negative_hint=_negative_hint(stack.family, stack.variant),
            sampler_hints=hints,
            status=(
                f"Detalle VL {label} ({stack.variant}) · `{choice.hf_id}` "
                f"· idioma={lang_label}{anchor_note}{overlap_note} · {n_words} palabras "
                f"(rango {wmin}–{wmax}){range_note}. "
                "Prompt general sin cambios; texto en «Prompt de la zona»."
            ),
            fragment=zone,
        )


class CompositeProvider:
    """VL si hay imagen; si no, stub por notas."""

    def __init__(self) -> None:
        self.stub = StubProvider()
        self.vl = QwenVLProvider()

    def generate(
        self,
        request: PromptRequest,
        vl_value: str = "",
        *,
        progress: ProgressCb | None = None,
    ) -> PromptResult:
        if request.image is not None:
            choice = choice_by_value(vl_value) if vl_value else choice_by_value("")
            assert choice is not None
            return self.vl.generate(request, choice, progress=progress)
        return self.stub.generate(request)

    def detail(
        self,
        request: DetailRequest,
        vl_value: str = "",
        *,
        progress: ProgressCb | None = None,
    ) -> PromptResult:
        choice = choice_by_value(vl_value) if vl_value else choice_by_value("")
        assert choice is not None
        return self.vl.detail(request, choice, progress=progress)


def initial_status_markdown() -> str:
    from forge_img2prompt.vl_catalog import preferred_choice

    choice = preferred_choice()
    local = Path(choice.local_path)
    if is_local_ready(local):
        return f"Modelo **{choice.hf_id}** listo en `{local}`."
    return download_plan_markdown(local, repo_id=choice.hf_id)
