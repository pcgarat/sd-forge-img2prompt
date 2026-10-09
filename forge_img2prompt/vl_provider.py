from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

from forge_img2prompt.log import log
from forge_img2prompt import prompt_log
from forge_img2prompt.provider import (
    DEFAULT_STRATEGY,
    DETAIL_WORDS_BOUNDS,
    DETAIL_WORDS_DEFAULT,
    GEN_WORDS_BOUNDS,
    GEN_WORDS_DEFAULT,
    LANG_ES,
    REF_IMAGES_MAX,
    STRATEGY_REGISTRY,
    DetailRequest,
    PromptRequest,
    PromptResult,
    StubProvider,
    _negative_hint,
    _sampler_hints,
    active_ref_pairs,
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
    normalize_strategy,
    normalize_zone_anchor,
    ref_images_instruction,
    ref_label,
    scene_context_snippet,
    shared_content_count,
    strategy_label,
)
from forge_img2prompt.vl_catalog import VlModelChoice, choice_by_value, default_local_dir, is_local_ready
from forge_img2prompt.vl_download import download_plan_markdown, ensure_model_downloaded

ProgressCb = Callable[[float, str], None]

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


def _caption_system_for(strategy: str | None) -> str:
    spec = STRATEGY_REGISTRY.get(strategy)
    return f"{spec.caption_system} {_COMMON_CAPTION_RULES}"


def _notes_system_for(strategy: str | None) -> str:
    spec = STRATEGY_REGISTRY.get(strategy)
    return f"{spec.notes_system} {_COMMON_NOTES_RULES}"


# Alias de compatibilidad (tests / imports antiguos = estrategia por defecto).
_CAPTION_SYSTEM = _caption_system_for(DEFAULT_STRATEGY)
_NOTES_SYSTEM = _notes_system_for(DEFAULT_STRATEGY)

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
    "Natural language only — never comma-separated tags. "
    "Be concrete, precise, and concise; no decorative filler. "
    "Never use negative phrasing (forbidden: 'there is no…', 'no hay…', "
    "'without…', listing absences). Describe only what is visible. "
    "Mention text in \"quotes\" only if readable text is visible in the non-gray area; "
    "otherwise do not mention text at all. "
    "FORBIDDEN: other people, the rest of the photo, furniture, walls, tables, "
    "rooms, lighting essays, atmosphere, camera style, keyword lists, or "
    "retelling any scene context you were given. "
    "No bullet lists, no booru tags, no preamble. "
    "Do not choose your own length; obey only the word-count range given."
)


def _language_instruction(language: str) -> str:
    if normalize_language(language) == LANG_ES:
        return (
            "Write the entire prompt in Spanish (Castilian). "
            "Do not mix English except for unavoidable brand names "
            "or on-image text that is actually visible (then use «quotes»)."
        )
    return (
        "Write the entire prompt in English. "
        "Quote on-image text only when it is actually visible."
    )


def _word_range_instruction(word_min: int, word_max: int) -> str:
    if word_min == word_max:
        return (
            f"LENGTH (mandatory, from UI sliders): write exactly {word_min} words "
            "(count them; no more, no fewer). Ignore any other length habit."
        )
    return (
        f"LENGTH (mandatory, from UI sliders): write between {word_min} and {word_max} "
        "words inclusive (count carefully; stay inside this range). "
        "Ignore any other length habit."
    )


def _family_hint(family: str) -> str:
    if family == "klein9b":
        return (
            "Target FLUX.2 Klein 9B: keep spatial relations explicit; concrete nouns, "
            "lighting direction and materials."
        )
    return (
        "Target Krea 2: emphasize composition, lighting and materials "
        "with concrete, information-dense language."
    )


def _strategy_user_hint(strategy: str | None) -> str:
    spec = STRATEGY_REGISTRY.get(strategy)
    if spec.user_hint:
        return spec.user_hint
    return "Be concrete, precise, and concise; no ornamental wording."


def _build_user_text(
    notes: str,
    family: str,
    language: str = LANG_ES,
    *,
    strategy: str = DEFAULT_STRATEGY,
    word_min: int = GEN_WORDS_DEFAULT[0],
    word_max: int = GEN_WORDS_DEFAULT[1],
    ref_count: int = 0,
    ref_indices: list[int] | tuple[int, ...] | None = None,
) -> str:
    parts = [
        "Describe this image as a ready-to-paste generation prompt in natural language.",
        _strategy_user_hint(strategy),
        "Never say what is absent (no 'there is no…' / 'no hay…').",
        "If the main image has no readable text, do not mention text at all.",
        _family_hint(family),
        _language_instruction(language),
        _word_range_instruction(word_min, word_max),
    ]
    ref_block = ref_images_instruction(
        ref_count, for_detail=False, indices=ref_indices
    )
    if ref_block:
        parts.append(ref_block)
    notes = (notes or "").strip()
    if notes:
        parts.append(f"User notes to respect or weave in: {notes}")
    return " ".join(parts)


def _build_notes_only_user_text(
    notes: str,
    family: str,
    language: str = LANG_ES,
    *,
    strategy: str = DEFAULT_STRATEGY,
    word_min: int = GEN_WORDS_DEFAULT[0],
    word_max: int = GEN_WORDS_DEFAULT[1],
    ref_indices: list[int] | tuple[int, ...] | None = None,
) -> str:
    """User message when Generate runs without a main image (brief → prompt)."""
    parts = [
        "No main photograph was provided.",
        "Write a ready-to-paste generation prompt from the user brief.",
        "Add concrete visual detail that enriches the brief; stay faithful to it.",
        _strategy_user_hint(strategy),
        "Never say what is absent (no 'there is no…' / 'no hay…').",
        "Mention text only if the brief requests readable on-image text.",
        _family_hint(family),
        _language_instruction(language),
        _word_range_instruction(word_min, word_max),
    ]
    ref_block = ref_images_instruction(
        0, for_detail=False, indices=ref_indices
    )
    if ref_block:
        parts.append(
            "Reference images may be attached (no primary scene photo). "
            + ref_block.replace(
                "first = main scene to caption (primary); then reference(s)",
                "attached reference(s)",
                1,
            )
        )
    brief = (notes or "").strip()
    if brief:
        parts.append(f"User brief: {brief}")
    else:
        parts.append("User brief: (empty — invent nothing; ask would be empty).")
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
    ref_count: int = 0,
    ref_indices: list[int] | tuple[int, ...] | None = None,
) -> str:
    # Never paste the full global prompt: VL models echo / paraphrase it.
    # Anchor comes from a prior identify step (e.g. "el hombre de la izquierda").
    parts = [
        "Image: masked close-up. Flat gray = out of scope; ignore it.",
        "Describe the non-gray subject only in natural language.",
        "Be concrete, precise, and concise; no ornamental wording.",
        "Never say what is absent (no 'there is no…' / 'no hay…').",
        "If there is no readable text in the non-gray area, do not mention text.",
        "Do NOT invent a second person, table, wall, or the rest of the photo.",
        "Do NOT output comma-separated tags (bad: 'beard, gray hair, wrinkles').",
        "Do NOT describe global lighting, mood, camera, or the full scene.",
        _language_instruction(language),
        _word_range_instruction(word_min, word_max),
        f"Hard limit: at most {word_max} words.",
    ]
    ref_block = ref_images_instruction(
        ref_count, for_detail=True, indices=ref_indices
    )
    if ref_block:
        parts.append(ref_block)
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


def _pack_primary_and_refs(
    primary: Image.Image,
    ref_slots: tuple[Image.Image | None, ...] | list[Image.Image | None] | None = None,
    *,
    # Back-compat: bare list of images (dense foto 1..N)
    ref_images: tuple[Image.Image, ...] | list[Image.Image] | None = None,
) -> tuple[list[Image.Image], list[int]]:
    """Return (images=[primary, …refs], slot_indices for each ref)."""
    if ref_slots is not None:
        pairs = active_ref_pairs(ref_slots)
    elif ref_images:
        pairs = [(i, img) for i, img in enumerate(list(ref_images)[:REF_IMAGES_MAX], start=1)]
    else:
        pairs = []
    images = [primary, *[img for _, img in pairs]]
    indices = [idx for idx, _ in pairs]
    return images, indices


def _pack_refs_only(
    ref_slots: tuple[Image.Image | None, ...] | list[Image.Image | None] | None = None,
    *,
    ref_images: tuple[Image.Image, ...] | list[Image.Image] | None = None,
) -> tuple[list[Image.Image], list[int]]:
    """Images + slot indices when there is no primary/main photo."""
    if ref_slots is not None:
        pairs = active_ref_pairs(ref_slots)
    elif ref_images:
        pairs = [
            (i, img)
            for i, img in enumerate(list(ref_images)[:REF_IMAGES_MAX], start=1)
        ]
    else:
        pairs = []
    return [img for _, img in pairs], [idx for idx, _ in pairs]


def _vl_user_content(
    images: list[Image.Image],
    user_text: str,
    *,
    ref_indices: list[int] | tuple[int, ...] | None = None,
    primary_label: str = "Primary image.",
    has_primary: bool = True,
) -> list[dict[str, Any]]:
    """Interleave images + labels for Qwen-style chat templates."""
    content: list[dict[str, Any]] = []
    for i, img in enumerate(images):
        if img.mode != "RGB":
            img = img.convert("RGB")
        content.append({"type": "image", "image": img})
        if has_primary and i == 0:
            content.append({"type": "text", "text": primary_label})
        else:
            # With primary: refs start at content index 1 → ref_indices[0]
            # Without primary: every image is a labeled ref
            ref_i = i - 1 if has_primary else i
            slot = (
                ref_indices[ref_i]
                if ref_indices is not None and 0 <= ref_i < len(ref_indices)
                else (i if has_primary else i + 1)
            )
            content.append({"type": "text", "text": ref_label(slot)})
    content.append({"type": "text", "text": user_text})
    return content


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
    notes.append(_release_cuda())
    result = "+".join(notes) if notes else "gc"
    log(f"VRAM liberada · {result}")
    return result


def _release_cuda() -> str:
    """GC + empty_cache (+ ipc_collect). Devuelve etiqueta corta para logs."""
    gc.collect()
    gc.collect()
    if not torch.cuda.is_available():
        return "gc"
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        free_gb = free_b / (1024**3)
        total_gb = total_b / (1024**3)
        log(f"CUDA libre ≈ {free_gb:.2f}/{total_gb:.2f} GB")
        return f"cuda_free≈{free_gb:.1f}G"
    except Exception:
        return "cuda_empty"


def _move_model_to_cpu(model: Any) -> None:
    """Best-effort: sacamos pesos de GPU aunque venga con device_map=auto."""
    if model is None:
        return
    try:
        if hasattr(model, "hf_device_map"):
            try:
                model.hf_device_map.clear()
            except Exception:
                pass
    except Exception:
        pass
    try:
        model.to("cpu")
        return
    except Exception:
        pass
    try:
        model.cpu()
        return
    except Exception:
        pass
    # Último recurso: mover parámetro a parámetro
    try:
        for p in model.parameters():
            if p.is_cuda:
                p.data = p.data.to("cpu")
        for b in model.buffers():
            if b.is_cuda:
                b.data = b.data.to("cpu")
    except Exception:
        pass


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
        model = self._model
        processor = self._processor
        path = self._loaded_from
        self._model = None
        self._processor = None
        self._loaded_from = None
        if model is not None or processor is not None:
            log(f"descargando modelo VL ({path or '?'})")
        if model is not None:
            _move_model_to_cpu(model)
            try:
                del model
            except Exception:
                pass
        if processor is not None:
            try:
                del processor
            except Exception:
                pass
        note = _release_cuda()
        log(f"VL unload hecho · {note}")

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
            # max_memory fuerza a caber; sin device_map el unload a CPU es más fiable.
            # Preferimos .to(cuda) explícito en GPU única.
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            if device == "cuda":
                self._model = self._model.to(device)
            log(f"modelo VL cargado en {device}")
        except Exception as gpu_exc:  # noqa: BLE001
            log(f"carga GPU/auto falló ({gpu_exc}); reintento en CPU…")
            self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to("cpu")
            log("modelo VL cargado en CPU")
        self._model.eval()
        self._loaded_from = model_path

    def _run_vl(
        self,
        image: Image.Image | list[Image.Image] | None,
        system: str,
        user_text: str,
        *,
        max_new_tokens: int = 320,
        ref_indices: list[int] | tuple[int, ...] | None = None,
        has_primary: bool = True,
    ) -> str:
        assert self._model is not None and self._processor is not None
        if image is None:
            images: list[Image.Image] = []
        elif isinstance(image, list):
            images = list(image)
        else:
            images = [image]
        images = [
            (im.convert("RGB") if im.mode != "RGB" else im) for im in images if im is not None
        ]

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system}],
            },
            {
                "role": "user",
                "content": _vl_user_content(
                    images,
                    user_text,
                    ref_indices=ref_indices,
                    has_primary=has_primary and bool(images),
                ),
            },
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_log.record(
            transport="transformers apply_chat_template → generate",
            payload=prompt_log.summarize_vl_messages(messages),
            params={
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "device": str(next(self._model.parameters()).device),
                "inputs": {
                    k: getattr(v, "shape", None) and list(v.shape)
                    for k, v in inputs.items()
                },
            },
            images=images,
        )
        try:
            model_device = next(self._model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        log(
            f"generando caption (max_new_tokens={max_new_tokens}, "
            f"images={len(images)}) en {model_device}…"
        )
        generated = None
        try:
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
        finally:
            # Los tensores de generate/inputs se quedan en VRAM si no se borran.
            try:
                del inputs
            except Exception:
                pass
            if generated is not None:
                try:
                    del generated
                except Exception:
                    pass
            _release_cuda()

    def _caption(
        self,
        image: Image.Image,
        notes: str,
        family: str,
        *,
        language: str = LANG_ES,
        strategy: str = DEFAULT_STRATEGY,
        uncensored: bool = False,
        word_min: int = GEN_WORDS_DEFAULT[0],
        word_max: int = GEN_WORDS_DEFAULT[1],
        ref_slots: tuple[Image.Image | None, ...] | list[Image.Image | None] | None = None,
        ref_images: tuple[Image.Image, ...] | list[Image.Image] | None = None,
    ) -> str:
        lang = normalize_language(language)
        strat = normalize_strategy(strategy)
        wmin, wmax = clamp_word_range(
            word_min, word_max, bounds=GEN_WORDS_BOUNDS, default=GEN_WORDS_DEFAULT
        )
        packed, indices = _pack_primary_and_refs(
            image, ref_slots=ref_slots, ref_images=ref_images
        )
        system = (
            f"{_caption_system_for(strat)} {_language_instruction(lang)} "
            f"{_word_range_instruction(wmin, wmax)}"
        )
        if uncensored:
            system = f"{system} {_CAPTION_UNCENSORED_EXTRA}"
        return self._run_vl(
            packed,
            system,
            _build_user_text(
                notes,
                family,
                lang,
                strategy=strat,
                word_min=wmin,
                word_max=wmax,
                ref_count=len(indices),
                ref_indices=indices,
            ),
            max_new_tokens=max_tokens_for_words(wmax, floor=64),
            ref_indices=indices,
            has_primary=True,
        )

    def _from_notes(
        self,
        notes: str,
        family: str,
        *,
        language: str = LANG_ES,
        strategy: str = DEFAULT_STRATEGY,
        uncensored: bool = False,
        word_min: int = GEN_WORDS_DEFAULT[0],
        word_max: int = GEN_WORDS_DEFAULT[1],
        ref_slots: tuple[Image.Image | None, ...] | list[Image.Image | None] | None = None,
    ) -> str:
        lang = normalize_language(language)
        strat = normalize_strategy(strategy)
        wmin, wmax = clamp_word_range(
            word_min, word_max, bounds=GEN_WORDS_BOUNDS, default=GEN_WORDS_DEFAULT
        )
        packed, indices = _pack_refs_only(ref_slots=ref_slots)
        system = (
            f"{_notes_system_for(strat)} {_language_instruction(lang)} "
            f"{_word_range_instruction(wmin, wmax)}"
        )
        if uncensored:
            system = f"{system} {_CAPTION_UNCENSORED_EXTRA}"
        return self._run_vl(
            packed or None,
            system,
            _build_notes_only_user_text(
                notes,
                family,
                lang,
                strategy=strat,
                word_min=wmin,
                word_max=wmax,
                ref_indices=indices,
            ),
            max_new_tokens=max_tokens_for_words(wmax, floor=64),
            ref_indices=indices,
            has_primary=False,
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
        ref_slots: tuple[Image.Image | None, ...] | list[Image.Image | None] | None = None,
        ref_images: tuple[Image.Image, ...] | list[Image.Image] | None = None,
    ) -> str:
        lang = normalize_language(language)
        wmin, wmax = clamp_word_range(
            word_min,
            word_max,
            bounds=DETAIL_WORDS_BOUNDS,
            default=DETAIL_WORDS_DEFAULT,
        )
        packed, indices = _pack_primary_and_refs(
            crop, ref_slots=ref_slots, ref_images=ref_images
        )
        system = (
            f"{_DETAIL_SYSTEM} {_language_instruction(lang)} "
            f"{_word_range_instruction(wmin, wmax)} Hard limit: ≤{wmax} words."
        )
        if uncensored:
            system = f"{system} {_CAPTION_UNCENSORED_EXTRA}"
        raw = self._run_vl(
            packed,
            system,
            _build_detail_user_text(
                notes,
                lang,
                anchor=anchor,
                word_min=wmin,
                word_max=wmax,
                ref_count=len(indices),
                ref_indices=indices,
            ),
            max_new_tokens=max_tokens_for_words(wmax, floor=24),
            ref_indices=indices,
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
            notes = (request.user_notes or "").strip()
            if not notes:
                return PromptResult(
                    prompt="",
                    negative_hint="",
                    sampler_hints=hints,
                    status=(
                        "Sin imagen principal: escribe en **Notas** qué quieres "
                        "generar y pulsa Generate (el modelo ampliará el brief)."
                    ),
                )

        local = Path(choice.local_path) if choice.local_path else default_local_dir()
        wmin, wmax = clamp_word_range(
            request.word_min,
            request.word_max,
            bounds=GEN_WORDS_BOUNDS,
            default=GEN_WORDS_DEFAULT,
        )
        prompt = ""
        notes_only = request.image is None
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
            if notes_only:
                report(0.8, "Expandiendo notas a prompt…")
                prompt = self._from_notes(
                    request.user_notes,
                    stack.family,
                    language=request.language,
                    strategy=request.strategy,
                    uncensored=choice.is_uncensored,
                    word_min=wmin,
                    word_max=wmax,
                    ref_slots=request.ref_slots,
                )
            else:
                report(0.8, "Generando caption…")
                prompt = self._caption(
                    request.image,
                    request.user_notes,
                    stack.family,
                    language=request.language,
                    strategy=request.strategy,
                    uncensored=choice.is_uncensored,
                    word_min=wmin,
                    word_max=wmax,
                    ref_slots=request.ref_slots,
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
        strat_label = strategy_label(request.strategy)
        mode = "notas→prompt" if notes_only else "imagen→prompt"
        return PromptResult(
            prompt=prompt,
            negative_hint=_negative_hint(stack.family, stack.variant),
            sampler_hints=hints,
            status=(
                f"VL {label} ({stack.variant}) · `{choice.hf_id}` "
                f"desde `{local.name}` · {mode} · estrategia={strat_label} · "
                f"idioma={lang_label} · "
                f"{n_words} palabras (rango {wmin}–{wmax}){range_note}. "
                "VL descargado de VRAM; Forge puede recargar el checkpoint."
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
                ref_slots=request.ref_slots,
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
                    f"VL `{choice.hf_id}` devolvió una lista de tags en lugar de lenguaje natural "
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
    """Caption VL/Ollama/NaN (imagen o notas); stub solo si no hay nada que enviar."""

    def __init__(self) -> None:
        self.stub = StubProvider()
        self.vl = QwenVLProvider()
        from forge_img2prompt.ollama_vl import NanVLProvider, OllamaVLProvider

        self.ollama = OllamaVLProvider()
        self.nan = NanVLProvider()
        # (predicado sobre la elección, método) — despacho table-driven.
        self._routes = (
            (lambda c: c.is_nan, "nan"),
            (lambda c: c.is_ollama, "ollama"),
        )

    @staticmethod
    def _remote(request: PromptRequest) -> bool:
        notes = (request.user_notes or "").strip()
        has_refs = bool(active_ref_pairs(request.ref_slots))
        return request.image is not None or bool(notes) or has_refs

    def _provider_for(self, choice: VlModelChoice):
        for matches, attr in self._routes:
            if matches(choice):
                return getattr(self, attr)
        return self.vl

    def generate(
        self,
        request: PromptRequest,
        vl_value: str = "",
        *,
        progress: ProgressCb | None = None,
    ) -> PromptResult:
        # Sin imagen: el modelo expande Notas (refs opcionales). Stub solo si vacío.
        if self._remote(request):
            choice = choice_by_value(vl_value) if vl_value else choice_by_value("")
            assert choice is not None
            return self._provider_for(choice).generate(request, choice, progress=progress)
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
        return self._provider_for(choice).detail(request, choice, progress=progress)


def initial_status_markdown() -> str:
    from forge_img2prompt.vl_catalog import preferred_choice

    choice = preferred_choice()
    if choice.is_nan:
        return f"Backend **NaN** · modelo `{choice.hf_id}` (API, sin VRAM Forge)."
    if choice.is_ollama:
        return f"Backend **Ollama** · modelo `{choice.hf_id}` (sin VRAM Forge)."
    local = Path(choice.local_path)
    if is_local_ready(local):
        return f"Modelo **{choice.hf_id}** listo en `{local}`."
    return download_plan_markdown(local, repo_id=choice.hf_id)
