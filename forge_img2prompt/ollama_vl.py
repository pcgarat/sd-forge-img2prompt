"""Caption / detalle vía backend OpenAI/Ollama (sin cargar VL en el proceso Forge)."""

from __future__ import annotations

from typing import Callable

from forge_img2prompt.log import log
from forge_img2prompt.ollama_client import chat_with_image
from forge_img2prompt.ollama_settings import OllamaConfig, get_ollama_config
from forge_img2prompt.provider import (
    DEFAULT_STRATEGY,
    DETAIL_WORDS_BOUNDS,
    DETAIL_WORDS_DEFAULT,
    GEN_WORDS_BOUNDS,
    GEN_WORDS_DEFAULT,
    LANG_ES,
    DetailRequest,
    PromptRequest,
    PromptResult,
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
    normalize_strategy,
    normalize_zone_anchor,
    scene_context_snippet,
    shared_content_count,
    strategy_label,
)
from forge_img2prompt.vl_catalog import VlModelChoice
from forge_img2prompt.vl_provider import (
    _CAPTION_UNCENSORED_EXTRA,
    _DETAIL_SYSTEM,
    _IDENTIFY_SYSTEM,
    _build_detail_user_text,
    _build_identify_user_text,
    _build_notes_only_user_text,
    _build_user_text,
    _caption_system_for,
    _language_instruction,
    _notes_system_for,
    _pack_primary_and_refs,
    _pack_refs_only,
    _word_range_instruction,
)

ProgressCb = Callable[[float, str], None]


class OllamaVLProvider:
    """Generate/detail vía backend remoto OpenAI-compatible (Ollama / NaN).

    Ollama habla su API nativa ``/api/chat`` + campo ``images``; NaN habla
    ``/chat/completions`` + ``image_url``. Toda la lógica de prompts, rangos,
    refs y filtros post-VL es idéntica: ``NanVLProvider`` solo cambia la
    etiqueta, la config y el transporte ``_run``.
    """

    backend_label = "Ollama"

    # --- Puntos de extensión -------------------------------------------------

    def _config(self, model: str | None) -> OllamaConfig:
        return get_ollama_config(model=model)

    def _run(
        self,
        image,
        system: str,
        user_text: str,
        *,
        max_new_tokens: int,
        model: str | None = None,
    ) -> str:
        cfg = self._config(model)
        return chat_with_image(
            cfg,
            system=system,
            user_text=user_text,
            image=image,
            num_predict=max_new_tokens,
        )

    def _status_note(self, cfg: OllamaConfig, choice: VlModelChoice) -> str:
        return (
            f"`{cfg.model}` @ `{cfg.base_url}` · {self.backend_label} · "
            f"VRAM Forge intacta (VL en {self.backend_label}; "
            "keep_alive=0 → descarga tras caption)."
        )

    def _error_hint(self, exc: Exception, cfg: OllamaConfig, choice: VlModelChoice) -> str:
        return f"(`{cfg.model}` @ `{cfg.base_url}`): {exc}"

    # --- Prompts (idénticos en ambos backends) -------------------------------

    def _caption(
        self,
        image,
        notes: str,
        family: str,
        *,
        language: str = LANG_ES,
        strategy: str = DEFAULT_STRATEGY,
        uncensored: bool = False,
        word_min: int = GEN_WORDS_DEFAULT[0],
        word_max: int = GEN_WORDS_DEFAULT[1],
        model: str | None = None,
        ref_slots=None,
        ref_images=None,
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
        return self._run(
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
            model=model,
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
        model: str | None = None,
        ref_slots=None,
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
        return self._run(
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
            model=model,
        )

    def _identify_zone(
        self,
        crop,
        scene_context: str,
        *,
        language: str = LANG_ES,
        uncensored: bool = False,
        model: str | None = None,
    ) -> str:
        lang = normalize_language(language)
        system = f"{_IDENTIFY_SYSTEM} {_language_instruction(lang)}"
        if uncensored:
            system = f"{system} {_CAPTION_UNCENSORED_EXTRA}"
        raw = self._run(
            crop,
            system,
            _build_identify_user_text(scene_context, lang),
            max_new_tokens=32,
            model=model,
        )
        return normalize_zone_anchor(raw)

    def _caption_detail(
        self,
        crop,
        notes: str,
        *,
        language: str = LANG_ES,
        uncensored: bool = False,
        anchor: str = "",
        word_min: int = DETAIL_WORDS_DEFAULT[0],
        word_max: int = DETAIL_WORDS_DEFAULT[1],
        model: str | None = None,
        ref_slots=None,
        ref_images=None,
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
        raw = self._run(
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
            model=model,
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
        cfg = self._config(choice.hf_id)

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
                        f"generar y pulsa Generate ({self.backend_label} ampliará el brief)."
                    ),
                )

        wmin, wmax = clamp_word_range(
            request.word_min,
            request.word_max,
            bounds=GEN_WORDS_BOUNDS,
            default=GEN_WORDS_DEFAULT,
        )
        prompt = ""
        notes_only = request.image is None
        backend = self.backend_label
        try:
            report(0.15, f"{backend} `{cfg.model}` en {cfg.base_url}…")
            if notes_only:
                report(0.4, f"Expandiendo notas a prompt ({backend})…")
                prompt = self._from_notes(
                    request.user_notes,
                    stack.family,
                    language=request.language,
                    strategy=request.strategy,
                    uncensored=choice.is_uncensored,
                    word_min=wmin,
                    word_max=wmax,
                    model=choice.hf_id,
                    ref_slots=request.ref_slots,
                )
            else:
                report(0.4, f"Generando caption ({backend})…")
                prompt = self._caption(
                    request.image,
                    request.user_notes,
                    stack.family,
                    language=request.language,
                    strategy=request.strategy,
                    uncensored=choice.is_uncensored,
                    word_min=wmin,
                    word_max=wmax,
                    model=choice.hf_id,
                    ref_slots=request.ref_slots,
                )
            report(1.0, f"Caption {backend} listo")
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR {backend} VL: {exc}")
            return PromptResult(
                prompt="",
                negative_hint="",
                sampler_hints=hints,
                status=f"Error {backend}: {self._error_hint(exc, cfg, choice)}",
            )

        if not prompt:
            return PromptResult(
                prompt="",
                negative_hint="",
                sampler_hints=hints,
                status=f"{backend} `{cfg.model}` devolvió vacío.",
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
                f"{backend} {label} ({stack.variant}) · "
                f"{self._status_note(cfg, choice)} · {mode} · "
                f"estrategia={strat_label} · idioma={lang_label} · "
                f"{n_words} palabras (rango {wmin}–{wmax}){range_note}."
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
        cfg = self._config(choice.hf_id)
        backend = self.backend_label

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
            report(0.2, f"{backend} detalle `{cfg.model}`…")
            ctx = scene_context_snippet(base)
            report(0.45, f"Identificando zona ({backend})…")
            anchor = self._identify_zone(
                request.crop,
                ctx,
                language=request.language,
                uncensored=choice.is_uncensored,
                model=choice.hf_id,
            )
            log(f"detalle {backend}: ancla={anchor or '∅'}")
            report(0.75, f"Generando detalle ({backend})…")
            fragment = self._caption_detail(
                request.crop,
                request.user_notes,
                language=request.language,
                uncensored=choice.is_uncensored,
                anchor=anchor,
                word_min=wmin,
                word_max=wmax,
                model=choice.hf_id,
                ref_slots=request.ref_slots,
            )
            report(1.0, f"Detalle {backend} listo")
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR {backend} detail: {exc}")
            return PromptResult(
                prompt=base,
                negative_hint="",
                sampler_hints=hints,
                status=f"Error {backend} detalle: {self._error_hint(exc, cfg, choice)}",
            )

        if not fragment:
            return PromptResult(
                prompt=base,
                negative_hint="",
                sampler_hints=hints,
                status=f"{backend} `{cfg.model}` devolvió detalle vacío; prompt sin cambios.",
            )

        if overlap_max > 0 and detail_looks_like_tag_soup(fragment):
            return PromptResult(
                prompt=base,
                negative_hint="",
                sampler_hints=hints,
                status=(
                    f"{backend} `{cfg.model}` devolvió tags en lugar de lenguaje natural; "
                    "no se añadió. Reintenta o umbral de descarte a 0."
                ),
                fragment="",
            )

        if detail_is_redundant(base, fragment, max_shared=overlap_max):
            shared = shared_content_count(base, fragment)
            return PromptResult(
                prompt=base,
                negative_hint="",
                sampler_hints=hints,
                status=(
                    f"{backend} `{cfg.model}`: detalle con {shared} palabras coincidentes "
                    f"(umbral {overlap_max}); no se añadió."
                ),
                fragment="",
            )

        if overlap_max > 0 and detail_looks_like_full_scene(fragment, word_max=wmax):
            return PromptResult(
                prompt=base,
                negative_hint="",
                sampler_hints=hints,
                status=(
                    f"{backend} `{cfg.model}` inventó una escena completa; "
                    "detalle descartado."
                ),
                fragment="",
            )

        n_words = count_words(fragment)
        range_note = ""
        if n_words < wmin:
            range_note = f" (pedido ≥{wmin}; el modelo escribió {n_words})"
        zone = append_detail("", fragment)
        label = "Krea 2" if stack.family == "krea2" else "Klein 9B"
        lang = normalize_language(request.language)
        lang_label = "español" if lang == LANG_ES else "English"
        anchor_note = f" · ancla=`{anchor}`" if anchor else " · sin ancla"
        return PromptResult(
            prompt=base,
            negative_hint=_negative_hint(stack.family, stack.variant),
            sampler_hints=hints,
            status=(
                f"Detalle {backend} {label} ({stack.variant}) · "
                f"`{cfg.model}` @ `{cfg.base_url}` · idioma={lang_label}{anchor_note} · "
                f"{n_words} palabras (rango {wmin}–{wmax}){range_note}. "
                "Prompt general sin cambios; texto en «Prompt de la zona»."
            ),
            fragment=zone,
        )


class NanVLProvider(OllamaVLProvider):
    """Mismo flujo que Ollama sobre la API OpenAI-compatible de NaN (nan.builders).

    Solo cambian el transporte (``/chat/completions`` + ``image_url``) y la
    config (base URL / key / timeout propios en Settings).
    """

    backend_label = "NaN"

    def _config(self, model: str | None) -> OllamaConfig:
        from forge_img2prompt.nan_settings import get_nan_config

        return get_nan_config(model=model)

    def _run(
        self,
        image,
        system: str,
        user_text: str,
        *,
        max_new_tokens: int,
        model: str | None = None,
    ) -> str:
        from forge_img2prompt.nan_client import chat_with_image as nan_chat

        cfg = self._config(model)
        return nan_chat(
            cfg,
            system=system,
            user_text=user_text,
            image=image,
            max_tokens=max_new_tokens,
        )

    def _status_note(self, cfg: OllamaConfig, choice: VlModelChoice) -> str:
        return (
            f"`{cfg.model}` @ `{cfg.base_url}` · NaN (clúster comunitario, "
            "UE, sin logs) · cero VRAM Forge."
        )

    def _error_hint(self, exc: Exception, cfg: OllamaConfig, choice: VlModelChoice) -> str:
        note = f"(`{cfg.model}` @ `{cfg.base_url}`): {exc}"
        if choice.risk == "premium":
            return f"{note}. `{choice.hf_id}` es tier premium de NaN."
        return note
