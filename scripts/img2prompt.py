"""Image → Prompt tab for Forge Neo (Krea 2 / Klein 9B + Qwen3-VL)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from modules import script_callbacks, scripts, shared


def _reload_ext_package() -> None:
    """Forge reloads scripts/*.py on change but keeps forge_img2prompt cached."""
    for name in list(sys.modules):
        if name == "forge_img2prompt" or name.startswith("forge_img2prompt."):
            sys.modules.pop(name, None)


_reload_ext_package()

from forge_img2prompt.log import log
from forge_img2prompt import prompt_log
from forge_img2prompt.mask_crop import VL_OUTSIDE_RGB, crop_from_editor, editor_to_rgb
from forge_img2prompt.prefs import (
    dual_range_html,
    format_range,
    load_prefs,
    parse_range_text,
    save_prefs,
)
from forge_img2prompt.provider import (
    DEFAULT_LANG,
    DEFAULT_STRATEGY,
    DETAIL_WORDS_BOUNDS,
    DETAIL_WORDS_DEFAULT,
    GEN_WORDS_BOUNDS,
    GEN_WORDS_DEFAULT,
    LANG_CHOICES,
    OVERLAP_DISCARD_BOUNDS,
    DetailRequest,
    PromptRequest,
    clamp_overlap_discard,
    normalize_ref_slots,
    strategy_choices,
)
from forge_img2prompt.strategies import (
    PromptStrategy,
    blank_strategy_template,
    configure as configure_strategies,
    delete_strategy,
    get_strategy,
    reset_strategy,
    slugify_id,
    upsert_strategy,
)
from forge_img2prompt.stack import detect_stack, pick_text_encoder
from forge_img2prompt.ollama_client import ping_tags
from forge_img2prompt.ollama_settings import get_ollama_config, register_ollama_settings
from forge_img2prompt.vl_catalog import (
    choice_by_value,
    dropdown_choices,
    is_local_ready,
    list_vl_models,
    preferred_value,
)
from forge_img2prompt.vl_download import download_plan_markdown, iter_model_download
from forge_img2prompt.vl_provider import CompositeProvider

EXT_DIR = scripts.basedir()
configure_strategies(EXT_DIR)
log(f"extensión cargada · basedir={EXT_DIR}")
_PROVIDER = CompositeProvider()
_CATALOG = list_vl_models()
_PREFS = load_prefs(EXT_DIR)
log(f"catálogo VL: {len(_CATALOG)} modelos · preferido={_CATALOG[0].hf_id if _CATALOG else '?'}")
log(
    f"prefs UI: gen={_PREFS['gen_wmin']}-{_PREFS['gen_wmax']} "
    f"det={_PREFS['det_wmin']}-{_PREFS['det_wmax']} overlap={_PREFS['det_overlap']}"
)


def _refresh_catalog() -> list:
    """Reconsulta Ollama /api/tags y actualiza el catálogo en memoria."""
    global _CATALOG
    _CATALOG = list_vl_models()
    log(f"catálogo VL refrescado: {len(_CATALOG)} modelos")
    return _CATALOG


def _opt(name: str, default=None):
    opts = getattr(shared, "opts", None)
    if opts is None:
        return default
    return getattr(opts, name, default)


def _forge_preset() -> str:
    return str(_opt("forge_preset", "") or "").strip()


def _checkpoint_name() -> str:
    preset = _forge_preset()
    if preset:
        keyed = _opt(f"forge_checkpoint_{preset}", None)
        if keyed:
            return str(keyed)
    return str(_opt("sd_model_checkpoint", "") or "")


def _module_paths() -> list[str]:
    preset = _forge_preset()
    keyed = None
    if preset:
        keyed = _opt(f"forge_additional_modules_{preset}", None)
    if keyed is None:
        keyed = _opt("forge_additional_modules", None)
    if not keyed:
        return []
    if isinstance(keyed, str):
        return [keyed] if keyed.strip() else []
    return [str(x) for x in keyed if x]


def _text_encoder_name() -> str:
    return pick_text_encoder(_module_paths())


def _current_stack():
    return detect_stack(
        _checkpoint_name(),
        _text_encoder_name(),
        preset=_forge_preset(),
    )


def _refresh_stack():
    stack = _current_stack()
    flag = "soportado" if stack.is_supported else "no soportado / TE incompatible"
    return f"**Stack:** `{stack.summary}` · {flag}"


def _plan_for_value(vl_value: str) -> str:
    choice = choice_by_value(vl_value, _CATALOG)
    assert choice is not None
    if choice.is_ollama:
        cfg = get_ollama_config(model=choice.hf_id)
        ok, msg = ping_tags(cfg)
        mark = "OK" if ok else "aviso"
        key_state = "sí" if cfg.api_key else "no"
        cloud_hint = (
            "\n- Cloud: API key o `ollama signin` en el host; "
            "URL directa `https://ollama.com` si no usas proxy local.\n"
            if cfg.is_cloud
            else "\n"
        )
        return (
            f"### Backend Ollama ({mark})\n\n"
            f"- URL: `{cfg.base_url}`\n"
            f"- Modelo: `{choice.hf_id}`\n"
            f"- API key: {key_state}\n"
            f"- Timeout: {cfg.timeout:.0f}s"
            f"{cloud_hint}\n"
            f"{msg}\n\n"
            "Conexión en **Settings → Image → Prompt / Ollama** (URL, API key, timeout). "
            "El modelo se elige en este dropdown. "
            "Si Forge corre en Docker y falla la conexión, en el host: "
            "`OLLAMA_HOST=0.0.0.0:11434` y URL `http://172.17.0.1:11434`."
        )
    dest = Path(choice.local_path)
    return download_plan_markdown(dest, repo_id=choice.hf_id)


def _log_markdown() -> str:
    """Historial de bodies/params crudos al modelo (acordeón Log)."""
    return prompt_log.render_markdown()


def _ref_image_slot(label: str, elem_id: str):
    """Gradio Image slot; tolerates older kwargs."""
    kwargs: dict[str, Any] = {
        "label": label,
        "type": "pil",
        "height": 220,
        "elem_id": elem_id,
        "elem_classes": ["img2prompt-ref-slot"],
    }
    try:
        return gr.Image(**kwargs, sources=["upload", "clipboard"])
    except TypeError:
        kwargs.pop("elem_classes", None)
        try:
            return gr.Image(**kwargs, sources=["upload", "clipboard"])
        except TypeError:
            return gr.Image(**kwargs)


def _ref_image_row(prefix: str) -> tuple:
    return (
        _ref_image_slot("Foto 1", f"img2prompt_{prefix}_ref1"),
        _ref_image_slot("Foto 2", f"img2prompt_{prefix}_ref2"),
        _ref_image_slot("Foto 3", f"img2prompt_{prefix}_ref3"),
    )


def on_ui_tabs():
    log("registrando pestaña Image → Prompt")
    pairs = dropdown_choices(_CATALOG)
    prefs = load_prefs(EXT_DIR)
    saved_vl = str(prefs.get("vl_value") or "")
    valid_vl = {v for _, v in pairs}
    default_vl = saved_vl if saved_vl in valid_vl else preferred_value(_CATALOG)
    default_lang = prefs.get("language") or DEFAULT_LANG
    if default_lang not in {v for _, v in LANG_CHOICES}:
        default_lang = DEFAULT_LANG
    strat_pairs = strategy_choices()
    default_strategy = prefs.get("strategy") or DEFAULT_STRATEGY
    if default_strategy not in {v for _, v in strat_pairs}:
        default_strategy = DEFAULT_STRATEGY
    gen_lo, gen_hi = int(prefs["gen_wmin"]), int(prefs["gen_wmax"])
    det_lo, det_hi = int(prefs["det_wmin"]), int(prefs["det_wmax"])
    overlap0 = int(prefs["det_overlap"])

    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "## Image → Prompt (Krea 2 / Klein 9B)\n"
            "Caption con **Ollama** (modelos con visión del dropdown; "
            "sin pelear VRAM con Forge) o **Qwen3-VL** transformers en disco.\n\n"
            "Ollama: elige el modelo aquí; conexión en "
            "**Settings → Image → Prompt / Ollama** (URL, API key, timeout). "
            "Transformers: en 8 GB elige **Huihui 2B**; el **4B** puede OOM.\n\n"
            "Sin imagen → escribe en **Notas** el brief; el modelo VL/Ollama "
            "lo amplía a un prompt con detalles (refs foto 1–3 opcionales). "
            "Tras Generate, pinta una **máscara** (pincel magenta) sobre la zona "
            "y pulsa **Añadir detalle**: se analiza **solo lo pintado** "
            "(el resto se tapa en gris) y el texto va a **Prompt de la zona** "
            "(no se mezcla solo con el prompt general). Opcional: notas de zona.\n\n"
            "Referencias opcionales (**Foto 1–3**): adjunta 1–3 imágenes junto a las "
            "notas (general o de zona) y menciónalas ahí "
            "(p. ej. *el chico lleve el sombrero de la foto 1*). "
            "Requieren imagen principal + modelo VL."
        )
        with gr.Row():
            with gr.Column(scale=1):
                brush_kwargs: dict[str, Any] = {}
                try:
                    brush_kwargs["brush"] = gr.Brush(
                        colors=["#ff00aa"],
                        color_mode="fixed",
                        default_size=24,
                    )
                except (TypeError, AttributeError):
                    pass

                image_kwargs = dict(
                    label="Imagen + máscara (pinta la zona a detallar)",
                    type="pil",
                    sources=["upload", "clipboard"],
                    layers=False,
                    height=560,
                    elem_id="img2prompt_image_editor",
                    **brush_kwargs,
                )
                try:
                    image = gr.ImageEditor(
                        **image_kwargs,
                        canvas_size=(512, 512),
                        fixed_canvas=True,
                    )
                except TypeError:
                    try:
                        image = gr.ImageEditor(**image_kwargs, canvas_size=(512, 512))
                    except TypeError:
                        image = gr.ImageEditor(**image_kwargs)

                with gr.Accordion("General", open=True):
                    notes = gr.Textbox(
                        label="Notas / descripción (Generate; obligatorias sin imagen)",
                        lines=3,
                        placeholder="Con imagen: prioriza la chaqueta… / Sin imagen: "
                        "un astronauta en Marte al atardecer… "
                        "(el modelo añade detalle visual).",
                    )
                    gr.Markdown(
                        "Referencias opcionales (**Foto 1–3**): cítalas en **Notas** "
                        "como *foto 1*, *foto 2* o *foto 3* (según el slot)."
                    )
                    with gr.Row():
                        gen_ref1, gen_ref2, gen_ref3 = _ref_image_row("gen")

                with gr.Accordion("Zona", open=False):
                    crop_preview = gr.Image(
                        label="Crop de la máscara (lo que se reanaliza)",
                        type="pil",
                        height=140,
                        interactive=False,
                        elem_id="img2prompt_crop_preview",
                    )
                    zone_notes = gr.Textbox(
                        label="Notas de zona (solo para Añadir detalle)",
                        lines=2,
                        placeholder="Opcional: p. ej. barba / costura / ojos… "
                        "o «el sombrero de la foto 1». "
                        "No uses aquí las notas globales de la escena.",
                    )
                    zone_prompt = gr.Textbox(
                        label="Prompt de la zona (solo máscara; no se mezcla al general)",
                        lines=2,
                        interactive=True,
                        show_copy_button=True,
                        placeholder="Tras Añadir detalle aparecerá aquí el texto de la zona…",
                    )
                    gr.Markdown(
                        "Referencias de zona (**Foto 1–3**): cítalas en **Notas de zona** "
                        "como *foto 1*, *foto 2* o *foto 3* (según el slot)."
                    )
                    with gr.Row():
                        zone_ref1, zone_ref2, zone_ref3 = _ref_image_row("zone")

                with gr.Row():
                    vl_dd = gr.Dropdown(
                        label="Modelo VL",
                        choices=pairs,
                        value=default_vl,
                        interactive=True,
                        scale=3,
                    )
                    refresh_vl_btn = gr.Button(
                        "↻",
                        scale=0,
                        min_width=40,
                        elem_id="img2prompt_refresh_vl",
                    )
                    strategy_dd = gr.Dropdown(
                        label="Estrategia",
                        choices=list(strat_pairs),
                        value=default_strategy,
                        interactive=True,
                        scale=1,
                    )
                    refresh_strat_btn = gr.Button(
                        "↻",
                        scale=0,
                        min_width=40,
                        elem_id="img2prompt_refresh_strategy",
                    )
                    lang_dd = gr.Dropdown(
                        label="Idioma del prompt",
                        choices=list(LANG_CHOICES),
                        value=default_lang,
                        interactive=True,
                        scale=1,
                    )
                with gr.Accordion("Longitud del prompt (palabras)", open=True):
                    gr.Markdown(
                        "El VL intenta el rango; si se pasa del **máximo**, "
                        "se **recorta** al límite (status lo indica). "
                        "Por debajo del mínimo no se inventan palabras. "
                        "Los valores se **guardan solos** entre reinicios."
                    )
                    gr.HTML(
                        dual_range_html(
                            widget_id="img2prompt_gen_dual",
                            bridge_id="img2prompt_gen_range",
                            label="Generate",
                            lo=gen_lo,
                            hi=gen_hi,
                            minimum=GEN_WORDS_BOUNDS[0],
                            maximum=GEN_WORDS_BOUNDS[1],
                            step=5,
                        )
                    )
                    gen_range = gr.Textbox(
                        value=format_range(gen_lo, gen_hi),
                        elem_id="img2prompt_gen_range",
                        visible=False,
                        label="gen_range",
                    )
                    gr.HTML(
                        dual_range_html(
                            widget_id="img2prompt_det_dual",
                            bridge_id="img2prompt_det_range",
                            label="Detalle",
                            lo=det_lo,
                            hi=det_hi,
                            minimum=DETAIL_WORDS_BOUNDS[0],
                            maximum=DETAIL_WORDS_BOUNDS[1],
                            step=1,
                        )
                    )
                    det_range = gr.Textbox(
                        value=format_range(det_lo, det_hi),
                        elem_id="img2prompt_det_range",
                        visible=False,
                        label="det_range",
                    )
                    det_overlap = gr.Slider(
                        minimum=OVERLAP_DISCARD_BOUNDS[0],
                        maximum=OVERLAP_DISCARD_BOUNDS[1],
                        value=overlap0,
                        step=1,
                        label="Detalle · umbral descarte (palabras coincidentes)",
                        info=(
                            "Descarta si hay ≥ N palabras de contenido compartidas con el "
                            "prompt general (también tag-soup / escena completa). "
                            "0 = no descartar nunca el fragmento."
                        ),
                    )
                download_plan = gr.Markdown(value=_plan_for_value(default_vl))
                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
                    detail_btn = gr.Button("Añadir detalle")
                    refresh_btn = gr.Button("Refresh stack")
            with gr.Column(scale=1):
                prompt_out = gr.Textbox(
                    label="Prompt",
                    lines=10,
                    show_copy_button=True,
                    interactive=True,
                )
                hints_out = gr.Textbox(label="Hints (sampler / negativo)", lines=3, interactive=False)
                status_out = gr.Markdown()
                with gr.Row():
                    send_t2i = gr.Button("Send to txt2img")
                    send_i2i = gr.Button("Send to img2img")
                with gr.Accordion("Log (últimas peticiones)", open=False):
                    with gr.Row():
                        log_out = gr.Markdown(value=_log_markdown())
                    with gr.Row():
                        refresh_log_btn = gr.Button("↻", min_width=40)
                        clear_log_btn = gr.Button("Limpiar log")

        def _persist_prefs(
            gen_raw: str,
            det_raw: str,
            overlap: float,
            language: str,
            strategy: str,
            vl_value: str,
        ):
            g_lo, g_hi = parse_range_text(
                gen_raw, bounds=GEN_WORDS_BOUNDS, default=GEN_WORDS_DEFAULT
            )
            d_lo, d_hi = parse_range_text(
                det_raw, bounds=DETAIL_WORDS_BOUNDS, default=DETAIL_WORDS_DEFAULT
            )
            save_prefs(
                EXT_DIR,
                {
                    "gen_wmin": g_lo,
                    "gen_wmax": g_hi,
                    "det_wmin": d_lo,
                    "det_wmax": d_hi,
                    "det_overlap": clamp_overlap_discard(overlap),
                    "language": language or DEFAULT_LANG,
                    "strategy": strategy or DEFAULT_STRATEGY,
                    "vl_value": vl_value or "",
                },
            )
            return None

        def _generate_with_plan(
            editor: dict[str, Any] | Image.Image | None,
            notes_val: str,
            vl_value: str,
            strategy: str,
            language: str,
            gen_raw: str,
            ref1=None,
            ref2=None,
            ref3=None,
            progress=gr.Progress(track_tqdm=True),
        ):
            stack = _current_stack()
            choice = choice_by_value(vl_value, _CATALOG)
            assert choice is not None
            img = editor_to_rgb(editor)
            refs = normalize_ref_slots(ref1, ref2, ref3)
            wmin, wmax = parse_range_text(
                gen_raw, bounds=GEN_WORDS_BOUNDS, default=GEN_WORDS_DEFAULT
            )
            plan = _plan_for_value(choice.value)
            n_refs = sum(1 for r in refs if r is not None)
            refs_status = f" · {n_refs} ref(s)" if n_refs else ""

            yield (
                "",
                "",
                f"**Stack:** `{stack.summary}`\n\nPreparando `{choice.hf_id}` "
                f"(rango {wmin}–{wmax} palabras{refs_status})…",
                plan,
                _log_markdown(),
            )

            if img is not None and not choice.is_ollama:
                dest = Path(choice.local_path)
                dest.mkdir(parents=True, exist_ok=True)
                if not is_local_ready(dest):
                    try:
                        for event in iter_model_download(
                            repo_id=choice.hf_id,
                            local_dir=dest,
                            progress=lambda f, d: progress(f * 0.7, desc=d),
                        ):
                            yield (
                                "",
                                "",
                                f"**Stack:** `{stack.summary}`\n\n{event.status}",
                                event.plan,
                                _log_markdown(),
                            )
                    except Exception as exc:  # noqa: BLE001
                        yield (
                            "",
                            "",
                            f"Error descargando `{choice.hf_id}`: {exc}",
                            download_plan_markdown(dest, repo_id=choice.hf_id),
                            _log_markdown(),
                        )
                        return

                    if not is_local_ready(dest):
                        yield (
                            "",
                            "",
                            f"Descarga incompleta en `{dest}`",
                            download_plan_markdown(dest, repo_id=choice.hf_id),
                            _log_markdown(),
                        )
                        return

                    yield (
                        "",
                        "",
                        f"**Stack:** `{stack.summary}`\n\nDescarga completa. Caption…",
                        download_plan_markdown(dest, repo_id=choice.hf_id),
                        _log_markdown(),
                    )

            progress(0.72, desc="Caption VL…")

            def on_prog_cap(frac: float, desc: str) -> None:
                progress(0.72 + 0.28 * frac, desc=desc)

            result = _PROVIDER.generate(
                PromptRequest(
                    image=img,
                    user_notes=notes_val or "",
                    stack=stack,
                    language=language,
                    strategy=strategy or DEFAULT_STRATEGY,
                    word_min=wmin,
                    word_max=wmax,
                    ref_slots=refs,
                ),
                vl_value=choice.value,
                progress=on_prog_cap,
            )
            hints = result.sampler_hints
            if result.negative_hint:
                hints = f"{hints}\n{result.negative_hint}".strip()
            status = f"{stack.summary}\n{result.status}".strip()
            yield result.prompt, hints, status, _plan_for_value(choice.value), _log_markdown()

        def _preview_mask_crop(editor: dict[str, Any] | Image.Image | None):
            if not isinstance(editor, dict):
                return None
            crop = crop_from_editor(editor, blackout=True, outside=(0, 0, 0))
            if crop is None:
                log("preview máscara: sin paint/crop")
                return None
            log(f"preview máscara: crop {crop.size[0]}×{crop.size[1]} (blackout UI)")
            return crop

        def _detail_with_plan(
            editor: dict[str, Any] | Image.Image | None,
            zone_notes_val: str,
            vl_value: str,
            language: str,
            base_prompt: str,
            det_raw: str,
            overlap_discard: float,
            ref1=None,
            ref2=None,
            ref3=None,
            progress=gr.Progress(track_tqdm=True),
        ):
            stack = _current_stack()
            choice = choice_by_value(vl_value, _CATALOG)
            assert choice is not None
            base = (base_prompt or "").strip()
            plan = _plan_for_value(choice.value)
            wmin, wmax = parse_range_text(
                det_raw, bounds=DETAIL_WORDS_BOUNDS, default=DETAIL_WORDS_DEFAULT
            )
            overlap_max = clamp_overlap_discard(overlap_discard)
            refs = normalize_ref_slots(ref1, ref2, ref3)
            n_refs = sum(1 for r in refs if r is not None)
            refs_status = f", {n_refs} ref(s)" if n_refs else ""

            if not base:
                yield (
                    base_prompt or "",
                    "",
                    f"**Stack:** `{stack.summary}`\n\n"
                    "Necesitas un prompt previo (Generate) antes de añadir detalle.",
                    plan,
                    None,
                    "",
                    _log_markdown(),
                )
                return

            # VL: gray-out outside brush so surrounding subjects do not leak
            crop = crop_from_editor(
                editor if isinstance(editor, dict) else None,
                blackout=True,
                outside=VL_OUTSIDE_RGB,
            )
            if crop is None:
                log("detalle: crop=None (máscara vacía o editor sin layers/diff)")
                yield (
                    base,
                    "",
                    f"**Stack:** `{stack.summary}`\n\n"
                    "No se detectó máscara. Usa el **pincel magenta**, pinta la zona "
                    "y espera a ver el crop debajo antes de Añadir detalle.",
                    plan,
                    None,
                    "",
                    _log_markdown(),
                )
                return

            preview = crop_from_editor(
                editor if isinstance(editor, dict) else None,
                blackout=True,
                outside=(0, 0, 0),
            )
            preview_ui = preview if preview is not None else crop
            log(
                f"detalle: VL bbox {crop.size[0]}×{crop.size[1]} "
                f"(máscara gris; rango {wmin}–{wmax} palabras; refs={n_refs})"
            )
            yield (
                base,
                "",
                f"**Stack:** `{stack.summary}`\n\nPreparando detalle `{choice.hf_id}` "
                f"(crop {crop.size[0]}×{crop.size[1]}, {wmin}–{wmax} palabras"
                f"{refs_status})…",
                plan,
                preview_ui,
                "",
                _log_markdown(),
            )

            if not choice.is_ollama:
                dest = Path(choice.local_path)
                dest.mkdir(parents=True, exist_ok=True)
                if not is_local_ready(dest):
                    try:
                        for event in iter_model_download(
                            repo_id=choice.hf_id,
                            local_dir=dest,
                            progress=lambda f, d: progress(f * 0.7, desc=d),
                        ):
                            yield (
                                base,
                                "",
                                f"**Stack:** `{stack.summary}`\n\n{event.status}",
                                event.plan,
                                preview_ui,
                                "",
                                _log_markdown(),
                            )
                    except Exception as exc:  # noqa: BLE001
                        yield (
                            base,
                            "",
                            f"Error descargando `{choice.hf_id}`: {exc}",
                            download_plan_markdown(dest, repo_id=choice.hf_id),
                            preview_ui,
                            "",
                            _log_markdown(),
                        )
                        return

                    if not is_local_ready(dest):
                        yield (
                            base,
                            "",
                            f"Descarga incompleta en `{dest}`",
                            download_plan_markdown(dest, repo_id=choice.hf_id),
                            preview_ui,
                            "",
                            _log_markdown(),
                        )
                        return

            progress(0.72, desc="Detalle VL…")

            def on_prog(frac: float, desc: str) -> None:
                progress(0.72 + 0.28 * frac, desc=desc)

            # Never reuse global Generate notes: they make the VL retell the scene.
            result = _PROVIDER.detail(
                DetailRequest(
                    crop=crop,
                    base_prompt=base,
                    stack=stack,
                    language=language,
                    user_notes=zone_notes_val or "",
                    word_min=wmin,
                    word_max=wmax,
                    overlap_discard=overlap_max,
                    ref_slots=refs,
                ),
                vl_value=choice.value,
                progress=on_prog,
            )
            hints = result.sampler_hints
            if result.negative_hint:
                hints = f"{hints}\n{result.negative_hint}".strip()
            status = f"{stack.summary}\n{result.status}".strip()
            # Prompt general intacto; el detalle solo alimenta zone_prompt.
            zone = getattr(result, "fragment", "") or ""
            log(f"detalle: fragmento={len(zone)} chars · prompt general sin cambios")
            yield (
                base,
                hints,
                status,
                _plan_for_value(choice.value),
                preview_ui,
                zone,
                _log_markdown(),
            )

        vl_dd.change(fn=_plan_for_value, inputs=[vl_dd], outputs=[download_plan])

        def _refresh_vl_dropdown(current: str):
            cat = _refresh_catalog()
            pairs_new = dropdown_choices(cat)
            valid = {v for _, v in pairs_new}
            value = current if current in valid else preferred_value(cat)
            return gr.update(choices=pairs_new, value=value), _plan_for_value(value)

        refresh_vl_btn.click(
            fn=_refresh_vl_dropdown,
            inputs=[vl_dd],
            outputs=[vl_dd, download_plan],
        )

        def _refresh_strategy_dropdown(current: str):
            pairs_new = strategy_choices()
            valid = {v for _, v in pairs_new}
            value = current if current in valid else DEFAULT_STRATEGY
            return gr.update(choices=pairs_new, value=value)

        refresh_strat_btn.click(
            fn=_refresh_strategy_dropdown,
            inputs=[strategy_dd],
            outputs=[strategy_dd],
        )
        ui.load(
            fn=_refresh_strategy_dropdown,
            inputs=[strategy_dd],
            outputs=[strategy_dd],
        )
        image.change(
            fn=_preview_mask_crop,
            inputs=[image],
            outputs=[crop_preview],
        )
        for src in (gen_range, det_range, det_overlap, lang_dd, strategy_dd, vl_dd):
            src.change(
                fn=_persist_prefs,
                inputs=[
                    gen_range,
                    det_range,
                    det_overlap,
                    lang_dd,
                    strategy_dd,
                    vl_dd,
                ],
            )
        # Forge Neo = Gradio 4.x → parámetro `js` (no `_js`).
        # El preprocesador fuerza el valor del dual-range al payload del click.
        _JS_SYNC_GEN = """
(img, notes, vl, strategy, lang, gen_raw, r1, r2, r3) => {
  if (window.img2promptSyncRanges) window.img2promptSyncRanges();
  const root = document.getElementById("img2prompt_gen_range");
  const el = root && root.querySelector("textarea, input");
  return [img, notes, vl, strategy, lang, el ? el.value : gen_raw, r1, r2, r3];
}
""".strip()
        _JS_SYNC_DET = """
(img, zone_notes, vl, lang, prompt, det_raw, overlap, r1, r2, r3) => {
  if (window.img2promptSyncRanges) window.img2promptSyncRanges();
  const root = document.getElementById("img2prompt_det_range");
  const el = root && root.querySelector("textarea, input");
  return [img, zone_notes, vl, lang, prompt, el ? el.value : det_raw, overlap, r1, r2, r3];
}
""".strip()
        generate_btn.click(
            fn=_generate_with_plan,
            inputs=[
                image,
                notes,
                vl_dd,
                strategy_dd,
                lang_dd,
                gen_range,
                gen_ref1,
                gen_ref2,
                gen_ref3,
            ],
            outputs=[
                prompt_out,
                hints_out,
                status_out,
                download_plan,
                log_out,
            ],
            js=_JS_SYNC_GEN,
        )
        detail_btn.click(
            fn=_detail_with_plan,
            inputs=[
                image,
                zone_notes,
                vl_dd,
                lang_dd,
                prompt_out,
                det_range,
                det_overlap,
                zone_ref1,
                zone_ref2,
                zone_ref3,
            ],
            outputs=[
                prompt_out,
                hints_out,
                status_out,
                download_plan,
                crop_preview,
                zone_prompt,
                log_out,
            ],
            js=_JS_SYNC_DET,
        )
        refresh_log_btn.click(
            fn=_log_markdown,
            inputs=[],
            outputs=[log_out],
        )

        def _clear_log():
            prompt_log.clear()
            return _log_markdown()

        clear_log_btn.click(fn=_clear_log, inputs=[], outputs=[log_out])
        refresh_btn.click(fn=_refresh_stack, inputs=[], outputs=[status_out])
        ui.load(fn=_refresh_stack, inputs=[], outputs=[status_out])
        ui.load(fn=lambda: _plan_for_value(default_vl), inputs=[], outputs=[download_plan])
        ui.load(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { if (window.img2promptInitRanges) window.img2promptInitRanges(); }",
        )

        try:
            from modules import infotext_utils as send

            send.register_paste_params_button(
                send.ParamBinding(
                    paste_button=send_t2i,
                    tabname="txt2img",
                    source_text_component=prompt_out,
                )
            )
            send.register_paste_params_button(
                send.ParamBinding(
                    paste_button=send_i2i,
                    tabname="img2img",
                    source_text_component=prompt_out,
                )
            )
        except Exception as exc:  # noqa: BLE001
            log(f"Send-to no disponible (infotext_utils): {exc}")
            gr.Markdown(f"Send-to no disponible (`infotext_utils`: {exc}). Usa el botón copiar del prompt.")

    log("pestaña Image → Prompt lista")

    with gr.Blocks(analytics_enabled=False) as strategies_ui:
        gr.Markdown(
            "## Estrategias de prompt\n"
            "Edita las estrategias integradas (**Prosa**, **Dentro → fuera**, "
            "**Capas (sujeto→calidad)**) o crea otras. Los textos se usan como "
            "system prompt / hint en **Generate** (Image → Prompt). "
            "Persisten en `strategies.json`.\n\n"
            "Tras guardar o crear, pulsa **↻** junto a Estrategia en Image → Prompt "
            "si el dropdown no se actualiza solo."
        )
        with gr.Row():
            strat_pick = gr.Dropdown(
                label="Estrategia a editar",
                choices=list(strat_pairs),
                value=default_strategy,
                interactive=True,
                scale=3,
            )
            new_strat_btn = gr.Button("Nueva", scale=1)
            refresh_editor_btn = gr.Button("↻ lista", scale=0, min_width=80)
        strat_id = gr.Textbox(
            label="ID (slug)",
            value=default_strategy,
            interactive=False,
            info="Solo editable al crear. Letras minúsculas, números y `_`.",
        )
        strat_label = gr.Textbox(
            label="Nombre",
            value=get_strategy(default_strategy).label,
        )
        strat_caption = gr.Textbox(
            label="System prompt · caption (con imagen)",
            lines=10,
            value=get_strategy(default_strategy).caption_system,
        )
        strat_notes = gr.Textbox(
            label="System prompt · notas → prompt (sin imagen)",
            lines=8,
            value=get_strategy(default_strategy).notes_system,
        )
        strat_hint = gr.Textbox(
            label="Hint en mensaje de usuario",
            lines=3,
            value=get_strategy(default_strategy).user_hint,
        )
        with gr.Row():
            save_strat_btn = gr.Button("Guardar", variant="primary")
            reset_strat_btn = gr.Button("Restaurar defaults")
            delete_strat_btn = gr.Button("Eliminar", variant="stop")
        strat_status = gr.Markdown()
        creating_flag = gr.State(False)

        def _editor_outputs(strat: PromptStrategy, *, creating: bool, status: str = ""):
            pairs_new = strategy_choices()
            pick_update = (
                gr.update(choices=pairs_new)
                if creating
                else gr.update(choices=pairs_new, value=strat.id)
            )
            return (
                pick_update,
                gr.update(value=strat.id, interactive=creating),
                strat.label,
                strat.caption_system,
                strat.notes_system,
                strat.user_hint,
                creating,
                status,
            )

        def _load_picked(strategy_id: str):
            strat = get_strategy(strategy_id or DEFAULT_STRATEGY)
            return _editor_outputs(strat, creating=False)

        def _start_new():
            blank = blank_strategy_template()
            return _editor_outputs(
                blank,
                creating=True,
                status="Nueva estrategia: pon un **Nombre** (y opcionalmente un ID) y pulsa Guardar.",
            )

        def _refresh_editor_list(current_id: str, creating: bool):
            pairs_new = strategy_choices()
            valid = {v for _, v in pairs_new}
            if creating:
                return gr.update(choices=pairs_new)
            value = current_id if current_id in valid else DEFAULT_STRATEGY
            return gr.update(choices=pairs_new, value=value)

        def _save(
            strategy_id: str,
            label: str,
            caption: str,
            notes: str,
            hint: str,
            creating: bool,
        ):
            sid = slugify_id(strategy_id) or slugify_id(label)
            strat, msg = upsert_strategy(
                strategy_id=sid,
                label=label,
                caption_system=caption,
                notes_system=notes,
                user_hint=hint,
                as_new=bool(creating),
            )
            if strat is None:
                draft = PromptStrategy(
                    id=sid,
                    label=label or "",
                    caption_system=caption or "",
                    notes_system=notes or "",
                    user_hint=hint or "",
                    builtin=False,
                )
                return _editor_outputs(
                    draft, creating=bool(creating), status=f"**Error:** {msg}"
                )
            return _editor_outputs(strat, creating=False, status=f"**OK:** {msg}")

        def _reset(strategy_id: str):
            strat, msg = reset_strategy(strategy_id or DEFAULT_STRATEGY)
            if strat is None:
                cur = get_strategy(strategy_id or DEFAULT_STRATEGY)
                return _editor_outputs(cur, creating=False, status=f"**Error:** {msg}")
            return _editor_outputs(strat, creating=False, status=f"**OK:** {msg}")

        def _delete(strategy_id: str):
            ok, msg = delete_strategy(strategy_id or "")
            fallback = get_strategy(DEFAULT_STRATEGY)
            if not ok:
                cur = get_strategy(strategy_id or DEFAULT_STRATEGY)
                return _editor_outputs(cur, creating=False, status=f"**Error:** {msg}")
            return _editor_outputs(fallback, creating=False, status=f"**OK:** {msg}")

        _editor_outs = [
            strat_pick,
            strat_id,
            strat_label,
            strat_caption,
            strat_notes,
            strat_hint,
            creating_flag,
            strat_status,
        ]
        strat_pick.change(
            fn=_load_picked,
            inputs=[strat_pick],
            outputs=_editor_outs,
        )
        new_strat_btn.click(fn=_start_new, inputs=[], outputs=_editor_outs)
        refresh_editor_btn.click(
            fn=_refresh_editor_list,
            inputs=[strat_id, creating_flag],
            outputs=[strat_pick],
        )
        save_strat_btn.click(
            fn=_save,
            inputs=[
                strat_id,
                strat_label,
                strat_caption,
                strat_notes,
                strat_hint,
                creating_flag,
            ],
            outputs=_editor_outs,
        )
        reset_strat_btn.click(
            fn=_reset,
            inputs=[strat_id],
            outputs=_editor_outs,
        )
        delete_strat_btn.click(
            fn=_delete,
            inputs=[strat_id],
            outputs=_editor_outs,
        )
        strategies_ui.load(
            fn=_load_picked,
            inputs=[strat_pick],
            outputs=_editor_outs,
        )

    log("pestaña Estrategias lista")
    return [
        (ui, "Image → Prompt", "img2prompt_tab"),
        (strategies_ui, "Estrategias", "img2prompt_strategies_tab"),
    ]


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(register_ollama_settings)
log("callback on_ui_tabs + on_ui_settings (Ollama) registrados")
