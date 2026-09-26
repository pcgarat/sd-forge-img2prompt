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
    DETAIL_WORDS_BOUNDS,
    DETAIL_WORDS_DEFAULT,
    GEN_WORDS_BOUNDS,
    GEN_WORDS_DEFAULT,
    LANG_CHOICES,
    OVERLAP_DISCARD_BOUNDS,
    DetailRequest,
    PromptRequest,
    clamp_overlap_discard,
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
            "Sin imagen → stub solo con **Notas**. "
            "Tras Generate, pinta una **máscara** (pincel magenta) sobre la zona "
            "y pulsa **Añadir detalle**: se analiza **solo lo pintado** "
            "(el resto se tapa en gris) y el texto va a **Prompt de la zona** "
            "(no se mezcla solo con el prompt general). Opcional: notas de zona."
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

                with gr.Row():
                    crop_preview = gr.Image(
                        label="Crop de la máscara (lo que se reanaliza)",
                        type="pil",
                        height=140,
                        interactive=False,
                        elem_id="img2prompt_crop_preview",
                    )
                zone_prompt = gr.Textbox(
                    label="Prompt de la zona (solo máscara; no se mezcla al general)",
                    lines=2,
                    interactive=True,
                    show_copy_button=True,
                    placeholder="Tras Añadir detalle aparecerá aquí el texto de la zona…",
                )
                zone_notes = gr.Textbox(
                    label="Notas de zona (solo para Añadir detalle)",
                    lines=2,
                    placeholder="Opcional: p. ej. barba / costura / ojos… "
                    "No uses aquí las notas globales de la escena.",
                )

                notes = gr.Textbox(
                    label="Notas / descripción (opcional, solo Generate)",
                    lines=3,
                    placeholder="Ej: prioriza la chaqueta roja; tono noir…",
                )
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

        def _persist_prefs(
            gen_raw: str,
            det_raw: str,
            overlap: float,
            language: str,
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
                    "vl_value": vl_value or "",
                },
            )
            return None

        def _generate_with_plan(
            editor: dict[str, Any] | Image.Image | None,
            notes_val: str,
            vl_value: str,
            language: str,
            gen_raw: str,
            progress=gr.Progress(track_tqdm=True),
        ):
            stack = _current_stack()
            choice = choice_by_value(vl_value, _CATALOG)
            assert choice is not None
            img = editor_to_rgb(editor)
            wmin, wmax = parse_range_text(
                gen_raw, bounds=GEN_WORDS_BOUNDS, default=GEN_WORDS_DEFAULT
            )
            plan = _plan_for_value(choice.value)

            yield (
                "",
                "",
                f"**Stack:** `{stack.summary}`\n\nPreparando `{choice.hf_id}` "
                f"(rango {wmin}–{wmax} palabras)…",
                plan,
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
                            )
                    except Exception as exc:  # noqa: BLE001
                        yield (
                            "",
                            "",
                            f"Error descargando `{choice.hf_id}`: {exc}",
                            download_plan_markdown(dest, repo_id=choice.hf_id),
                        )
                        return

                    if not is_local_ready(dest):
                        yield (
                            "",
                            "",
                            f"Descarga incompleta en `{dest}`",
                            download_plan_markdown(dest, repo_id=choice.hf_id),
                        )
                        return

                    yield (
                        "",
                        "",
                        f"**Stack:** `{stack.summary}`\n\nDescarga completa. Caption…",
                        download_plan_markdown(dest, repo_id=choice.hf_id),
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
                    word_min=wmin,
                    word_max=wmax,
                ),
                vl_value=choice.value,
                progress=on_prog_cap,
            )
            hints = result.sampler_hints
            if result.negative_hint:
                hints = f"{hints}\n{result.negative_hint}".strip()
            status = f"{stack.summary}\n{result.status}".strip()
            yield result.prompt, hints, status, _plan_for_value(choice.value)

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

            if not base:
                yield (
                    base_prompt or "",
                    "",
                    f"**Stack:** `{stack.summary}`\n\n"
                    "Necesitas un prompt previo (Generate) antes de añadir detalle.",
                    plan,
                    None,
                    "",
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
                f"(máscara gris; rango {wmin}–{wmax} palabras)"
            )
            yield (
                base,
                "",
                f"**Stack:** `{stack.summary}`\n\nPreparando detalle `{choice.hf_id}` "
                f"(crop {crop.size[0]}×{crop.size[1]}, {wmin}–{wmax} palabras)…",
                plan,
                preview_ui,
                "",
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
                            )
                    except Exception as exc:  # noqa: BLE001
                        yield (
                            base,
                            "",
                            f"Error descargando `{choice.hf_id}`: {exc}",
                            download_plan_markdown(dest, repo_id=choice.hf_id),
                            preview_ui,
                            "",
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
        image.change(
            fn=_preview_mask_crop,
            inputs=[image],
            outputs=[crop_preview],
        )
        for src in (gen_range, det_range, det_overlap, lang_dd, vl_dd):
            src.change(
                fn=_persist_prefs,
                inputs=[gen_range, det_range, det_overlap, lang_dd, vl_dd],
            )
        # Forge Neo = Gradio 4.x → parámetro `js` (no `_js`).
        # El preprocesador fuerza el valor del dual-range al payload del click.
        _JS_SYNC_GEN = """
(img, notes, vl, lang, gen_raw) => {
  if (window.img2promptSyncRanges) window.img2promptSyncRanges();
  const root = document.getElementById("img2prompt_gen_range");
  const el = root && root.querySelector("textarea, input");
  return [img, notes, vl, lang, el ? el.value : gen_raw];
}
""".strip()
        _JS_SYNC_DET = """
(img, zone_notes, vl, lang, prompt, det_raw, overlap) => {
  if (window.img2promptSyncRanges) window.img2promptSyncRanges();
  const root = document.getElementById("img2prompt_det_range");
  const el = root && root.querySelector("textarea, input");
  return [img, zone_notes, vl, lang, prompt, el ? el.value : det_raw, overlap];
}
""".strip()
        generate_btn.click(
            fn=_generate_with_plan,
            inputs=[image, notes, vl_dd, lang_dd, gen_range],
            outputs=[prompt_out, hints_out, status_out, download_plan],
            js=_JS_SYNC_GEN,
        )
        detail_btn.click(
            fn=_detail_with_plan,
            inputs=[image, zone_notes, vl_dd, lang_dd, prompt_out, det_range, det_overlap],
            outputs=[prompt_out, hints_out, status_out, download_plan, crop_preview, zone_prompt],
            js=_JS_SYNC_DET,
        )
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
    return [(ui, "Image → Prompt", "img2prompt_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(register_ollama_settings)
log("callback on_ui_tabs + on_ui_settings (Ollama) registrados")
