"""Image → Prompt tab for Forge Neo (Krea 2 / Klein 9B + Qwen3-VL)."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from PIL import Image

from modules import script_callbacks, scripts, shared

from forge_img2prompt.log import log
from forge_img2prompt.provider import DEFAULT_LANG, LANG_CHOICES, PromptRequest
from forge_img2prompt.stack import detect_stack, pick_text_encoder
from forge_img2prompt.vl_catalog import (
    choice_by_value,
    dropdown_choices,
    is_local_ready,
    list_vl_models,
    preferred_value,
)
from forge_img2prompt.vl_download import download_plan_markdown, format_download_plan, list_repo_files
from forge_img2prompt.vl_provider import CompositeProvider

EXT_DIR = scripts.basedir()
log(f"extensión cargada · basedir={EXT_DIR}")
_PROVIDER = CompositeProvider()
_CATALOG = list_vl_models()
log(f"catálogo VL: {len(_CATALOG)} modelos · preferido={_CATALOG[0].hf_id if _CATALOG else '?'}")


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
    dest = Path(choice.local_path)
    return download_plan_markdown(dest, repo_id=choice.hf_id)


def on_ui_tabs():
    log("registrando pestaña Image → Prompt")
    pairs = dropdown_choices(_CATALOG)
    default_vl = preferred_value(_CATALOG)

    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "## Image → Prompt (Krea 2 / Klein 9B)\n"
            "Caption con **Qwen3-VL** (transformers). En RTX 4060 8 GB elige "
            "**Huihui 2B abliterated** (uncensor, ~5 GB VRAM). El **4B** mejora "
            "calidad pero puede OOM. La **primera** Generate con imagen descarga "
            "el modelo seleccionado a `TextEncoders/<nombre>/`.\n\n"
            "Sin imagen → stub solo con **Notas**."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(
                    label="Imagen",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=320,
                )
                notes = gr.Textbox(
                    label="Notas / descripción (opcional)",
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
                    lang_dd = gr.Dropdown(
                        label="Idioma del prompt",
                        choices=list(LANG_CHOICES),
                        value=DEFAULT_LANG,
                        interactive=True,
                        scale=1,
                    )
                download_plan = gr.Markdown(value=_plan_for_value(default_vl))
                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
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

        def _generate_with_plan(
            img: Image.Image | None,
            notes_val: str,
            vl_value: str,
            language: str,
            progress=gr.Progress(track_tqdm=True),
        ):
            from huggingface_hub import hf_hub_download

            stack = _current_stack()
            choice = choice_by_value(vl_value, _CATALOG)
            assert choice is not None
            dest = Path(choice.local_path)
            dest.mkdir(parents=True, exist_ok=True)

            yield (
                "",
                "",
                f"**Stack:** `{stack.summary}`\n\nPreparando `{choice.hf_id}`…",
                download_plan_markdown(dest, repo_id=choice.hf_id),
            )

            if img is not None and not is_local_ready(dest):
                try:
                    items = list_repo_files(choice.hf_id)
                except Exception as exc:  # noqa: BLE001
                    yield (
                        "",
                        "",
                        f"No se pudo listar HF (`{choice.hf_id}`): {exc}",
                        download_plan_markdown(dest, repo_id=choice.hf_id),
                    )
                    return

                done: set[str] = set()
                for it in items:
                    target = dest / it.filename
                    if target.is_file() and target.stat().st_size > 0:
                        done.add(it.filename)

                yield (
                    "",
                    "",
                    f"**Stack:** `{stack.summary}`\n\nDescargando {len(items)} ficheros de `{choice.hf_id}`…",
                    format_download_plan(items, dest, repo_id=choice.hf_id, done=done),
                )

                total = max(len(items), 1)
                for it in items:
                    if it.filename in done:
                        continue
                    frac = len(done) / total
                    progress(frac * 0.7, desc=f"[{len(done)+1}/{total}] {it.filename} ({it.size_label})")
                    yield (
                        "",
                        "",
                        f"⬇️ `{it.filename}` ({it.size_label})",
                        format_download_plan(
                            items, dest, repo_id=choice.hf_id, done=done, current=it.filename
                        ),
                    )
                    try:
                        hf_hub_download(
                            repo_id=choice.hf_id,
                            filename=it.filename,
                            local_dir=str(dest),
                            local_dir_use_symlinks=False,
                            resume_download=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        yield (
                            "",
                            "",
                            f"Error descargando `{it.filename}`: {exc}",
                            format_download_plan(
                                items, dest, repo_id=choice.hf_id, done=done, current=it.filename
                            ),
                        )
                        return
                    done.add(it.filename)
                    yield (
                        "",
                        "",
                        f"✅ `{it.filename}`",
                        format_download_plan(items, dest, repo_id=choice.hf_id, done=done),
                    )

                if not is_local_ready(dest):
                    yield (
                        "",
                        "",
                        f"Descarga incompleta en `{dest}`",
                        format_download_plan(items, dest, repo_id=choice.hf_id, done=done),
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
                ),
                vl_value=choice.value,
                progress=on_prog_cap,
            )
            hints = result.sampler_hints
            if result.negative_hint:
                hints = f"{hints}\n{result.negative_hint}".strip()
            status = f"{stack.summary}\n{result.status}".strip()
            yield result.prompt, hints, status, download_plan_markdown(dest, repo_id=choice.hf_id)

        vl_dd.change(fn=_plan_for_value, inputs=[vl_dd], outputs=[download_plan])
        generate_btn.click(
            fn=_generate_with_plan,
            inputs=[image, notes, vl_dd, lang_dd],
            outputs=[prompt_out, hints_out, status_out, download_plan],
        )
        refresh_btn.click(fn=_refresh_stack, inputs=[], outputs=[status_out])
        ui.load(fn=_refresh_stack, inputs=[], outputs=[status_out])
        ui.load(fn=lambda: _plan_for_value(preferred_value(_CATALOG)), inputs=[], outputs=[download_plan])

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
log("callback on_ui_tabs registrado")
