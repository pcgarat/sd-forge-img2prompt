"""Image → Prompt tab for Forge Neo (Krea 2 / Klein 9B, stub provider)."""

from __future__ import annotations

import gradio as gr
from PIL import Image

from modules import script_callbacks, scripts, shared

from forge_img2prompt.provider import PromptRequest, StubProvider
from forge_img2prompt.stack import detect_stack, pick_text_encoder

EXT_DIR = scripts.basedir()
_PROVIDER = StubProvider()


def _opt(name: str, default=None):
    opts = getattr(shared, "opts", None)
    if opts is None:
        return default
    return getattr(opts, name, default)


def _forge_preset() -> str:
    return str(_opt("forge_preset", "") or "").strip()


def _checkpoint_name() -> str:
    """Prefer the checkpoint bound to the active Forge UI preset (what the dropdown shows)."""
    preset = _forge_preset()
    if preset:
        keyed = _opt(f"forge_checkpoint_{preset}", None)
        if keyed:
            return str(keyed)
    return str(_opt("sd_model_checkpoint", "") or "")


def _module_paths() -> list[str]:
    preset = _forge_preset()
    if preset:
        keyed = _opt(f"forge_additional_modules_{preset}", None)
        if keyed:
            return [str(x) for x in keyed]
    modules = _opt("forge_additional_modules", None) or []
    return [str(x) for x in modules]


def _text_encoder_name() -> str:
    return pick_text_encoder(_module_paths())


def _current_stack():
    return detect_stack(
        _checkpoint_name(),
        _text_encoder_name(),
        preset=_forge_preset(),
    )


def _generate(image: Image.Image | None, notes: str):
    stack = _current_stack()
    result = _PROVIDER.generate(
        PromptRequest(image=image, user_notes=notes or "", stack=stack)
    )
    hints = result.sampler_hints
    if result.negative_hint:
        hints = f"{hints}\n{result.negative_hint}".strip()
    status = f"{stack.summary}\n{result.status}".strip()
    return result.prompt, hints, status


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "## Image → Prompt (Krea 2 / Klein 9B)\n"
            "Lee el **UI Preset** activo de Forge Neo (`forge_checkpoint_<preset>`), "
            "no solo `sd_model_checkpoint` (puede quedar desfasado).\n\n"
            "v1 stub: usa **notas** + stack. La imagen queda para un backend futuro."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(
                    label="Imagen",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=360,
                )
                notes = gr.Textbox(
                    label="Notas / descripción",
                    lines=4,
                    placeholder="Sujeto, entorno, luz, estilo… (prosa o frases cortas)",
                )
                generate_btn = gr.Button("Generate", variant="primary")
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

        generate_btn.click(
            fn=_generate,
            inputs=[image, notes],
            outputs=[prompt_out, hints_out, status_out],
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
            gr.Markdown(f"Send-to no disponible (`infotext_utils`: {exc}). Usa el botón copiar del prompt.")

    return [(ui, "Image → Prompt", "img2prompt_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)
