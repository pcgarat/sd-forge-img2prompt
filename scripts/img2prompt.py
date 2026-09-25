"""Image → Prompt tab for Forge Neo (Krea 2 v1, stub provider)."""

from __future__ import annotations

import gradio as gr
from PIL import Image

from modules import script_callbacks, scripts, sd_models, shared

from forge_img2prompt.provider import PromptRequest, StubProvider
from forge_img2prompt.stack import detect_stack

EXT_DIR = scripts.basedir()
_PROVIDER = StubProvider()


def _checkpoint_name() -> str:
    try:
        info = sd_models.model_data.sd_model or getattr(shared, "sd_model", None)
        if info is not None and getattr(info, "sd_model_checkpoint", None):
            return str(info.sd_model_checkpoint)
    except Exception:
        pass
    opts = getattr(shared, "opts", None)
    if opts is not None:
        return str(getattr(opts, "sd_model_checkpoint", "") or "")
    return ""


def _text_encoder_name() -> str:
    opts = getattr(shared, "opts", None)
    if opts is None:
        return ""
    # Neo often exposes TE in the shared VAE / Text Encoder dropdown (`sd_vae`).
    for attr in ("sd_text_encoder", "sd_vae"):
        val = getattr(opts, attr, None)
        if val and str(val).strip() and str(val).strip().lower() not in ("automatic", "none"):
            return str(val)
    return ""


def _current_stack():
    return detect_stack(_checkpoint_name(), _text_encoder_name())


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
            "## Image → Prompt (Krea 2)\n"
            "v1 stub: usa **notas** + stack seleccionado. La imagen queda para un backend futuro."
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
        except Exception as exc:  # noqa: BLE001 — UI must load even if paste API differs
            gr.Markdown(f"Send-to no disponible (`infotext_utils`: {exc}). Usa el botón copiar del prompt.")

    return [(ui, "Image → Prompt", "img2prompt_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)
