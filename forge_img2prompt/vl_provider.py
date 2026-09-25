from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

from forge_img2prompt.log import log
from forge_img2prompt.provider import PromptRequest, PromptResult, StubProvider, _negative_hint, _sampler_hints
from forge_img2prompt.vl_catalog import VlModelChoice, choice_by_value, default_local_dir, is_local_ready
from forge_img2prompt.vl_download import download_plan_markdown, ensure_model_downloaded, list_repo_files

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


def _build_user_text(notes: str, family: str) -> str:
    parts = [
        "Describe this image as a ready-to-paste generation prompt.",
        _family_hint(family),
    ]
    notes = (notes or "").strip()
    if notes:
        parts.append(f"User notes to respect or weave in: {notes}")
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

    def _caption(self, image: Image.Image, notes: str, family: str, *, uncensored: bool = False) -> str:
        assert self._model is not None and self._processor is not None
        if image.mode != "RGB":
            image = image.convert("RGB")

        system = _CAPTION_SYSTEM
        if uncensored:
            system = f"{_CAPTION_SYSTEM} {_CAPTION_UNCENSORED_EXTRA}"

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _build_user_text(notes, family)},
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

        log(f"generando caption (max_new_tokens=320) en {model_device}…")
        with torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=320, do_sample=False)

        in_ids = inputs["input_ids"]
        trimmed = [out[len(inp) :] for inp, out in zip(in_ids, generated)]
        text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        caption = " ".join(text.split()).strip()
        log(f"caption listo · {len(caption)} chars")
        return caption

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
        prompt = ""
        try:
            if not is_local_ready(local):
                report(0.0, "Preparando descarga del modelo VL…")
                items = list_repo_files(choice.hf_id)
                report(0.05, f"Descarga: {len(items)} ficheros · `{choice.hf_id}`")
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
                uncensored=choice.is_uncensored,
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

        label = "Krea 2" if stack.family == "krea2" else "Klein 9B"
        return PromptResult(
            prompt=prompt,
            negative_hint=_negative_hint(stack.family, stack.variant),
            sampler_hints=hints,
            status=(
                f"VL {label} ({stack.variant}) · `{choice.hf_id}` "
                f"desde `{local.name}`. Checkpoint Forge liberado durante el caption."
            ),
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


def initial_status_markdown() -> str:
    from forge_img2prompt.vl_catalog import preferred_choice

    choice = preferred_choice()
    local = Path(choice.local_path)
    if is_local_ready(local):
        return f"Modelo **{choice.hf_id}** listo en `{local}`."
    return download_plan_markdown(local, repo_id=choice.hf_id)
