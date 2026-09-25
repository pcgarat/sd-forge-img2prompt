from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Último Qwen3-VL Instruct que cabe cómodo en ~8 GB VRAM (BF16 ≈ 4.8 GB).
DEFAULT_HF_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_LOCAL_NAME = "Qwen3-VL-2B-Instruct"
DEFAULT_LABEL = "Qwen3-VL-2B-Instruct (HF → TextEncoders, ~5 GB VRAM)"


@dataclass(frozen=True)
class VlModelChoice:
    label: str
    local_path: str
    hf_id: str
    kind: str
    caption_ready: bool
    mmproj_path: str = ""
    blocked_reason: str = ""

    @property
    def value(self) -> str:
        return self.local_path or f"hf:{self.hf_id}"


def text_encoder_root() -> Path:
    """Carpeta TextEncoders de Forge Neo / docker-neo."""
    candidates: list[Path] = []
    try:
        from modules import shared

        data = getattr(getattr(shared, "cmd_opts", None), "data_dir", None)
        models = None
        try:
            from modules import paths

            models = getattr(paths, "models_path", None)
        except Exception:
            models = None
        if models:
            candidates.append(Path(models) / "TextEncoders")
            candidates.append(Path(models) / "text_encoder")
        if data:
            candidates.append(Path(data) / "Models" / "TextEncoders")
            candidates.append(Path(data) / "models" / "text_encoder")
    except Exception:
        pass

    for p in (
        "/data/Models/TextEncoders",
        "/home/pacogarat/Applications/Data/Models/TextEncoders",
    ):
        candidates.append(Path(p))

    for c in candidates:
        if c.is_dir():
            return c
    # Crear layout docker-neo por defecto si hace falta
    fallback = Path("/data/Models/TextEncoders")
    if fallback.parent.is_dir():
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    host = Path("/home/pacogarat/Applications/Data/Models/TextEncoders")
    host.mkdir(parents=True, exist_ok=True)
    return host


def default_local_dir() -> Path:
    return text_encoder_root() / DEFAULT_LOCAL_NAME


def is_local_ready(local_dir: Path | None = None) -> bool:
    root = local_dir or default_local_dir()
    if not root.is_dir():
        return False
    has_config = (root / "config.json").is_file()
    weights = list(root.glob("*.safetensors")) + list(root.glob("model*.safetensors"))
    # shards: model-00001-of-00002.safetensors etc.
    if not weights:
        weights = list(root.glob("**/*.safetensors"))
    return has_config and len(weights) > 0


def sole_model_choice() -> VlModelChoice:
    local = default_local_dir()
    ready = is_local_ready(local)
    tag = "listo en disco" if ready else "se descargará la 1ª vez"
    return VlModelChoice(
        label=f"{DEFAULT_LABEL} · {tag}",
        local_path=str(local),
        hf_id=DEFAULT_HF_ID,
        kind="hf",
        caption_ready=True,
    )


def list_vl_models(extra_dirs=None) -> list[VlModelChoice]:
    """Solo el modelo por defecto (Qwen3-VL-2B-Instruct)."""
    del extra_dirs  # API estable; ignorado a propósito
    return [sole_model_choice()]


def dropdown_choices(catalog: list[VlModelChoice] | None = None) -> list[tuple[str, str]]:
    catalog = catalog if catalog is not None else list_vl_models()
    return [(c.label, c.value) for c in catalog]


def preferred_value(catalog: list[VlModelChoice] | None = None) -> str:
    catalog = catalog if catalog is not None else list_vl_models()
    return catalog[0].value


def choice_by_value(value: str, catalog: list[VlModelChoice] | None = None) -> VlModelChoice | None:
    catalog = catalog if catalog is not None else list_vl_models()
    for c in catalog:
        if c.value == value or c.hf_id == value or value in (c.local_path, f"hf:{c.hf_id}"):
            return c
    # Siempre devolvemos el único modelo soportado
    return catalog[0] if catalog else sole_model_choice()
