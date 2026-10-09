from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from forge_img2prompt.nan_settings import (
    DEFAULT_NAN_BASE_URL,
    DEFAULT_NAN_MODEL,
    NAN_VALUE,
    get_nan_config,
    is_nan_value,
    nan_vision_model_ids,
    parse_nan_value,
    vision_label,
    vision_risk,
)
from forge_img2prompt.ollama_settings import (
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_VALUE,
    default_base_url,
    get_ollama_config,
    is_cloud_model,
    is_ollama_value,
    parse_ollama_value,
)

# RTX 4060 8 GB: tras liberar el checkpoint Forge, ~5 GB libres para VL.
# - 2B BF16 ≈ 4.3 GB en disco / ~5 GB VRAM → cabe cómodo.
# - 4B BF16 ≈ 8.9 GB en disco / ≥9 GB VRAM → justo o OOM; se ofrece avisado.
# - 8B BF16 ≈ 17 GB → fuera de juego sin cuantización GGUF (otro backend).
# - Ollama VL (descubrimiento /api/tags + vision) fuera del proceso Forge.
# - FP8 Huihui/Qwen: vLLM/SGLang; Transformers aún no carga esos pesos.


@dataclass(frozen=True)
class VlModelSpec:
    hf_id: str
    local_name: str
    short_label: str
    vram_hint: str
    recommended: bool = False
    risk: str = ""


# Orden del dropdown: recomendado primero.
VL_SPECS: tuple[VlModelSpec, ...] = (
    VlModelSpec(
        hf_id="huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated",
        local_name="Huihui-Qwen3-VL-2B-Instruct-abliterated",
        short_label="Huihui 2B abliterated",
        vram_hint="~5 GB VRAM · mejor uncensor para 8 GB",
        recommended=True,
    ),
    VlModelSpec(
        hf_id="Qwen/Qwen3-VL-2B-Instruct",
        local_name="Qwen3-VL-2B-Instruct",
        short_label="Qwen3-VL-2B-Instruct",
        vram_hint="~5 GB VRAM · oficial (puede rechazar NSFW)",
    ),
    VlModelSpec(
        hf_id="huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
        local_name="Huihui-Qwen3-VL-4B-Instruct-abliterated",
        short_label="Huihui 4B abliterated",
        vram_hint="~9 GB pesos · calidad+ · riesgo OOM en 8 GB",
        risk="oom_8gb",
    ),
)

DEFAULT_HF_ID = VL_SPECS[0].hf_id
DEFAULT_LOCAL_NAME = VL_SPECS[0].local_name
DEFAULT_LABEL = VL_SPECS[0].short_label


@dataclass(frozen=True)
class VlModelChoice:
    label: str
    local_path: str
    hf_id: str
    kind: str
    caption_ready: bool
    mmproj_path: str = ""
    blocked_reason: str = ""
    recommended: bool = False
    risk: str = ""

    @property
    def value(self) -> str:
        if self.kind == "nan":
            model = (self.hf_id or "").strip()
            return f"nan:{model}" if model else NAN_VALUE
        if self.kind == "ollama":
            tag = (self.hf_id or "").strip() or DEFAULT_OLLAMA_MODEL
            return f"ollama:{tag}"
        return self.local_path or f"hf:{self.hf_id}"

    @property
    def is_uncensored(self) -> bool:
        blob = f"{self.hf_id} {self.label}".lower()
        return any(tok in blob for tok in ("abliterat", "uncensor", "huihui"))

    @property
    def is_ollama(self) -> bool:
        return self.kind == "ollama"

    @property
    def is_nan(self) -> bool:
        return self.kind == "nan"

    @property
    def is_remote(self) -> bool:
        """Backend API/servicio (sin descarga HF ni VRAM en el proceso Forge)."""
        return self.kind in ("ollama", "nan")


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
    fallback = Path("/data/Models/TextEncoders")
    if fallback.parent.is_dir():
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    host = Path("/home/pacogarat/Applications/Data/Models/TextEncoders")
    host.mkdir(parents=True, exist_ok=True)
    return host


def local_dir_for(spec: VlModelSpec | VlModelChoice | str) -> Path:
    if isinstance(spec, VlModelChoice):
        return Path(spec.local_path) if spec.local_path else text_encoder_root() / DEFAULT_LOCAL_NAME
    if isinstance(spec, VlModelSpec):
        return text_encoder_root() / spec.local_name
    # hf_id o local_name
    for s in VL_SPECS:
        if spec in (s.hf_id, s.local_name, f"hf:{s.hf_id}"):
            return text_encoder_root() / s.local_name
    return text_encoder_root() / DEFAULT_LOCAL_NAME


def default_local_dir() -> Path:
    return local_dir_for(VL_SPECS[0])


def is_local_ready(local_dir: Path | None = None) -> bool:
    root = local_dir or default_local_dir()
    if not root.is_dir():
        return False
    has_config = (root / "config.json").is_file()
    weights = list(root.glob("*.safetensors")) + list(root.glob("model*.safetensors"))
    if not weights:
        weights = list(root.glob("**/*.safetensors"))
    return has_config and len(weights) > 0


def _choice_from_spec(spec: VlModelSpec) -> VlModelChoice:
    local = local_dir_for(spec)
    ready = is_local_ready(local)
    tag = "listo en disco" if ready else "se descargará la 1ª vez"
    stars = "★ " if spec.recommended else ""
    risk = f" · ⚠ {spec.vram_hint}" if spec.risk else f" · {spec.vram_hint}"
    return VlModelChoice(
        label=f"{stars}{spec.short_label}{risk} · {tag}",
        local_path=str(local),
        hf_id=spec.hf_id,
        kind="hf",
        caption_ready=True,
        recommended=spec.recommended,
        risk=spec.risk,
    )


def sole_model_choice() -> VlModelChoice:
    """Compat: el modelo preferido (recomendado / primero del catálogo)."""
    return preferred_choice()


def _ollama_choice_for(tag: str, *, recommended: bool = False) -> VlModelChoice:
    tag = (tag or "").strip() or DEFAULT_OLLAMA_MODEL
    where = "cloud" if is_cloud_model(tag) else "local"
    stars = "★ " if recommended else ""
    return VlModelChoice(
        label=f"{stars}Ollama · {where} · {tag} · sin VRAM Forge",
        local_path="",
        hf_id=tag,
        kind="ollama",
        caption_ready=True,
        recommended=recommended,
    )


def ollama_choice(tag: str | None = None) -> VlModelChoice:
    """Una entrada Ollama (compat). Sin tag → default."""
    return _ollama_choice_for(
        tag or DEFAULT_OLLAMA_MODEL,
        recommended=(tag or DEFAULT_OLLAMA_MODEL) == DEFAULT_OLLAMA_MODEL,
    )


def ollama_choices() -> list[VlModelChoice]:
    """Entradas Ollama con visión (descubrimiento /api/tags o fallback)."""
    from forge_img2prompt.ollama_client import list_vision_models

    tags = list_vision_models(get_ollama_config())
    if not tags:
        tags = [DEFAULT_OLLAMA_MODEL]
    out: list[VlModelChoice] = []
    for i, tag in enumerate(tags):
        out.append(
            _ollama_choice_for(
                tag,
                recommended=tag == DEFAULT_OLLAMA_MODEL or (i == 0 and DEFAULT_OLLAMA_MODEL not in tags),
            )
        )
    return out


def _nan_choice_for(model: str, *, recommended: bool = False) -> VlModelChoice:
    model = (model or "").strip() or DEFAULT_NAN_MODEL
    risk = vision_risk(model)
    risk_note = " · ⚠ tier premium" if risk == "premium" else ""
    stars = "★ " if recommended else ""
    return VlModelChoice(
        label=f"{stars}NaN · {vision_label(model)}{risk_note} · sin VRAM Forge",
        local_path="",
        hf_id=model,
        kind="nan",
        caption_ready=True,
        recommended=recommended,
        risk=risk,
    )


def nan_choice(model: str | None = None) -> VlModelChoice:
    """Una entrada NaN (compat). Sin modelo → default."""
    return _nan_choice_for(
        model or DEFAULT_NAN_MODEL,
        recommended=(model or DEFAULT_NAN_MODEL) == DEFAULT_NAN_MODEL,
    )


def nan_configured() -> bool:
    """True si NaN tiene URL no-default o API key (env o Settings). Evita red inútil."""
    try:
        cfg = get_nan_config()
    except Exception:  # noqa: BLE001
        return False
    return bool(cfg.api_key) or cfg.base_url.rstrip("/") != DEFAULT_NAN_BASE_URL


def nan_choices(*, discover: bool = True) -> list[VlModelChoice]:
    """Entradas NaN con visión.

    Con ``discover`` y NaN configurado, consulta ``/v1/models`` una vez; si no,
    usa el registro curado (sin red), de modo que el arranque de Forge no llama
    a un endpoint que no se va a usar.
    """
    if not discover or not nan_configured():
        models = list(nan_vision_model_ids())
    else:
        from forge_img2prompt.nan_client import list_vision_models

        models = list_vision_models(get_nan_config())
    if not models:
        models = [DEFAULT_NAN_MODEL]
    if DEFAULT_NAN_MODEL in models:
        ordered = [DEFAULT_NAN_MODEL, *[m for m in models if m != DEFAULT_NAN_MODEL]]
    else:
        ordered = list(models)
    return [
        _nan_choice_for(
            model,
            recommended=model == DEFAULT_NAN_MODEL
            or (i == 0 and DEFAULT_NAN_MODEL not in ordered),
        )
        for i, model in enumerate(ordered)
    ]


def list_vl_models(extra_dirs=None) -> list[VlModelChoice]:
    del extra_dirs
    return [
        *nan_choices(),
        *ollama_choices(),
        *(_choice_from_spec(s) for s in VL_SPECS),
    ]


def dropdown_choices(catalog: list[VlModelChoice] | None = None) -> list[tuple[str, str]]:
    catalog = catalog if catalog is not None else list_vl_models()
    return [(c.label, c.value) for c in catalog]


def _ollama_configured() -> bool:
    """True si Ollama tiene URL no-default o API key (evita asumir local disponible)."""
    try:
        cfg = get_ollama_config()
    except Exception:  # noqa: BLE001
        return False
    return bool(cfg.api_key) or cfg.base_url.rstrip("/") != default_base_url().rstrip("/")


def _prefer_nan(catalog: list[VlModelChoice]) -> VlModelChoice | None:
    for c in catalog:
        if c.is_nan and c.hf_id == DEFAULT_NAN_MODEL:
            return c
    for c in catalog:
        if c.is_nan:
            return c
    return None


def preferred_choice(catalog: list[VlModelChoice] | None = None) -> VlModelChoice:
    catalog = catalog if catalog is not None else list_vl_models()
    if not catalog:
        return ollama_choice()
    # NaN configurado explícitamente gana cuando Ollama no lo está: así no se
    # propone un backend local que nadie configuró por delante del que sí.
    if nan_configured() and not _ollama_configured():
        nan_pref = _prefer_nan(catalog)
        if nan_pref is not None:
            return nan_pref
    # Ollama: default → local → cloud.
    for c in catalog:
        if c.is_ollama and c.hf_id == DEFAULT_OLLAMA_MODEL:
            return c
    for c in catalog:
        if c.is_ollama and not is_cloud_model(c.hf_id):
            return c
    for c in catalog:
        if c.is_ollama:
            return c
    nan_pref = _prefer_nan(catalog)
    if nan_pref is not None:
        return nan_pref
    for c in catalog:
        if c.recommended and c.local_path and is_local_ready(Path(c.local_path)):
            return c
    for c in catalog:
        if c.recommended:
            return c
    for c in catalog:
        if c.local_path and is_local_ready(Path(c.local_path)):
            return c
    return catalog[0]


def preferred_value(catalog: list[VlModelChoice] | None = None) -> str:
    return preferred_choice(catalog).value


def _resolve_nan(value: str, catalog: list[VlModelChoice]) -> VlModelChoice:
    model = parse_nan_value(value)
    for c in catalog:
        if c.is_nan and c.hf_id == model:
            return c
    # Modelo no listado aún (prefs antiguos / id nuevo): entrada ad-hoc.
    if value == NAN_VALUE:
        for c in catalog:
            if c.is_nan:
                return c
    return nan_choice(model)


def _resolve_ollama(value: str, catalog: list[VlModelChoice]) -> VlModelChoice:
    tag = parse_ollama_value(value)
    for c in catalog:
        if c.is_ollama and c.hf_id == tag:
            return c
    # Tag no listado aún (prefs antiguos / modelo nuevo): entrada ad-hoc.
    if value == OLLAMA_VALUE:
        return preferred_choice(catalog)
    return ollama_choice(tag)


def _resolve_local(value: str, catalog: list[VlModelChoice]) -> VlModelChoice:
    for c in catalog:
        if c.value == value or c.hf_id == value or value in (c.local_path, f"hf:{c.hf_id}"):
            return c
    for c in catalog:
        if c.local_path and (
            value.endswith(Path(c.local_path).name) or Path(c.local_path).name in value
        ):
            return c
    return preferred_choice(catalog)


# (predicado, resolvedor): despacho table-driven por prefijo del valor.
_CHOICE_RESOLVERS = (
    (is_nan_value, _resolve_nan),
    (is_ollama_value, _resolve_ollama),
)


def choice_by_value(value: str, catalog: list[VlModelChoice] | None = None) -> VlModelChoice | None:
    catalog = catalog if catalog is not None else list_vl_models()
    value = (value or "").strip()
    if not value:
        return preferred_choice(catalog)
    for matches, resolve in _CHOICE_RESOLVERS:
        if matches(value):
            return resolve(value, catalog)
    return _resolve_local(value, catalog)
