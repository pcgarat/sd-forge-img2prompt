from __future__ import annotations

from dataclasses import dataclass

_KREA_CKPT = ("krea2", "krea-2", "krea_2", "krea 2", "/krea/")
_KLEIN_CKPT = ("klein", "flux2-klein", "flux.2-klein", "flux_2_klein")
_TURBO = ("turbo",)
_RAW = ("raw", "oss_raw")
_BASE = ("base",)
_KREA_TE = ("qwen3vl", "qwen3-vl", "qwen_3_vl", "qwen3_vl")
_KLEIN_TE = ("qwen3_8b", "qwen_3_8b", "qwen3-8b", "qwen3 8b")
_VAE_HINTS = ("vae", "ae.safetensors", "flux2-vae", "krea2realvae")


def _norm(name: str) -> str:
    return (name or "").strip().lower().replace("\\", "/")


def _basename(path: str) -> str:
    return _norm(path).rsplit("/", 1)[-1]


def pick_text_encoder(modules: list[str] | tuple[str, ...] | None) -> str:
    """Pick TE-like file from Forge Neo VAE/Text Encoder multiselect paths."""
    if not modules:
        return ""
    candidates: list[str] = []
    for m in modules:
        base = _basename(str(m))
        if not base or any(h in base for h in _VAE_HINTS):
            continue
        candidates.append(str(m))
    for m in candidates:
        b = _basename(m)
        if any(tok in b for tok in _KREA_TE + _KLEIN_TE) or "qwen" in b:
            return m
    return candidates[0] if candidates else ""


@dataclass(frozen=True)
class StackInfo:
    family: str
    variant: str
    checkpoint: str
    text_encoder: str
    is_supported: bool
    preset: str = ""

    @property
    def summary(self) -> str:
        te = _basename(self.text_encoder) if self.text_encoder else "(none)"
        ckpt = _basename(self.checkpoint) if self.checkpoint else "?"
        preset = f" preset={self.preset}" if self.preset else ""
        return f"{self.family}/{self.variant} · ckpt={ckpt} · te={te}{preset}"


def detect_stack(
    checkpoint: str,
    text_encoder: str = "",
    *,
    preset: str = "",
) -> StackInfo:
    ckpt = _norm(checkpoint)
    te = _norm(text_encoder)
    ckpt_base = _basename(checkpoint)

    is_krea = any(tok in ckpt_base for tok in _KREA_CKPT) or any(tok in ckpt for tok in _KREA_CKPT)
    is_klein = (not is_krea) and (
        any(tok in ckpt_base for tok in _KLEIN_CKPT) or any(tok in ckpt for tok in _KLEIN_CKPT)
    )

    if is_krea:
        family = "krea2"
        if any(tok in ckpt_base for tok in _TURBO):
            variant = "turbo"
        elif any(tok in ckpt_base for tok in _RAW):
            variant = "raw"
        else:
            variant = "unknown"
        # TE vacío: Forge a veces no expone el módulo hasta regenerar; confiar en ckpt.
        te_ok = (not te) or any(tok in te for tok in _KREA_TE) or "qwen3" in te and "vl" in te
        if te and "qwen" in te and "vl" in te:
            te_ok = True
    elif is_klein:
        family = "klein9b"
        if any(tok in ckpt_base for tok in _BASE):
            variant = "base"
        else:
            # Distilled 4-step is the common local Klein 9B packaging.
            variant = "distilled"
        te_ok = (not te) or any(tok in te for tok in _KLEIN_TE) or ("qwen3" in te and "8b" in te)
        if te and "qwen" in te and "8b" in te:
            te_ok = True
    else:
        family = "unknown"
        variant = "unknown"
        te_ok = False

    return StackInfo(
        family=family,
        variant=variant,
        checkpoint=checkpoint or "",
        text_encoder=text_encoder or "",
        is_supported=family in ("krea2", "klein9b") and te_ok,
        preset=preset or "",
    )
