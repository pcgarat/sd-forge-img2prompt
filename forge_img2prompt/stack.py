from __future__ import annotations

from dataclasses import dataclass

_KREA_CKPT = ("krea2", "krea-2", "krea_2", "krea 2")
_TURBO = ("turbo",)
_RAW = ("raw", "oss_raw")
_TE = ("qwen3vl", "qwen3-vl", "qwen_3_vl")


def _norm(name: str) -> str:
    return (name or "").strip().lower().replace("\\", "/")


@dataclass(frozen=True)
class StackInfo:
    family: str
    variant: str
    checkpoint: str
    text_encoder: str
    is_supported: bool

    @property
    def summary(self) -> str:
        te = self.text_encoder or "(none)"
        return f"{self.family}/{self.variant} · ckpt={self.checkpoint or '?'} · te={te}"


def detect_stack(checkpoint: str, text_encoder: str = "") -> StackInfo:
    ckpt = _norm(checkpoint)
    te = _norm(text_encoder)
    ckpt_base = ckpt.rsplit("/", 1)[-1]

    is_krea = any(tok in ckpt_base for tok in _KREA_CKPT) or any(
        tok in ckpt for tok in _KREA_CKPT
    )
    te_ok = (not te) or any(tok in te for tok in _TE)

    if is_krea and any(tok in ckpt_base for tok in _TURBO):
        variant = "turbo"
    elif is_krea and any(tok in ckpt_base for tok in _RAW):
        variant = "raw"
    elif is_krea:
        variant = "unknown"
    else:
        variant = "unknown"

    family = "krea2" if is_krea else "unknown"
    supported = family == "krea2" and te_ok

    return StackInfo(
        family=family,
        variant=variant if is_krea else "unknown",
        checkpoint=checkpoint or "",
        text_encoder=text_encoder or "",
        is_supported=supported,
    )
