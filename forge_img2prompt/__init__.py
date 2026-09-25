"""Image → prompt helpers for Forge Neo (Krea 2 / Klein + Qwen VL)."""

from forge_img2prompt.log import log
from forge_img2prompt.provider import PromptProvider, PromptRequest, PromptResult, StubProvider
from forge_img2prompt.stack import StackInfo, detect_stack
from forge_img2prompt.vl_catalog import VlModelChoice, list_vl_models
from forge_img2prompt.vl_provider import CompositeProvider, QwenVLProvider

__all__ = [
    "PromptProvider",
    "PromptRequest",
    "PromptResult",
    "StubProvider",
    "CompositeProvider",
    "QwenVLProvider",
    "StackInfo",
    "VlModelChoice",
    "detect_stack",
    "list_vl_models",
    "log",
]
