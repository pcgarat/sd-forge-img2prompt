"""Image → prompt helpers for Forge Neo (Krea 2 / Klein + Qwen VL)."""

from forge_img2prompt.log import log
from forge_img2prompt.provider import (
    DetailRequest,
    PromptProvider,
    PromptRequest,
    PromptResult,
    StubProvider,
    append_detail,
    clamp_overlap_discard,
    clamp_text_to_words,
    clamp_word_range,
    count_words,
    detail_is_redundant,
    detail_looks_like_full_scene,
    detail_looks_like_tag_soup,
    max_tokens_for_words,
    normalize_zone_anchor,
    scene_context_snippet,
    shared_content_count,
)
from forge_img2prompt.stack import StackInfo, detect_stack
from forge_img2prompt.vl_catalog import VlModelChoice, list_vl_models
from forge_img2prompt.vl_provider import CompositeProvider, QwenVLProvider

__all__ = [
    "PromptProvider",
    "PromptRequest",
    "DetailRequest",
    "PromptResult",
    "StubProvider",
    "CompositeProvider",
    "QwenVLProvider",
    "StackInfo",
    "VlModelChoice",
    "append_detail",
    "clamp_overlap_discard",
    "clamp_text_to_words",
    "clamp_word_range",
    "count_words",
    "detail_is_redundant",
    "detail_looks_like_full_scene",
    "detail_looks_like_tag_soup",
    "max_tokens_for_words",
    "normalize_zone_anchor",
    "scene_context_snippet",
    "shared_content_count",
    "detect_stack",
    "list_vl_models",
    "log",
]
