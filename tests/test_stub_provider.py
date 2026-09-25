from forge_img2prompt.provider import StubProvider, PromptRequest
from forge_img2prompt.stack import detect_stack


def _krea(variant_ckpt: str = "krea2_turbo.safetensors"):
    return detect_stack(variant_ckpt, "qwen3vl_4b.safetensors")


def test_stub_rewrites_notes_to_prose():
    provider = StubProvider()
    result = provider.generate(
        PromptRequest(
            image=None,
            user_notes="a red ceramic teapot on a wooden table in morning light",
            stack=_krea(),
        )
    )
    assert "teapot" in result.prompt.lower()
    assert result.prompt[0].isupper()
    assert "8 steps" in result.sampler_hints or "Turbo" in result.sampler_hints
    assert result.status
    assert "," not in result.prompt[:20] or "teapot" in result.prompt


def test_stub_avoids_claiming_optimization_when_unsupported():
    provider = StubProvider()
    stack = detect_stack("sdxl.safetensors", "clip_l.safetensors")
    result = provider.generate(PromptRequest(image=None, user_notes="a cat", stack=stack))
    assert result.prompt == ""
    assert "no reconocido" in result.status.lower() or "No se genera" in result.status


def test_stub_empty_notes_uses_template():
    provider = StubProvider()
    result = provider.generate(PromptRequest(image=None, user_notes="  ", stack=_krea("krea2_raw.safetensors")))
    assert "subject" in result.prompt.lower() or "photograph" in result.prompt.lower()
    assert "RAW" in result.sampler_hints or "28" in result.sampler_hints
    assert "comillas" in result.prompt.lower() or "quotes" in result.prompt.lower()


def test_stub_softens_tag_soup():
    provider = StubProvider()
    notes = "1girl, solo, long hair, masterpiece, best quality, cyberpunk, neon, city, rain"
    result = provider.generate(PromptRequest(image=None, user_notes=notes, stack=_krea()))
    assert "natural-language" in result.prompt.lower() or "coherent" in result.prompt.lower()
