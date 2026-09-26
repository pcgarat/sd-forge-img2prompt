from forge_img2prompt.provider import LANG_EN, LANG_ES, StubProvider, PromptRequest
from forge_img2prompt.stack import detect_stack


def _krea(variant_ckpt: str = "krea2_turbo.safetensors"):
    return detect_stack(variant_ckpt, "qwen3vl_4b.safetensors")


def _klein():
    return detect_stack("klein_Flux2-Klein-9B.safetensors", "qwen_3_8b.safetensors")


def test_stub_rewrites_notes_to_prose():
    provider = StubProvider()
    result = provider.generate(
        PromptRequest(
            image=None,
            user_notes="a red ceramic teapot on a wooden table in morning light",
            stack=_krea(),
            language=LANG_EN,
        )
    )
    assert "teapot" in result.prompt.lower()
    assert result.prompt[0].isupper()
    assert "8 steps" in result.sampler_hints or "Turbo" in result.sampler_hints
    assert "notas" in result.status.lower() or "caracteres" in result.status.lower()


def test_different_notes_produce_different_prompts():
    provider = StubProvider()
    a = provider.generate(PromptRequest(None, "a blue sports car at dusk", _krea(), LANG_EN)).prompt
    b = provider.generate(PromptRequest(None, "an old wooden cabin in snow", _krea(), LANG_EN)).prompt
    assert a != b
    assert "sports car" in a.lower()
    assert "cabin" in b.lower()


def test_stub_klein_hints():
    provider = StubProvider()
    result = provider.generate(
        PromptRequest(image=None, user_notes="a cat on a windowsill", stack=_klein(), language=LANG_EN)
    )
    assert "cat" in result.prompt.lower()
    assert "4 steps" in result.sampler_hints or "Klein" in result.sampler_hints
    assert "Klein" in result.status


def test_stub_avoids_claiming_optimization_when_unsupported():
    provider = StubProvider()
    stack = detect_stack("sdxl.safetensors", "clip_l.safetensors")
    result = provider.generate(PromptRequest(image=None, user_notes="a cat", stack=stack))
    assert result.prompt == ""
    assert "no reconocido" in result.status.lower() or "No se genera" in result.status


def test_stub_empty_notes_no_fake_prompt():
    provider = StubProvider()
    result = provider.generate(PromptRequest(image=None, user_notes="  ", stack=_krea("krea2_raw.safetensors")))
    assert result.prompt == ""
    assert "no analiza" in result.status.lower() or "Notas" in result.status
    assert "RAW" in result.sampler_hints or "28" in result.sampler_hints


def test_stub_softens_tag_soup():
    provider = StubProvider()
    notes = "1girl, solo, long hair, masterpiece, best quality, cyberpunk, neon, city, rain"
    result = provider.generate(
        PromptRequest(image=None, user_notes=notes, stack=_krea(), language=LANG_EN)
    )
    assert "natural language" in result.prompt.lower() or "prose" in result.prompt.lower() or "keyword" in result.prompt.lower()


def test_stub_spanish_framing():
    provider = StubProvider()
    result = provider.generate(
        PromptRequest(
            image=None,
            user_notes="un gato en el alféizar",
            stack=_krea(),
            language=LANG_ES,
        )
    )
    assert "gato" in result.prompt.lower()
    assert "composición" in result.prompt.lower() or "iluminación" in result.prompt.lower()
    assert "español" in result.status.lower()


def test_stub_spanish_tag_soup():
    provider = StubProvider()
    notes = "1girl, solo, long hair, masterpiece, best quality, cyberpunk, neon, city, rain"
    result = provider.generate(
        PromptRequest(image=None, user_notes=notes, stack=_krea(), language=LANG_ES)
    )
    assert "escena detallada" in result.prompt.lower() or "prosa" in result.prompt.lower()
