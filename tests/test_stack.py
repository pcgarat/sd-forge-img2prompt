from forge_img2prompt.stack import detect_stack


def test_detects_krea2_turbo():
    s = detect_stack("krea2_turbo_fp8.safetensors", "qwen3vl_4b_bf16.safetensors")
    assert s.family == "krea2"
    assert s.variant == "turbo"
    assert s.is_supported


def test_detects_krea2_raw():
    s = detect_stack("models/Krea2_RAW.safetensors", "text_encoder/qwen3vl_4b_fp8_scaled.safetensors")
    assert s.family == "krea2"
    assert s.variant == "raw"
    assert s.is_supported


def test_krea_without_variant_still_supported():
    s = detect_stack("krea2.safetensors", "qwen3vl_4b.safetensors")
    assert s.family == "krea2"
    assert s.variant == "unknown"
    assert s.is_supported


def test_rejects_unrelated_checkpoint():
    s = detect_stack("flux2_klein_9b.safetensors", "qwen_3_8b.safetensors")
    assert s.family == "unknown"
    assert not s.is_supported


def test_krea_with_wrong_te_not_supported():
    s = detect_stack("krea2_turbo.safetensors", "clip_l.safetensors")
    assert s.family == "krea2"
    assert not s.is_supported
