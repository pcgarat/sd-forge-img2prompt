from forge_img2prompt.stack import detect_stack, pick_text_encoder


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


def test_krea_path_and_empty_te_supported():
    s = detect_stack("Krea/krea2_krea2Int4Convrot_v10Turbo.safetensors", "", preset="krea")
    assert s.family == "krea2"
    assert s.variant == "turbo"
    assert s.is_supported
    assert s.preset == "krea"


def test_detects_klein_distilled():
    s = detect_stack("klein_Flux2-Klein-9B-int4-ConvRot.safetensors", "qwen_3_8b_fp8mixed.safetensors")
    assert s.family == "klein9b"
    assert s.variant == "distilled"
    assert s.is_supported


def test_detects_klein_base():
    s = detect_stack("flux-2-klein-base-9b.safetensors", "qwen3_8b.safetensors")
    assert s.family == "klein9b"
    assert s.variant == "base"
    assert s.is_supported


def test_rejects_unrelated_checkpoint():
    s = detect_stack("juggernautXL.safetensors", "clip_l.safetensors")
    assert s.family == "unknown"
    assert not s.is_supported


def test_krea_with_wrong_te_not_supported():
    s = detect_stack("krea2_turbo.safetensors", "clip_l.safetensors")
    assert s.family == "krea2"
    assert not s.is_supported


def test_pick_text_encoder_skips_vae():
    te = pick_text_encoder(
        [
            "/data/Models/VAE/flux2-vae.safetensors",
            "/data/Models/TextEncoders/qwen_3_8b_fp8mixed.safetensors",
        ]
    )
    assert "qwen_3_8b" in te


def test_pick_krea_te_from_modules():
    te = pick_text_encoder(
        [
            "/data/Models/TextEncoders/Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled.safetensors",
            "/data/Models/VAE/krea2RealVae_v10.safetensors",
        ]
    )
    assert "Qwen3-VL" in te or "qwen3-vl" in te.lower()
