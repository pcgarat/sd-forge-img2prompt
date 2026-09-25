from pathlib import Path

from forge_img2prompt.vl_catalog import (
    DEFAULT_HF_ID,
    dropdown_choices,
    is_local_ready,
    list_vl_models,
    preferred_value,
    sole_model_choice,
)


def test_sole_catalog():
    catalog = list_vl_models()
    assert len(catalog) == 1
    assert catalog[0].hf_id == DEFAULT_HF_ID
    assert "2B" in catalog[0].label
    pairs = dropdown_choices(catalog)
    assert len(pairs) == 1
    assert preferred_value(catalog) == catalog[0].value
    assert sole_model_choice().hf_id == DEFAULT_HF_ID


def test_is_local_ready_false_on_empty(tmp_path: Path):
    assert not is_local_ready(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    assert not is_local_ready(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert is_local_ready(tmp_path)


def test_format_plan_marks():
    from forge_img2prompt.vl_download import DownloadItem, format_download_plan

    items = [
        DownloadItem("model.safetensors", 1024 * 1024 * 100),
        DownloadItem("config.json", 100),
    ]
    md = format_download_plan(items, Path("/tmp/x"), done={"config.json"}, current="model.safetensors")
    assert "✅" in md and "⬇️" in md
    assert "config.json" in md
