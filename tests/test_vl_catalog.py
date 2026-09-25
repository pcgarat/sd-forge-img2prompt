from pathlib import Path

from forge_img2prompt.vl_catalog import (
    DEFAULT_HF_ID,
    VL_SPECS,
    choice_by_value,
    dropdown_choices,
    is_local_ready,
    list_vl_models,
    preferred_choice,
    preferred_value,
    sole_model_choice,
)


def test_catalog_has_abliterated_and_official():
    catalog = list_vl_models()
    assert len(catalog) == len(VL_SPECS) == 3
    ids = {c.hf_id for c in catalog}
    assert "huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated" in ids
    assert "Qwen/Qwen3-VL-2B-Instruct" in ids
    assert "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated" in ids
    assert preferred_choice(catalog).recommended
    assert preferred_choice(catalog).hf_id == DEFAULT_HF_ID
    assert sole_model_choice().hf_id == DEFAULT_HF_ID
    pairs = dropdown_choices(catalog)
    assert len(pairs) == 3
    assert preferred_value(catalog) == preferred_choice(catalog).value


def test_choice_by_value_and_uncensored_flag():
    catalog = list_vl_models()
    abl = next(c for c in catalog if "abliterated" in c.hf_id and "2B" in c.hf_id)
    assert abl.is_uncensored
    official = next(c for c in catalog if c.hf_id.startswith("Qwen/"))
    assert not official.is_uncensored
    assert choice_by_value(abl.value, catalog).hf_id == abl.hf_id
    assert choice_by_value(abl.hf_id, catalog).hf_id == abl.hf_id
    four = next(c for c in catalog if "4B" in c.hf_id)
    assert four.risk == "oom_8gb"


def test_is_local_ready_false_on_empty(tmp_path: Path):
    assert not is_local_ready(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    assert not is_local_ready(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert is_local_ready(tmp_path)


def test_format_plan_marks_and_repo():
    from forge_img2prompt.vl_download import DownloadItem, format_download_plan

    items = [
        DownloadItem("model.safetensors", 1024 * 1024 * 100),
        DownloadItem("config.json", 100),
    ]
    md = format_download_plan(
        items,
        Path("/tmp/x"),
        repo_id="huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated",
        done={"config.json"},
        current="model.safetensors",
    )
    assert "✅" in md and "⬇️" in md
    assert "config.json" in md
    assert "huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated" in md
