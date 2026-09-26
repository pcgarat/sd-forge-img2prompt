from pathlib import Path
from unittest.mock import patch

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


def _fake_vision_tags():
    return ["qwen3-vl:8b-instruct", "kimi-k3:cloud", "gemma4:31b-cloud"]


def test_catalog_has_ollama_vision_and_hf():
    with patch(
        "forge_img2prompt.ollama_client.list_vision_models",
        return_value=_fake_vision_tags(),
    ):
        catalog = list_vl_models()
    ollama = [c for c in catalog if c.is_ollama]
    assert len(ollama) == 3
    assert len(catalog) == 3 + len(VL_SPECS)
    assert catalog[0].is_ollama
    assert catalog[0].value == "ollama:qwen3-vl:8b-instruct"
    ids = {c.hf_id for c in catalog}
    assert "huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated" in ids
    assert "Qwen/Qwen3-VL-2B-Instruct" in ids
    assert "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated" in ids
    assert preferred_choice(catalog).is_ollama
    assert preferred_choice(catalog).hf_id == "qwen3-vl:8b-instruct"
    assert preferred_choice(catalog).recommended
    assert sole_model_choice().is_ollama
    pairs = dropdown_choices(catalog)
    assert len(pairs) == 3 + len(VL_SPECS)
    assert preferred_value(catalog) == "ollama:qwen3-vl:8b-instruct"
    assert DEFAULT_HF_ID == "huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated"


def test_choice_by_value_ollama_and_uncensored():
    with patch(
        "forge_img2prompt.ollama_client.list_vision_models",
        return_value=_fake_vision_tags(),
    ):
        catalog = list_vl_models()
    kimi = choice_by_value("ollama:kimi-k3:cloud", catalog)
    assert kimi is not None and kimi.hf_id == "kimi-k3:cloud"
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


def test_byte_weights_prefer_large_file():
    from forge_img2prompt.vl_download import DownloadItem, _bytes_total, _item_weight

    items = [
        DownloadItem("model.safetensors", 4_000_000_000),
        DownloadItem("config.json", 1000),
    ]
    assert _bytes_total(items) == 4_000_001_000
    assert _item_weight(items[0], items) == 4_000_000_000
    # El peso del grande es ~100% del progreso; por ficheros sería 50%.
    assert _item_weight(items[0], items) / _bytes_total(items) > 0.99


def test_throttled_file_progress_maps_bytes():
    from forge_img2prompt.vl_download import _throttled_file_progress

    seen: list[tuple[float, str]] = []

    cb = _throttled_file_progress(
        filename="model.safetensors",
        size_label="3.72 GB",
        progress=lambda f, d: seen.append((f, d)),
        frac_start=0.1,
        frac_end=0.8,
        min_interval_s=0.0,
    )
    cb(0, 1000)
    cb(500, 1000)
    cb(1000, 1000)
    assert seen[0][0] == 0.1
    assert abs(seen[1][0] - 0.45) < 1e-9
    assert abs(seen[-1][0] - 0.8) < 1e-9
    assert "50%" in seen[1][1]


def test_iter_model_download_already_ready(tmp_path: Path):
    from forge_img2prompt.vl_download import DownloadEvent, iter_model_download

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    events = list(iter_model_download(repo_id="unused/repo", local_dir=tmp_path))
    assert len(events) == 1
    assert isinstance(events[0], DownloadEvent)
    assert "ya descargado" in events[0].status.lower() or "listo" in events[0].plan.lower()


def test_ensure_model_downloaded_delegates(tmp_path: Path):
    from forge_img2prompt.vl_download import ensure_model_downloaded

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    plans: list[str] = []
    statuses: list[str] = []
    out = ensure_model_downloaded(
        repo_id="unused/repo",
        local_dir=tmp_path,
        plan_update=plans.append,
        status_update=statuses.append,
    )
    assert out == tmp_path
    assert plans and statuses
