from pathlib import Path

from forge_img2prompt.prefs import (
    format_range,
    load_prefs,
    parse_range_text,
    save_prefs,
)
from forge_img2prompt.provider import (
    DETAIL_WORDS_BOUNDS,
    DETAIL_WORDS_DEFAULT,
    GEN_WORDS_BOUNDS,
    GEN_WORDS_DEFAULT,
)
from forge_img2prompt.sequence import SEQ_MAX_SCENES


def test_format_and_parse_range():
    assert format_range(45, 90) == "45,90"
    assert parse_range_text(
        "45,90", bounds=GEN_WORDS_BOUNDS, default=GEN_WORDS_DEFAULT
    ) == (45, 90)
    assert parse_range_text(
        "10-40", bounds=DETAIL_WORDS_BOUNDS, default=DETAIL_WORDS_DEFAULT
    ) == (10, 40)


def test_parse_range_swaps_inverted():
    lo, hi = parse_range_text(
        "90,45", bounds=GEN_WORDS_BOUNDS, default=GEN_WORDS_DEFAULT
    )
    assert lo == 45 and hi == 90


def test_prefs_roundtrip(tmp_path: Path):
    saved = save_prefs(
        tmp_path,
        {
            "gen_wmin": 50,
            "gen_wmax": 100,
            "det_wmin": 10,
            "det_wmax": 30,
            "det_overlap": 0,
            "language": "en",
            "strategy": "inside_out",
            "vl_value": "huihui-2b",
        },
    )
    assert saved["gen_wmin"] == 50
    assert (tmp_path / "ui_prefs.json").is_file()
    loaded = load_prefs(tmp_path)
    assert loaded["gen_wmax"] == 100
    assert loaded["det_overlap"] == 0
    assert loaded["language"] == "en"
    assert loaded["strategy"] == "inside_out"
    assert loaded["vl_value"] == "huihui-2b"


def test_prefs_defaults_when_missing(tmp_path: Path):
    data = load_prefs(tmp_path)
    assert data["gen_wmin"] == GEN_WORDS_DEFAULT[0]
    assert data["det_wmax"] == DETAIL_WORDS_DEFAULT[1]
    assert data["strategy"] == "prose"


def test_prefs_loras_roundtrip(tmp_path: Path):
    save_prefs(
        tmp_path,
        {
            "gen_loras": "<lora:pelo:0.8>",
            "seq_loras": ["<lora:uno:1>", "", "<lora:tres:0.5>"],
        },
    )
    data = load_prefs(tmp_path)
    assert data["gen_loras"] == "<lora:pelo:0.8>"
    assert data["seq_loras"][0] == "<lora:uno:1>"
    assert data["seq_loras"][2] == "<lora:tres:0.5>"
    assert len(data["seq_loras"]) == SEQ_MAX_SCENES
    assert data["seq_loras"][3:] == [""] * (SEQ_MAX_SCENES - 3)


def test_prefs_loras_defaults_are_safe(tmp_path: Path):
    data = load_prefs(tmp_path)
    assert data["gen_loras"] == ""
    assert data["seq_loras"] == [""] * SEQ_MAX_SCENES


def test_prefs_loras_tolerates_garbage(tmp_path: Path):
    (tmp_path / "ui_prefs.json").write_text(
        '{"gen_loras": null, "seq_loras": "no-es-lista"}', encoding="utf-8"
    )
    data = load_prefs(tmp_path)
    assert data["gen_loras"] == ""
    assert data["seq_loras"] == [""] * SEQ_MAX_SCENES


def test_prefs_loras_truncates_extra_scenes(tmp_path: Path):
    save_prefs(
        tmp_path,
        {"seq_loras": [f"lora{i}" for i in range(SEQ_MAX_SCENES + 5)]},
    )
    data = load_prefs(tmp_path)
    assert len(data["seq_loras"]) == SEQ_MAX_SCENES
    assert data["seq_loras"][SEQ_MAX_SCENES - 1] == f"lora{SEQ_MAX_SCENES - 1}"
