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
            "vl_value": "huihui-2b",
        },
    )
    assert saved["gen_wmin"] == 50
    assert (tmp_path / "ui_prefs.json").is_file()
    loaded = load_prefs(tmp_path)
    assert loaded["gen_wmax"] == 100
    assert loaded["det_overlap"] == 0
    assert loaded["language"] == "en"
    assert loaded["vl_value"] == "huihui-2b"


def test_prefs_defaults_when_missing(tmp_path: Path):
    data = load_prefs(tmp_path)
    assert data["gen_wmin"] == GEN_WORDS_DEFAULT[0]
    assert data["det_wmax"] == DETAIL_WORDS_DEFAULT[1]
