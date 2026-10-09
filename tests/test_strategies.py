from pathlib import Path

import pytest

from forge_img2prompt.strategies import (
    DEFAULT_STRATEGY,
    STRATEGY_INSIDE_OUT,
    STRATEGY_LAYERED,
    STRATEGY_PROSE,
    configure,
    delete_strategy,
    factory_defaults,
    get_strategy,
    load_all,
    normalize_strategy,
    reset_strategy,
    strategy_choices,
    strategy_label,
    upsert_strategy,
)
from forge_img2prompt.vl_provider import _caption_system_for, _strategy_user_hint


@pytest.fixture(autouse=True)
def _reset_strategies_config():
    yield
    configure(None)


def test_factory_defaults_include_builtins():
    defaults = factory_defaults()
    assert STRATEGY_PROSE in defaults
    assert STRATEGY_INSIDE_OUT in defaults
    assert STRATEGY_LAYERED in defaults
    assert defaults[STRATEGY_PROSE].builtin
    assert defaults[STRATEGY_LAYERED].builtin
    assert "subject, action/pose" in defaults[STRATEGY_PROSE].caption_system
    layered = defaults[STRATEGY_LAYERED]
    assert layered.label == "Capas (sujeto→calidad)"
    assert "LAYER 1 SUBJECT" in layered.caption_system
    assert "LAYER 4" in layered.caption_system
    assert "masterpiece" in layered.caption_system.lower()
    assert "(word:1.5)" in layered.caption_system or "(word:1.5)" in layered.user_hint
    assert "four layers" in layered.user_hint.lower() or "LAYER 1" in layered.user_hint


def test_upsert_custom_and_choices(tmp_path: Path):
    configure(tmp_path)
    strat, msg = upsert_strategy(
        strategy_id="cinematic",
        label="Cinemático",
        caption_system="Caption cinematic system.",
        notes_system="Notes cinematic system.",
        user_hint="Prefer widescreen framing.",
        as_new=True,
    )
    assert strat is not None
    assert "guardada" in msg.lower()
    assert (tmp_path / "strategies.json").is_file()
    assert normalize_strategy("cinematic") == "cinematic"
    assert strategy_label("cinematic") == "Cinemático"
    labels = [label for label, _ in strategy_choices()]
    assert "Cinemático" in labels
    assert get_strategy("cinematic").user_hint == "Prefer widescreen framing."


def test_cannot_delete_builtin(tmp_path: Path):
    configure(tmp_path)
    ok, msg = delete_strategy(STRATEGY_PROSE)
    assert not ok
    assert "integradas" in msg.lower() or "restaurar" in msg.lower()


def test_edit_and_reset_builtin(tmp_path: Path):
    configure(tmp_path)
    original = get_strategy(STRATEGY_PROSE).caption_system
    edited, _ = upsert_strategy(
        strategy_id=STRATEGY_PROSE,
        label="Prosa",
        caption_system="EDITED CAPTION SYSTEM",
        notes_system=get_strategy(STRATEGY_PROSE).notes_system,
        user_hint=get_strategy(STRATEGY_PROSE).user_hint,
        as_new=False,
    )
    assert edited is not None
    assert get_strategy(STRATEGY_PROSE).caption_system == "EDITED CAPTION SYSTEM"
    assert _caption_system_for(STRATEGY_PROSE) == "EDITED CAPTION SYSTEM"
    restored, _ = reset_strategy(STRATEGY_PROSE)
    assert restored is not None
    assert get_strategy(STRATEGY_PROSE).caption_system == original


def test_delete_custom(tmp_path: Path):
    configure(tmp_path)
    upsert_strategy(
        strategy_id="temp_one",
        label="Temporal",
        caption_system="c",
        notes_system="n",
        user_hint="h",
        as_new=True,
    )
    ok, _ = delete_strategy("temp_one")
    assert ok
    assert "temp_one" not in load_all()
    assert normalize_strategy("temp_one") == DEFAULT_STRATEGY


def test_normalize_aliases_still_work(tmp_path: Path):
    configure(tmp_path)
    assert normalize_strategy("Prosa") == STRATEGY_PROSE
    assert normalize_strategy("Dentro → fuera") == STRATEGY_INSIDE_OUT
    assert normalize_strategy("weird") == DEFAULT_STRATEGY


def test_custom_strategy_feeds_vl_helpers(tmp_path: Path):
    configure(tmp_path)
    upsert_strategy(
        strategy_id="dense_tags",
        label="Tags densos",
        caption_system="Use dense visual tags.",
        notes_system="Notes dense tags.",
        user_hint="Prefer comma-free dense phrasing.",
        as_new=True,
    )
    assert _caption_system_for("dense_tags") == "Use dense visual tags."
    assert "dense phrasing" in _strategy_user_hint("dense_tags")


def test_duplicate_new_rejected(tmp_path: Path):
    configure(tmp_path)
    upsert_strategy(
        strategy_id="dup",
        label="Dup",
        caption_system="c",
        notes_system="n",
        user_hint="h",
        as_new=True,
    )
    again, msg = upsert_strategy(
        strategy_id="dup",
        label="Dup 2",
        caption_system="c2",
        notes_system="n2",
        user_hint="h2",
        as_new=True,
    )
    assert again is None
    assert "existe" in msg.lower()
