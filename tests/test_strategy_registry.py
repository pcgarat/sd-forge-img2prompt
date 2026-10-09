"""Registro de estrategias: carga desde strategies.json y fallback embebido."""

from __future__ import annotations

import json

import pytest

import forge_img2prompt.strategy_registry as sr
from forge_img2prompt.provider import (
    DEFAULT_STRATEGY,
    STRATEGY_CHOICES,
    STRATEGY_REGISTRY,
    normalize_strategy,
    strategy_label,
)
from forge_img2prompt.vl_provider import _caption_system_for, _notes_system_for, _strategy_user_hint


def test_strategies_json_loads_layered():
    ids = [s.id for s in sr.load_strategies()]
    assert "prose" in ids
    assert "inside_out" in ids
    # strategies.json define la tercera estrategia; se carga y se expone en la UI.
    assert "layered" in ids
    assert ("Capas (sujeto→calidad)", "layered") in STRATEGY_CHOICES


def test_registry_normalizes_aliases_and_labels():
    assert normalize_strategy("Dentro → fuera") == "inside_out"
    assert normalize_strategy("prosa") == "prose"
    assert normalize_strategy("unknown-xyz") == DEFAULT_STRATEGY
    assert strategy_label("inside_out") == "Dentro → fuera"
    assert strategy_label("layered") == "Capas (sujeto→calidad)"


def test_registry_prompts_come_from_json():
    caption = _caption_system_for("layered").lower()
    assert "capa 1" in caption or "layered" in caption or "capas" in caption
    notes = _notes_system_for("layered").lower()
    assert notes
    assert _strategy_user_hint("inside_out")


def test_invalid_json_falls_back_to_builtins(tmp_path, monkeypatch):
    bad = tmp_path / "strategies.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    monkeypatch.setattr(sr, "_strategies_path", lambda: bad)
    specs = sr.load_strategies()
    ids = [s.id for s in specs]
    assert ids == ["prose", "inside_out"]


def test_empty_strategies_falls_back_to_builtins(tmp_path, monkeypatch):
    empty = tmp_path / "strategies.json"
    empty.write_text(json.dumps({"strategies": []}), encoding="utf-8")
    monkeypatch.setattr(sr, "_strategies_path", lambda: empty)
    registry = sr.make_registry()
    assert registry.default.id == "prose"
    assert registry.choices() == (("Prosa", "prose"), ("Dentro → fuera", "inside_out"))


def test_registry_skips_invalid_entries(tmp_path, monkeypatch):
    path = tmp_path / "strategies.json"
    path.write_text(
        json.dumps(
            {
                "strategies": [
                    {"id": "good", "label": "Buena", "caption_system": "c", "notes_system": "n"},
                    {"id": "", "caption_system": "c", "notes_system": "n"},
                    {"id": "noprompts"},
                    {"id": "good", "caption_system": "dup", "notes_system": "dup"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sr, "_strategies_path", lambda: path)
    specs = sr.load_strategies()
    assert [s.id for s in specs] == ["good"]


def test_compat_module_level_registry_is_consistency():
    assert STRATEGY_REGISTRY.default.id == DEFAULT_STRATEGY
    assert {v for _, v in STRATEGY_CHOICES} == {s.id for s in STRATEGY_REGISTRY.specs}
