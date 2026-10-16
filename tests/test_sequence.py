"""Secuencia: biblia de continuidad, deltas, ensamblado y recomposición."""

from __future__ import annotations

import pytest

from forge_img2prompt.provider import LANG_EN, LANG_ES
from forge_img2prompt.sequence import (
    ANCHOR_FIELDS,
    ANCHOR_WORDS_DEFAULT,
    DELTA_FIELDS,
    SEQ_MAX_SCENES,
    SEQ_MIN_SCENES,
    ConsistencyAnchor,
    SceneDelta,
    SequenceRequest,
    anchor_system,
    anchor_user_text,
    clamp_scene_count,
    compose_prompt,
    compose_sequence,
    continuity_notes,
    degraded_prompts,
    delta_system,
    delta_user_text,
    field_for_key,
    handle,
    parse_anchor,
    parse_bible,
    parse_delta,
    recompose,
    score_duplicate,
    state_payload,
)

_BIBLE_JSON = """
Aquí tienes la ficha:
```json
{
  "anchor": {
    "subject": "un chico de 17 años con pelo castaño revuelto",
    "wardrobe": "camiseta gris de algodón",
    "setting": "habitación juvenil de los 90 con pósters",
    "lighting": "luz de amanecer entrando por la ventana"
  },
  "scene_1": {"action": "bostezando sentado en la cama"}
}
```
"""


def test_clamp_scene_count_bounds():
    assert clamp_scene_count(None) == 2
    assert clamp_scene_count(0) == SEQ_MIN_SCENES
    assert clamp_scene_count(999) == SEQ_MAX_SCENES
    assert clamp_scene_count("3") == 3


def test_field_for_key_maps_spanish_and_english():
    assert field_for_key("subject") == "subject"
    assert field_for_key("Sujeto") == "subject"
    assert field_for_key("ropa / vestuario") == "wardrobe"
    assert field_for_key("acción") == "action"
    assert field_for_key("iluminación") == "lighting"
    assert field_for_key("unknown-key") is None


def test_parse_bible_splits_anchor_and_scene1():
    anchor, scene1 = parse_bible(_BIBLE_JSON)
    assert anchor.subject.startswith("un chico de 17")
    assert anchor.wardrobe == "camiseta gris de algodón"
    assert anchor.lighting.startswith("luz de amanecer")
    assert scene1.action == "bostezando sentado en la cama"


def test_parse_anchor_tolerates_fences_and_trailing_text():
    anchor = parse_anchor('bla {```json\n{"subject": "un gato naranja"}\n```} bla')
    assert anchor.subject == "un gato naranja"


def test_parse_anchor_trailing_comma():
    anchor = parse_anchor('{"subject": "un gato", "setting": "un tejado",}')
    assert anchor.subject == "un gato"
    assert anchor.setting == "un tejado"


def test_parse_delta_falls_back_to_key_value_lines():
    delta = parse_delta("- Action: se levanta de la cama\n- Luz: amanece más claro")
    assert delta.action == "se levanta de la cama"
    assert delta.lighting == "amanece más claro"


def test_parse_anchor_empty_text_gives_empty_anchor():
    assert parse_anchor("").is_empty()
    assert parse_delta("").is_empty()


def test_parse_anchor_unwraps_single_nested_object():
    anchor = parse_anchor('{"biblia": {"sujeto": "una mujer con gafas"}}')
    assert anchor.subject == "una mujer con gafas"


def test_consistency_anchor_ignores_unknown_fields():
    anchor = ConsistencyAnchor.from_mapping({"subject": "A", "inventado": "B"})
    assert anchor.subject == "A"
    assert not hasattr(anchor, "inventado")
    assert anchor.with_fields({"subject": "C", "nope": "x"}).subject == "C"


def test_from_fields_is_positional():
    values = [f"v{i}" for i in range(len(ANCHOR_FIELDS))]
    anchor = ConsistencyAnchor.from_fields(*values)
    assert anchor.to_dict() == {k: f"v{i}" for i, k in enumerate(ANCHOR_FIELDS)}
    assert ConsistencyAnchor.from_fields("solo").setting == ""


def test_scene_delta_ignores_anchor_only_fields():
    delta = SceneDelta.from_mapping({"action": "corre", "setting": "otro sitio"})
    assert delta.action == "corre"
    assert not hasattr(delta, "setting")
    assert set(delta.to_dict()) == set(DELTA_FIELDS)


def test_compose_prompt_order_and_punctuation():
    anchor = ConsistencyAnchor(
        subject="Una mujer con abrigo rojo",
        setting="Una calle mojada",
        lighting="Luz de farolas",
        camera="Plano medio, 50mm",
    )
    prompt = compose_prompt(anchor, SceneDelta(action="caminando bajo la lluvia"))
    assert prompt.startswith("Una mujer con abrigo rojo,")
    assert prompt.endswith(".")
    assert prompt.index("caminando bajo la lluvia") < prompt.index("Una calle mojada")
    assert prompt.index("Una calle mojada") < prompt.index("Luz de farolas")


def test_compose_prompt_delta_override_wins():
    anchor = ConsistencyAnchor(subject="Un chico", camera="Plano medio")
    prompt = compose_prompt(anchor, SceneDelta(action="corre", camera="Plano general"))
    assert "Plano general" in prompt
    assert "Plano medio" not in prompt


def test_compose_prompt_dedupes_repeated_segments():
    anchor = ConsistencyAnchor(subject="Un chico con gorra")
    prompt = compose_prompt(anchor, SceneDelta(action="un chico con gorra"))
    assert prompt.count("Un chico con gorra") == 1


def test_compose_sequence_repeats_anchor_verbatim():
    anchor = ConsistencyAnchor(subject="Andrea", setting="Un taller", camera="35mm")
    prompts = compose_sequence(
        anchor,
        (SceneDelta(action="martillea"), SceneDelta(action="mide")),
    )
    assert len(prompts) == 2
    for prompt in prompts:
        assert "Andrea" in prompt and "Un taller" in prompt and "35mm" in prompt
    assert "martillea" in prompts[0]
    assert "mide" in prompts[1]


def test_degraded_prompts_repeat_goal_and_change():
    prompts = degraded_prompts("chico al amanecer", (SceneDelta(action="despierta"), SceneDelta(action="camina")))
    assert prompts[0] == "Chico al amanecer, despierta."
    assert prompts[1] == "Chico al amanecer, camina."


def test_handle_two_phase_repeats_anchor_in_every_prompt():
    seen: list[str] = []

    def run(system: str, user: str, tokens: int) -> str:
        seen.append(system)
        if len(seen) == 1:
            return _BIBLE_JSON
        return '{"action": "sentándose al borde de la cama", "lighting": "luz más cálida"}'

    result = handle(
        SequenceRequest(goal="chico bostezando al amanecer", deltas=("se sienta en la cama",)),
        run,
        service="Doble",
    )
    assert len(seen) == 2
    assert len(result.prompts) == 2
    for prompt in result.prompts:
        low = prompt.lower()
        assert "un chico de 17 años con pelo castaño revuelto" in low
        assert "camiseta gris de algodón" in prompt
    assert "bostezando sentado en la cama" in result.prompts[0]
    assert "sentándose al borde de la cama" in result.prompts[1]
    assert "luz más cálida" in result.prompts[1]
    assert "luz de amanecer" not in result.prompts[1]
    assert "Doble" in result.status


def test_handle_skips_bible_when_anchor_given():
    calls: list[str] = []

    def run(system: str, user: str, tokens: int) -> str:
        calls.append(system)
        return '{"action": "abre la persiana"}'

    result = handle(
        SequenceRequest(
            goal="la misma habitación",
            deltas=("abre la persiana",),
            anchor=ConsistencyAnchor(subject="El hombre", setting="Habitación de los 90"),
        ),
        run,
    )
    assert len(calls) == 1  # solo deltas, sin fase de biblia
    assert "El hombre" in result.prompts[0]
    assert "Habitación de los 90" in result.prompts[1]


def test_handle_delta_falls_back_to_user_text_without_backend():
    def run(system: str, user: str, tokens: int) -> str:
        raise RuntimeError("sin conexión")

    result = handle(
        SequenceRequest(goal="chico al amanecer", deltas=("se levanta", "se viste")),
        run,
    )
    assert len(result.prompts) == 3
    assert "se levanta" in result.prompts[1]
    assert "se viste" in result.prompts[2]
    assert "Sin biblia" in result.status
    # Sin biblia, el objetivo viaja entero en cada prompt (continuidad básica).
    for prompt in result.prompts:
        assert "chico al amanecer" in prompt.lower()


def test_handle_backend_down_after_first_delta_keeps_later_changes():
    state = {"n": 0}

    def run(system: str, user: str, tokens: int) -> str:
        state["n"] += 1
        if state["n"] == 1:
            return _BIBLE_JSON
        raise RuntimeError("boom")

    result = handle(
        SequenceRequest(goal="objetivo", deltas=("se sienta", "se levanta")),
        run,
    )
    assert result.deltas[1].action == "se sienta"
    assert result.deltas[2].action == "se levanta"


def test_handle_pure_text_gives_single_prompt():
    result = handle(
        SequenceRequest(goal="una calle vacía al amanecer"),
        lambda s, u, t: _BIBLE_JSON,
    )
    assert len(result.prompts) == 1
    assert result.prompts[0].startswith("Un chico")


def test_state_payload_and_recompose_roundtrip():
    result = handle(
        SequenceRequest(goal="chico al amanecer", deltas=("se sienta",)),
        lambda s, u, t: _BIBLE_JSON if "continuity sheet" in u.lower() or "anchor" in u else "",
    )
    payload = state_payload(result, ("chico al amanecer", "se sienta"))
    assert payload["anchor"]["subject"].startswith("un chico")
    assert len(payload["scenes"]) == 2

    edited = ConsistencyAnchor.from_mapping({**payload["anchor"], "wardrobe": "un pijama azul"})
    rebuilt = recompose(edited, "chico al amanecer", ("se sienta",), state=payload)
    assert "un pijama azul" in rebuilt.prompts[0]
    assert "un pijama azul" in rebuilt.prompts[1]
    assert rebuilt.prompts[0].endswith(".")


def test_recompose_empty_anchor_keeps_goal_only():
    rebuilt = recompose(ConsistencyAnchor(), "un faro en la costa", ())
    assert rebuilt.prompts == ("Un faro en la costa.",)


def test_recompose_manual_anchor_without_backend():
    rebuilt = recompose(
        ConsistencyAnchor(subject="Un pescador", setting="Un muelle"),
        "amanecer",
        ("recoge redes", "enciende un cigarro"),
    )
    # La escena 1 es el objetivo; las extra son los cambios.
    assert len(rebuilt.prompts) == 3
    assert "amanecer" in rebuilt.prompts[0]
    for prompt in rebuilt.prompts:
        assert "Un pescador" in prompt
        assert "Un muelle" in prompt


def test_continuity_notes_flags_duplicate_scenes():
    anchor = ConsistencyAnchor(subject="Un hombre en una cocina")
    deltas = (SceneDelta(action="corta cebolla"), SceneDelta(action="corta cebolla"))
    prompts = compose_sequence(anchor, deltas)
    notes = continuity_notes(anchor, deltas, prompts)
    assert any("casi idénticas" in n for n in notes)


def test_continuity_notes_flags_single_scene():
    notes = continuity_notes(ConsistencyAnchor(subject="A"), (SceneDelta(action="algo"),), ("A, algo.",))
    assert any("Solo hay una escena" in n for n in notes)


def test_continuity_notes_flags_missing_action():
    notes = continuity_notes(
        ConsistencyAnchor(subject="A"),
        (SceneDelta(action="algo"), SceneDelta()),
        ("A, algo.", "A."),
    )
    assert any("sin acción" in n for n in notes)


def test_continuity_notes_flags_anchor_over_budget():
    anchor = ConsistencyAnchor(subject=" ".join(f"palabra{i}" for i in range(30)))
    notes = continuity_notes(anchor, (SceneDelta(action="algo"),), ("x",), anchor_word_max=20)
    assert any("recorta campos" in n for n in notes)


def test_score_duplicate_range():
    assert score_duplicate("un gato negro", "un gato negro") == pytest.approx(1.0)
    assert score_duplicate("un gato negro", "una tormenta eléctrica") == 0.0


def test_prompt_builders_ask_for_json_and_respect_language():
    es = anchor_system(LANG_ES)
    en = anchor_system(LANG_EN)
    assert "JSON object" in es and "Spanish" in es
    assert "JSON object" in en and "English" in en
    assert "JSON object" in delta_system(LANG_ES)
    user = anchor_user_text("un chico al amanecer", word_min=40, word_max=90)
    assert "40" in user and "90" in user


def test_delta_user_text_includes_anchor_payload():
    text = delta_user_text(
        ConsistencyAnchor(subject="El hombre", setting="Habitación 90"),
        goal="amanecer",
        change="abre la persiana",
        index=3,
        total=4,
    )
    assert "Scene 3 of 4" in text
    assert "El hombre" in text and "Habitación 90" in text
    assert "abre la persiana" in text
    assert "do NOT rewrite" in text


def test_delta_user_text_survives_empty_change():
    text = delta_user_text(
        ConsistencyAnchor(subject="A"),
        goal="g",
        change="",
        index=2,
        total=2,
    )
    assert "(empty" in text


def test_defaults_are_reasonable():
    assert ANCHOR_WORDS_DEFAULT[0] < ANCHOR_WORDS_DEFAULT[1]
    assert SEQ_MIN_SCENES == 1
