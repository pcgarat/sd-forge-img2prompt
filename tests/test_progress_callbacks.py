"""Contrato del callback de progreso: nunca se evalúa como booleano.

``gr.Progress`` es un objeto llamable que implementa ``__len__`` sobre las
barras activas. Un ``if progress:`` con la lista vacía lanza ``IndexError`` en
mitad de una secuencia o de una descarga. Estos tests fijan el contrato en las
tres capas que reciben el callback desde la UI: motor de secuencia, proveedores
remotos y descarga de modelos.
"""

from __future__ import annotations

import pytest

from forge_img2prompt.progress import Progress
from forge_img2prompt.provider import PromptRequest
from forge_img2prompt.stack import detect_stack


class _GradioLikeProgress:
    """Reproduce ``gr.Progress``: llamable y con ``__len__`` que revienta."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, str]] = []

    def __len__(self) -> int:
        raise IndexError("list index out of range")

    def __call__(self, frac: float, desc: str | None = None, **kwargs) -> None:
        self.calls.append((frac, desc or ""))


def _krea():
    return detect_stack("krea2_turbo.safetensors", "qwen3vl_4b.safetensors")


def test_progress_wrap_accepts_none_callback_and_itself():
    empty = Progress.wrap(None)
    empty.notify(0.5, "nadie escucha")  # no-op, no revienta

    seen: list[tuple[float, str]] = []
    raw = Progress.wrap(lambda f, d: seen.append((f, d)))
    assert isinstance(raw, Progress)
    raw.notify(0.25, "hito")

    assert seen == [(0.25, "hito")]
    assert Progress.wrap(raw) is raw


def test_progress_notify_clamps_fraction():
    seen: list[tuple[float, str]] = []
    sink = Progress(lambda f, d: seen.append((f, d)))

    sink.notify(-0.5, "abajo")
    sink.notify(1.5, "arriba")

    assert seen == [(0.0, "abajo"), (1.0, "arriba")]


def test_ollama_generate_reports_progress_without_bool_check():
    from forge_img2prompt.ollama_vl import OllamaVLProvider
    from forge_img2prompt.vl_catalog import _ollama_choice_for

    provider = OllamaVLProvider()
    provider._run = lambda *a, **k: "un gato naranja en un tejado mojado"  # type: ignore[method-assign]
    progress = _GradioLikeProgress()
    choice = _ollama_choice_for("qwen3-vl:8b-instruct")

    result = provider.generate(
        PromptRequest(None, "un gato en un tejado", _krea(), "es"),
        choice,
        progress=progress,
    )

    assert result.prompt
    assert progress.calls, "el callback recibió al menos un hito"
    assert all(desc for _, desc in progress.calls)


def test_nan_sequence_reports_progress_without_bool_check():
    from forge_img2prompt.ollama_vl import NanVLProvider
    from forge_img2prompt.sequence import SequenceRequest
    from forge_img2prompt.vl_catalog import _nan_choice_for

    bible = (
        '{"anchor": {"subject": "un chico de 17"}, "scene_1": {"action": "bosteza"}}'
    )
    provider = NanVLProvider()
    provider._run = lambda *a, **k: bible  # type: ignore[method-assign]
    progress = _GradioLikeProgress()
    choice = _nan_choice_for("glm5.3-flash")

    result = provider.sequence(
        SequenceRequest(goal="chico al amanecer", deltas=("se sienta",)),
        choice,
        progress=progress,
    )

    assert len(result.prompts) == 2
    assert [desc for _, desc in progress.calls] == [
        "Biblia de continuidad…",
        "Escena 2…",
    ]


def test_iter_model_download_reports_progress_without_bool_check(tmp_path):
    from forge_img2prompt.vl_download import iter_model_download

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    progress = _GradioLikeProgress()

    events = list(
        iter_model_download(
            repo_id="unused/repo", local_dir=tmp_path, progress=progress
        )
    )

    assert len(events) == 1
    assert progress.calls == [(1.0, "Modelo ya descargado")]


@pytest.mark.parametrize("callback", [None, _GradioLikeProgress()])
def test_progress_callback_is_optional_everywhere(callback):
    from forge_img2prompt.sequence import SequenceRequest, handle

    result = handle(
        SequenceRequest(goal="una calle vacía al amanecer"),
        lambda s, u, t: (
            '{"anchor": {"subject": "un faro"}, "scene_1": {"action": "gira"}}'
        ),
        progress=callback,
    )
    assert result.prompts
