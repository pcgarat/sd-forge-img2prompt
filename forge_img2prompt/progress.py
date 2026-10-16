"""Progreso opcional: un receptor de hitos que nunca se evalúa como verdad.

``gr.Progress`` de Gradio es un objeto llamable que además implementa
``__len__`` sobre las barras activas. Mientras no hay ninguna barra creada,
``len(progress)`` lanza ``IndexError``, así que un ``if progress:`` en mitad de
una secuencia, de un caption o de una descarga aborta el trabajo real.

Para que ninguna capa tenga que recordarlo, las entradas públicas aceptan
``Progress `` | callback | None`` y lo normalizan una sola vez con
:meth:`Progress.wrap`; a partir de ahí siempre se llama a ``notify`` sobre un
objeto, aunque esté vacío.
"""

from __future__ import annotations

from typing import Callable

ProgressCb = Callable[[float, str], None]


class Progress:
    """Receptor de hitos (fracción + descripción). Vacío = no hay UI escuchando."""

    __slots__ = ("_sink",)

    def __init__(self, sink: ProgressCb | None = None) -> None:
        self._sink = sink

    @classmethod
    def wrap(cls, target: ProgressLike) -> Progress:
        """Normaliza ``Progress``, callback crudo o ``None`` a un receptor usable."""
        if isinstance(target, cls):
            return target
        return cls(target)

    def notify(self, frac: float, desc: str) -> None:
        if self._sink is None:
            return
        self._sink(min(max(frac, 0.0), 1.0), desc)


ProgressLike = Progress | ProgressCb | None
