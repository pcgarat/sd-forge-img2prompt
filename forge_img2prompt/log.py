"""Log visible en el terminal de Forge Neo (stdout + flush)."""

from __future__ import annotations

import sys

_PREFIX = "[img2prompt]"


def log(msg: str) -> None:
    """Escribe una línea al stdout de Forge; flush inmediato para ver el progreso."""
    print(f"{_PREFIX} {msg}", flush=True, file=sys.stdout)
