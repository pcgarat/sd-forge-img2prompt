from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from forge_img2prompt.log import log
from forge_img2prompt.vl_catalog import DEFAULT_HF_ID, default_local_dir, is_local_ready

ProgressCb = Callable[[float, str], None]
PlanCb = Callable[[str], None]
StatusCb = Callable[[str], None]


@dataclass(frozen=True)
class DownloadItem:
    filename: str
    size: int

    @property
    def size_label(self) -> str:
        if self.size <= 0:
            return "?"
        mb = self.size / (1024 * 1024)
        if mb >= 1024:
            return f"{mb / 1024:.2f} GB"
        return f"{mb:.0f} MB"


@dataclass(frozen=True)
class DownloadEvent:
    """Paso observable de la descarga (para Gradio yield o callbacks)."""

    status: str
    plan: str


def _fmt_total(items: list[DownloadItem]) -> str:
    total = sum(i.size for i in items if i.size > 0)
    if total <= 0:
        return "?"
    return f"{total / (1024**3):.2f} GB"


def _bytes_total(items: list[DownloadItem]) -> int:
    """Peso total para progreso: suma de tamaños conocidos, o nº de ficheros."""
    known = sum(i.size for i in items if i.size > 0)
    if known > 0:
        return known
    return max(len(items), 1)


def _item_weight(item: DownloadItem, items: list[DownloadItem]) -> int:
    if item.size > 0:
        return item.size
    known = sum(i.size for i in items if i.size > 0)
    if known > 0:
        return max(known // max(len(items), 1), 1)
    return 1


def list_repo_files(repo_id: str = DEFAULT_HF_ID) -> list[DownloadItem]:
    from huggingface_hub import model_info

    info = model_info(repo_id, files_metadata=True)
    items: list[DownloadItem] = []
    skip_suffixes = (".png", ".jpg", ".jpeg", ".gif", ".webp")
    for sib in info.siblings or []:
        name = sib.rfilename
        if name.endswith(skip_suffixes) or name.startswith(".") or name.upper().startswith("README"):
            continue
        size = int(getattr(sib, "size", None) or 0)
        items.append(DownloadItem(filename=name, size=size))
    items.sort(key=lambda x: (-x.size, x.filename))
    return items


def format_download_plan(
    items: list[DownloadItem],
    local_dir: Path,
    *,
    repo_id: str = DEFAULT_HF_ID,
    done: set[str] | None = None,
    current: str | None = None,
) -> str:
    done = done or set()
    lines = [
        f"**Descarga** `{repo_id}` → `{local_dir}`",
        f"Total estimado: **{_fmt_total(items)}** · {len(items)} fichero(s)",
        "",
    ]
    for i, it in enumerate(items, 1):
        if it.filename in done:
            mark = "✅"
        elif current == it.filename:
            mark = "⬇️"
        else:
            mark = "⬜"
        lines.append(f"{mark} {i}. `{it.filename}` — {it.size_label}")
    return "\n".join(lines)


def _file_ready(local_dir: Path, filename: str) -> bool:
    target = local_dir / filename
    return target.is_file() and target.stat().st_size > 0


@contextmanager
def _bridge_hf_tqdm(on_bytes: Callable[[int, int | None], None]) -> Iterator[None]:
    """Parchea el tqdm de huggingface_hub para reportar bytes (también en no-TTY)."""
    import huggingface_hub.utils.tqdm as hf_tqdm_mod
    from tqdm.auto import tqdm as TqdmBase

    class BridgedTqdm(TqdmBase):
        def __init__(self, *args, **kwargs):
            kwargs.pop("name", None)
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)
            self._emit()

        def update(self, n=1):
            result = super().update(n)
            self._emit()
            return result

        def refresh(self, **kwargs):
            result = super().refresh(**kwargs)
            self._emit()
            return result

        def _emit(self) -> None:
            total = int(self.total) if self.total else None
            on_bytes(int(self.n or 0), total)

    previous = hf_tqdm_mod.tqdm
    hf_tqdm_mod.tqdm = BridgedTqdm  # type: ignore[misc, assignment]
    try:
        yield
    finally:
        hf_tqdm_mod.tqdm = previous  # type: ignore[misc]


def _throttled_file_progress(
    *,
    filename: str,
    size_label: str,
    progress: ProgressCb | None,
    frac_start: float,
    frac_end: float,
    min_interval_s: float = 0.2,
) -> Callable[[int, int | None], None]:
    """Mapea bytes del fichero actual → [frac_start, frac_end] con throttle."""
    last_t = 0.0
    last_frac = -1.0

    def on_bytes(n: int, total: int | None) -> None:
        nonlocal last_t, last_frac
        if progress is None:
            return
        denom = total if total and total > 0 else None
        if denom is None:
            file_frac = 0.0 if n <= 0 else 0.5
        else:
            file_frac = max(0.0, min(1.0, n / denom))
        frac = frac_start + (frac_end - frac_start) * file_frac
        now = time.monotonic()
        if frac < 0.999 and (now - last_t) < min_interval_s and abs(frac - last_frac) < 0.005:
            return
        last_t = now
        last_frac = frac
        if denom and denom > 0:
            pct = 100.0 * n / denom
            mb = n / (1024 * 1024)
            desc = f"{filename} ({size_label}) · {pct:.0f}% · {mb:.0f} MB"
        else:
            desc = f"{filename} ({size_label})"
        progress(max(0.0, min(1.0, frac)), desc)

    return on_bytes


def download_item(
    *,
    repo_id: str,
    item: DownloadItem,
    local_dir: Path,
    progress: ProgressCb | None = None,
    frac_start: float = 0.0,
    frac_end: float = 1.0,
) -> None:
    """Descarga un fichero del repo HF reportando progreso por bytes."""
    import inspect

    from huggingface_hub import hf_hub_download

    on_bytes = _throttled_file_progress(
        filename=item.filename,
        size_label=item.size_label,
        progress=progress,
        frac_start=frac_start,
        frac_end=frac_end,
    )
    if progress:
        progress(frac_start, f"{item.filename} ({item.size_label})")

    kwargs: dict = {
        "repo_id": repo_id,
        "filename": item.filename,
        "local_dir": str(local_dir),
    }
    params = inspect.signature(hf_hub_download).parameters
    if "local_dir_use_symlinks" in params:
        kwargs["local_dir_use_symlinks"] = False
    if "resume_download" in params:
        kwargs["resume_download"] = True

    with _bridge_hf_tqdm(on_bytes):
        hf_hub_download(**kwargs)
    if progress:
        progress(frac_end, f"{item.filename} ({item.size_label})")


def iter_model_download(
    *,
    repo_id: str = DEFAULT_HF_ID,
    local_dir: Path | None = None,
    progress: ProgressCb | None = None,
) -> Iterator[DownloadEvent]:
    """Único bucle de descarga. Emite (status, plan) en hitos; `progress` lleva la barra."""
    dest = Path(local_dir) if local_dir else default_local_dir()
    dest.mkdir(parents=True, exist_ok=True)

    def report(frac: float, desc: str) -> None:
        log(desc)
        if progress:
            progress(max(0.0, min(1.0, frac)), desc)

    if is_local_ready(dest):
        report(1.0, "Modelo ya descargado")
        yield DownloadEvent(status="Modelo ya descargado", plan=f"Modelo listo en `{dest}`.")
        return

    report(0.02, "Consultando lista de ficheros en Hugging Face…")
    items = list_repo_files(repo_id)
    if not items:
        raise RuntimeError(f"El repo {repo_id} no listó ficheros descargables")

    done: set[str] = {it.filename for it in items if _file_ready(dest, it.filename)}
    pending = [it for it in items if it.filename not in done]
    total_w = _bytes_total(items)

    def plan(current: str | None = None) -> str:
        return format_download_plan(items, dest, repo_id=repo_id, done=done, current=current)

    status = f"Plan: {len(items)} ficheros · {_fmt_total(items)} · pendientes={len(pending)}"
    report(0.05, status)
    yield DownloadEvent(status=status, plan=plan())

    if not pending and is_local_ready(dest):
        report(1.0, "Nada pendiente")
        yield DownloadEvent(status="Nada pendiente", plan=f"Modelo listo en `{dest}`.")
        return

    bytes_done = sum(_item_weight(it, items) for it in items if it.filename in done)

    for it in items:
        if it.filename in done:
            continue
        weight = _item_weight(it, items)
        frac_start = 0.05 + 0.9 * (bytes_done / total_w)
        frac_end = 0.05 + 0.9 * ((bytes_done + weight) / total_w)
        status = f"⬇️ `{it.filename}` ({it.size_label})"
        report(frac_start, status)
        yield DownloadEvent(status=status, plan=plan(current=it.filename))
        download_item(
            repo_id=repo_id,
            item=it,
            local_dir=dest,
            progress=progress,
            frac_start=frac_start,
            frac_end=frac_end,
        )
        done.add(it.filename)
        bytes_done += weight
        status = f"✅ `{it.filename}`"
        yield DownloadEvent(status=status, plan=plan())

    if not is_local_ready(dest):
        raise RuntimeError(
            f"Descarga incompleta en {dest}: falta config.json o pesos .safetensors"
        )

    report(1.0, f"Descarga completa → {dest}")
    yield DownloadEvent(
        status=f"Descarga completa → `{dest}`",
        plan=f"Modelo listo en `{dest}`.",
    )


def ensure_model_downloaded(
    *,
    repo_id: str = DEFAULT_HF_ID,
    local_dir: Path | None = None,
    progress: ProgressCb | None = None,
    plan_update: PlanCb | None = None,
    status_update: StatusCb | None = None,
) -> Path:
    """Wrapper síncrono sobre `iter_model_download` (providers / callers no-Gradio)."""
    dest = Path(local_dir) if local_dir else default_local_dir()
    for event in iter_model_download(repo_id=repo_id, local_dir=dest, progress=progress):
        if status_update:
            status_update(event.status)
        if plan_update:
            plan_update(event.plan)
    return dest


def download_plan_markdown(
    local_dir: Path | None = None,
    *,
    repo_id: str = DEFAULT_HF_ID,
) -> str:
    dest = Path(local_dir) if local_dir else default_local_dir()
    if is_local_ready(dest):
        return f"Modelo listo en `{dest}`."
    try:
        return format_download_plan(list_repo_files(repo_id), dest, repo_id=repo_id)
    except Exception as exc:  # noqa: BLE001
        return (
            f"Modelo **aún no descargado** → `{dest}`\n"
            f"Repo: `{repo_id}`\n\n"
            f"(No se pudo listar HF ahora: {exc}. Al pulsar Generate se reintentará.)"
        )
