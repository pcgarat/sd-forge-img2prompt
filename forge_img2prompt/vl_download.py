from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from forge_img2prompt.log import log
from forge_img2prompt.vl_catalog import DEFAULT_HF_ID, default_local_dir, is_local_ready

ProgressCb = Callable[[float, str], None]
PlanCb = Callable[[str], None]


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


def _fmt_total(items: list[DownloadItem]) -> str:
    total = sum(i.size for i in items if i.size > 0)
    if total <= 0:
        return "?"
    return f"{total / (1024**3):.2f} GB"


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


def ensure_model_downloaded(
    *,
    repo_id: str = DEFAULT_HF_ID,
    local_dir: Path | None = None,
    progress: ProgressCb | None = None,
    plan_update: PlanCb | None = None,
) -> Path:
    """Descarga fichero a fichero a TextEncoders con progreso y checklist."""
    dest = Path(local_dir) if local_dir else default_local_dir()
    dest.mkdir(parents=True, exist_ok=True)

    def report(frac: float, desc: str) -> None:
        log(desc)
        if progress:
            progress(max(0.0, min(1.0, frac)), desc)

    if is_local_ready(dest):
        report(1.0, "Modelo ya descargado")
        if plan_update:
            plan_update(f"Modelo listo en `{dest}`.")
        return dest

    report(0.02, "Consultando lista de ficheros en Hugging Face…")
    items = list_repo_files(repo_id)
    if not items:
        raise RuntimeError(f"El repo {repo_id} no listó ficheros descargables")

    done: set[str] = {it.filename for it in items if _file_ready(dest, it.filename)}
    pending = [it for it in items if it.filename not in done]

    def push_plan(current: str | None = None) -> None:
        if plan_update:
            plan_update(format_download_plan(items, dest, repo_id=repo_id, done=done, current=current))

    push_plan()
    report(0.05, f"Plan: {len(items)} ficheros · {_fmt_total(items)} · pendientes={len(pending)}")

    if not pending and is_local_ready(dest):
        report(1.0, "Nada pendiente")
        return dest

    from huggingface_hub import hf_hub_download

    total = max(len(items), 1)
    for it in items:
        if it.filename in done:
            continue
        frac = 0.05 + 0.9 * (len(done) / total)
        report(frac, f"[{len(done)+1}/{total}] {it.filename} ({it.size_label})")
        push_plan(current=it.filename)
        hf_hub_download(
            repo_id=repo_id,
            filename=it.filename,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        done.add(it.filename)
        push_plan()

    if not is_local_ready(dest):
        raise RuntimeError(
            f"Descarga incompleta en {dest}: falta config.json o pesos .safetensors"
        )

    report(1.0, f"Descarga completa → {dest}")
    if plan_update:
        plan_update(f"Modelo listo en `{dest}`.")
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
