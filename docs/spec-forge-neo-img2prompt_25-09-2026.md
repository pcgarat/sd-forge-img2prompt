# Última modificación: 2026-09-26

# Spec: Forge Neo — Image → Prompt (Krea 2 / Klein 9B v1)

## Forge Neo — cómo se crea una extensión (hallazgos)

Mecanismo heredado de A1111 / Forge Classic (`Haoming02/sd-webui-forge-classic` rama `neo`):

| Pieza | Rol |
|-------|-----|
| Carpeta en `extensions/` o `extensions-builtin/` | Unidad instalable; el nombre de carpeta es el id por defecto |
| `scripts/*.py` | El loader importa todo lo que hay aquí. Suele contener una subclase de `modules.scripts.Script` **o** registrar `script_callbacks` |
| Paquete hermano en la raíz (p. ej. `forge_krea2_depth/`) | Lógica fuera del script; patrón usado por Depth ControlNet en este repo |
| `install.py` (raíz) | Se ejecuta al arranque vía `run_extension_installer` **salvo** `--skip-install` |
| `metadata.ini` | Opcional: `Name`, orden `Before`/`After` entre extensiones |
| `javascript/`, `style.css`, `preload.py` | Opcionales |

**Dos formas de UI:**

1. **`scripts.Script` + `show() → AlwaysVisible`** — accordion dentro de txt2img/img2img (Moodboard, Identity Edit, Depth). Ideal si necesitas el **stack ya seleccionado** en esa pestaña.
2. **`script_callbacks.on_ui_tabs`** — pestaña top-level propia (p. ej. [Adeliox/forge-neo-image2prompt](https://github.com/Adeliox/forge-neo-image2prompt)). Mejor para tools pesados de VRAM; el “send to prompt” suele hacerse con JS sobre `#txt2img_prompt textarea`.

**Instalación en la práctica:**

- UI: Extensions → Install from URL (repo git) → Apply and restart UI.
- CLI: `git clone <url> $EXTENSIONS_PATH/<nombre>`.
- En `docker-neo`: `make seed-extensions` / `make up` copia `repo/extensions/*` → `EXTENSIONS_PATH` **solo si no existe**; la imagen arranca con **`--skip-install`**, así que deps de `install.py` **no** se instalan en runtime — hay que hornearlas en la imagen o no depender de ellas.

**Decisión v1 (revisada con skill forge-neo-extensions):** forma (2) **`on_ui_tabs`**, sin `install.py`, sin deps extra. Motivo: es una *tool* (no hook `process`); el stack se lee de `shared` al generar; el send-to idiomático en Neo es `modules.infotext_utils.register_paste_params_button`. AlwaysVisible queda como alternativa solo si se prioriza cero cambio de pestaña. No forkear Adeliox (VL propio + tag-soup; distinto objetivo).

---

## Objective

Extensión de **Forge Neo** (formato WebUI estándar) que, a partir de una imagen (upload o pegado desde portapapeles en el control Gradio), produce un **prompt en prosa** listo para pegar en txt2img, afinado al stack **Krea 2** o **FLUX.2 Klein 9B** actualmente seleccionado (checkpoint + text encoder + variante).

**Instalación:** igual que el resto de extensiones — **Extensions → Install from URL**, o `git clone` en `extensions/` / `EXTENSIONS_PATH`. Tras instalar: **Apply and restart UI**. No requiere pasos Docker especiales ni ir en la imagen como builtin. En `docker-neo` local: volumen `IMG2PROMPT_EXT_PATH` (ver compose).

**Usuario:** operador local de Forge Neo (incl. `docker-neo`).

**Por qué:** dejar de adaptar a mano captions genéricos a Qwen3-VL / Krea 2 / Klein.

**Éxito v1:** instalable por el flujo estándar + UI usable + detección de stack Krea 2 / Klein + escritura del prompt vía `PromptProvider` inyectable; el provider real (VL/API) no existe aún — stub que parte de una descripción opcional del usuario.

**Fuera de alcance v1:** backend VL/API, auto-aplicar sampler/steps/CFG (solo *hints* informativos), app desktop aparte, lógica basada en VAE, hornear la extensión en `builtin-extensions/` / imagen Docker.

---

## Tech Stack

| Pieza | Elección |
|-------|----------|
| Host | Forge Neo / A1111-compatible extension loader |
| Lenguaje | Python 3 (el del runtime Forge) |
| UI | Gradio + `script_callbacks.on_ui_tabs` |
| Layout | Extensión WebUI clásica en repo dedicado |
| Distribución | Repo propio [`pcgarat/sd-forge-img2prompt`](https://github.com/pcgarat/sd-forge-img2prompt) — Install from URL / `git clone` en `EXTENSIONS_PATH` |
| Persistencia | Ninguna en v1 (sin DB, sin files de caché obligatorios) |
| Backend visión/LLM | **No** en v1; interfaz preparada (`PromptProvider`) |
| `install.py` | **No** en v1 — compatible con docker-neo `--skip-install` |

Referencia de prompting Krea 2 / FLUX prosa: [`docs/guia_img_prompts_23-09-2026.md`](guia_img_prompts_23-09-2026.md) (principios de prosa natural; perfil Krea 2 prioriza detalle de composición/luz/materiales y texto entre comillas). Origen: `docker-neo/guia_img_prompts.md`.

---

## Commands

**Instalar (Forge Neo / host o contenedor):**

```bash
# Opción A — UI: Extensions → Install from URL
# https://github.com/pcgarat/sd-forge-img2prompt

# Opción B — CLI
git clone https://github.com/pcgarat/sd-forge-img2prompt.git "$EXTENSIONS_PATH/sd-forge-img2prompt"
# Luego Apply and restart UI (o make restart)
```

**Desarrollo:** en el repo de la extensión (no como copia permanente en `docker-neo/extensions/`).

```bash
cd /path/to/sd-forge-img2prompt
python -m pytest tests -q
```

---

## Project Structure

```text
# https://github.com/pcgarat/sd-forge-img2prompt
sd-forge-img2prompt/
  README.md
  metadata.ini
  LICENSE
  docs/
    spec-forge-neo-img2prompt_25-09-2026.md
    guia_img_prompts_23-09-2026.md
  tasks/
    plan.md
    todo.md
  scripts/
    img2prompt.py             # on_ui_tabs + paste_params
  forge_img2prompt/
    __init__.py
    stack.py
    provider.py               # Protocol + StubProvider + DetailRequest + append_detail
    mask_crop.py              # ImageEditor brush → bbox crop
    vl_provider.py            # Qwen VL generate + detail
  tests/
    test_stack.py
    test_stub_provider.py
    test_mask_crop.py
    test_append_detail.py
```

---

## Feature: máscara → detalle (post-v1)

Flujo: **Generate** (caption global) → pintar zona en `gr.ImageEditor` (`layers=False`) → **Añadir detalle**.

- Crop del bounding box de la capa de pincel (`mask_crop.crop_from_editor`).
- VL con prompt de detalle (`DetailRequest` / `CompositeProvider.detail`), `max_new_tokens≈120`.
- El fragmento va a **Prompt de la zona**; el prompt general no se modifica.
- Validaciones: prompt previo + máscara no vacía; si falla, status claro y prompt intacto.

Fuera de alcance de esta feature: rewrite completo, imagen atenuada, inpaint/ControlNet, multi-máscaras.

---

## Code Style

- Extensión como las Neo con pestaña propia: `script_callbacks.on_ui_tabs`, `analytics_enabled=False`.
- Nombres en inglés en código; UI y docs de usuario en español.
- Sin lógica de prompt acoplada a Gradio: UI → `PromptRequest` / `DetailRequest` → provider → textbox + paste_params.
- Send-to: `modules.infotext_utils` (no `generation_parameters_copypaste`). Guardar `scripts.basedir()` en import si se necesitan paths.

Ejemplo de contrato (ilustrativo):

```python
from typing import Protocol
from dataclasses import dataclass
from PIL import Image

@dataclass(frozen=True)
class StackInfo:
    family: str          # "krea2" | "unknown"
    variant: str         # "turbo" | "raw" | "unknown"
    checkpoint: str
    text_encoder: str
    is_supported: bool

@dataclass(frozen=True)
class PromptRequest:
    image: Image.Image | None
    user_notes: str
    stack: StackInfo

@dataclass(frozen=True)
class PromptResult:
    prompt: str
    negative_hint: str   # vacío o aviso si distilled/turbo
    sampler_hints: str   # texto informativo, no aplica settings

class PromptProvider(Protocol):
    def generate(self, request: PromptRequest) -> PromptResult: ...
```

Stub v1: si hay `user_notes`, reescribe/envuelve con plantilla Krea 2; si no, genera un prompt mínimo pidiendo completar + metadatos de stack en un comentario HTML de UI (no en el prompt final salvo que aporten).

---

## Testing Strategy

| Nivel | Qué | Dónde |
|-------|-----|-------|
| Unit | Detección `family`/`variant` a partir de nombres de checkpoint/TE | `tests/test_stack.py` |
| Unit | Stub genera prosa no-tag-soup y respeta perfil Krea 2 | `tests/test_stub_provider.py`, `test_krea2_profile.py` |
| Manual | Subir/pegar imagen, notes, Generate → prompt en txt2img con checkpoint Krea 2 | Smoke en Forge Neo GPU |

**No** tests e2e de WebUI en v1 (Ask first si se quieren).

Cobertura: no hay umbral numérico; sí deben pasar todos los unit tests del paquete antes de dar v1 por hecha.

---

## Boundaries

**Always**
- Detectar stack antes de generar; si no es Krea 2, mostrar aviso claro y no fingir optimización.
- Mantener `PromptProvider` desacoplado (un solo sitio para enchufar backend).
- Prompts en prosa natural; no booru/tag spam.
- Fecha en docs nuevos según convención del repo.

**Ask first**
- Target `make` en docker-neo para clonar/actualizar `sd-forge-img2prompt`.
- Meterla en `builtin-extensions/` (imagen) — por defecto **no**.
- Añadir dependencias Python / `install.py` no vacío.
- Auto-aplicar steps/CFG/sampler (v1 solo hints).
- Soporte Klein u otras familias.
- Implementar provider real (local VL o API).
- AlwaysVisible en lugar de pestaña (si se prioriza cero cambio de tab).

**Never**
- Enviar imágenes a APIs sin decisión explícita.
- Acoplar prompts al VAE.
- Exigir un procedimiento de install distinto al de cualquier otra extensión WebUI.
- Sobrescribir extensiones ajenas o el patch de backend Krea2.
- Incluir secretos / API keys en el repo.

---

## Success Criteria

1. Se puede instalar con **Extensions → Install from URL** (o `git clone` en `extensions/`) y aparece tras **Apply and restart UI**, sin pasos Docker especiales.
2. Layout válido de extensión WebUI (`scripts/` + paquete; sin deps extra en v1).
3. Pestaña top-level **Image → Prompt** visible tras Apply and restart UI.
4. Acepta imagen por upload y por pegado en el control de imagen Gradio.
5. Campo opcional **notas / descripción** del usuario.
6. Con checkpoint/TE reconocibles como Krea 2, Generate produce prosa y Send a txt2img/img2img la vuelca vía `infotext_utils` (o copy fallback).
7. Muestra hints de sampler (Turbo ≈ 8 steps / CFG bajo; RAW ≈ más steps / CFG ~4.5; Klein distilled 4 steps CFG=1) sin mutar settings.
8. Si el stack no es Krea 2 ni Klein 9B, UI avisa y no promete prompt “optimizado”.
9. Existe `PromptProvider` + stub; cambiar de provider no exige reescribir la UI.
10. `pytest` de `tests/` pasa sin levantar Forge.
11. README documenta Install from URL, uso v1 y el punto de extensión del provider.
12. Smoke en Forge Neo: Generate → Send to txt2img con preset Krea 2.

---

## Open Questions

Pendientes diferidos (post-v1):

- ¿Provider local reutilizando Qwen3-VL cargado vs API?
- ¿Auto-aplicar preset `krea2` (sampler/steps/CFG)?
- Target `make` en docker-neo para clonar/actualizar la extensión (Ask first).

---

## Assumptions (revisión humana)

1. Código canónico en **https://github.com/pcgarat/sd-forge-img2prompt** (no seed en `docker-neo/extensions/`).
2. Extensión WebUI estándar (Install from URL / clone en `extensions/`).
3. “Sin backend” = sin servicio/VL/API; el stub **sí** puede transformar `user_notes` + perfil en prosa usable.
4. Clipboard = pegar en el `gr.Image` del navegador.
5. Detección por heurística de nombres de checkpoint/TE (Krea 2 + Klein 9B).
6. Nombre de carpeta: `sd-forge-img2prompt`.
7. Klein 9B entra en v1 (perfil + hints); no solo Krea 2.
