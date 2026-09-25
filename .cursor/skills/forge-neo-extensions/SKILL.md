---
name: forge-neo-extensions
description: >-
  Desarrolla extensiones para Forge Neo (loader compatible A1111/Forge Classic).
  Usar al crear o modificar extensiones WebUI, scripts Gradio, install.py,
  metadata.ini, script_callbacks, Accordion AlwaysVisible, o al integrar
  algo en extensions/ / extensions-builtin/ de docker-neo.
---

# Extensiones Forge Neo

Forge Neo (rama `neo` de Forge Classic) carga extensiones igual que
[A1111 Developing extensions](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Developing-extensions)
y [Developing custom scripts](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Developing-custom-scripts).
Esta skill documenta solo lo que **sigue siendo válido** en Neo + convenciones del repo `docker-neo`.

API detallada de `scripts.Script` / hooks: skill **forge-neo-scripts**.

## Cuándo aplicar

- Nueva extensión o cambio en `extensions/` / `builtin-extensions/`
- UI en txt2img/img2img, pestaña propia, deps, orden de carga
- Dudas A1111 vs Neo

## Anatomía

Una extensión = subcarpeta en `extensions/` o `extensions-builtin/`.

| Pieza | Obligatorio | Rol |
|-------|-------------|-----|
| `scripts/*.py` | Sí (casi siempre) | Loader importa todo; `Script` y/o `script_callbacks` |
| Paquete hermano en raíz (p. ej. `forge_foo/`) | Recomendado | Lógica fuera del script; `sys.path` incluye la raíz |
| `install.py` | No | Deps vía `launch.run_pip` **antes** del WebUI |
| `metadata.ini` | No | `Name`, `Requires`, orden `Before`/`After` |
| `javascript/` | No | JS añadido a la página |
| `style.css` | No | CSS global de la extensión |
| `preload.py` | No | Antes de parsear CLI; `preload(parser)` puede añadir args |
| `localizations/` | No | JSON de i18n |

### Reglas de import

1. `sys.path` incluye la raíz de la extensión → `import forge_foo` OK.
2. Nombres de módulo **únicos** (o carpeta con nombre único): el módulo queda en el árbol global de Python.
3. Guardar `scripts.basedir()` **en import time**. Fuera de esa fase devuelve el root del WebUI.

```python
from modules import scripts
EXT_DIR = scripts.basedir()  # solo aquí
```

## Dos formas de UI (elige una)

| Forma | API | Cuándo |
|-------|-----|--------|
| Accordion en txt2img/img2img | `scripts.Script` + `show() → AlwaysVisible` + `InputAccordion` | Necesitas checkpoint/TE/stack del tab, o hook en `process`/`postprocess` |
| Pestaña top-level | `script_callbacks.on_ui_tabs` | Tool aparte (browser, IIB, vision pesada) |

Patrón Neo canónico (Moodboard, Identity Edit, Depth):

```python
from modules import scripts
from modules.ui_components import InputAccordion
import gradio as gr

class MyExt(scripts.Script):
    sorting_priority = 500  # orden relativo entre AlwaysVisible

    def title(self):
        return "My Ext"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enable:
            ...
        return [enable, ...]

    def process(self, p, enable, *args):
        if not enable:
            return
        ...
```

- Dropdown clásico: `show()` → `True`/`False`/`is_img2img`; lógica en `run(p, ...)`.
- AlwaysVisible: hooks `process` / `before_process` / `postprocess` (no `run`).
- `elem_id=self.elem_id("foo")` para IDs estables.

Pestaña propia:

```python
from modules import script_callbacks
import gradio as gr

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as demo:
        ...
    return [(demo, "Título UI", "elem_id_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)
```

## `install.py`

Proceso aparte, `PYTHONPATH` = root WebUI:

```python
import launch

if not launch.is_installed("paquete"):
    launch.run_pip("install paquete==1.2.3", "deps de MiExt")
```

**docker-neo:** arranca con `--skip-install` → `install.py` **no corre en runtime**.
Deps nuevas: hornearlas en la imagen (Dockerfile / `make build`) o no depender de ellas.
Extensiones seed en volumen: preferir cero deps extra.

## `metadata.ini` (resumen)

```ini
[Extension]
Name = mi-extension
Requires = otra-ext|alternativa

[scripts]
Before = otra-ext
After = yet-another
```

Keys case-sensitive; sections no. Detalle: [reference.md](reference.md).

## `preload.py`

```python
def preload(parser):
    parser.add_argument("--mi-ext-dir", type=str, default=None)
```

## Callbacks útiles

Registrar al final del script (import time):

| Callback | Uso |
|----------|-----|
| `on_ui_tabs` | Pestaña |
| `on_ui_settings` | Opciones en Settings |
| `on_app_started` | Montar FastAPI / init post-UI |
| `on_before_ui` / `on_after_component` | Parches de UI |

Lista ampliada: [reference.md](reference.md).

## docker-neo: dónde va el código

| Destino | Path en repo | Destino runtime | Cuándo |
|---------|--------------|-----------------|--------|
| Custom / seed | `extensions/<nombre>/` | `EXTENSIONS_PATH` (`/data/extensions`) | `make up` copia **solo si no existe** |
| Builtin imagen | `builtin-extensions/<nombre>/` | `extensions-builtin/` en imagen | Debe sobrevivir volumen vacío |

- No duplicar la misma extensión en ambos sitios.
- Install canónico fuera de Docker: **Extensions → Install from URL** o `git clone` en `extensions/`.
- Tras instalar: **Apply and restart UI**.

## Checklist nueva extensión

```
- [ ] Carpeta con nombre único (p. ej. sd-forge-...)
- [ ] scripts/*.py con Script AlwaysVisible o on_ui_tabs
- [ ] Lógica en paquete hermano, no todo en scripts/
- [ ] basedir() guardado en import si hace falta
- [ ] Sin install.py / sin deps nuevas (salvo hornear en imagen)
- [ ] metadata.ini si hay orden/Requires
- [ ] README: install + uso
- [ ] ¿Seed en extensions/ o builtin? (Ask first si no está claro)
```

## Anti-patrones

- Confiar en `install.py` dentro del contenedor docker-neo.
- Módulos genéricos (`utils.py`, `helpers.py`) en `scripts/` sin prefijo → colisiones.
- Usar `scripts.basedir()` en `process()` esperando la ruta de la extensión.
- Meter pesos/modelos en la extensión; van en `DATA_PATH/Models/…`.
- Asumir `modules.generation_parameters_copypaste`: en Forge Classic/Neo es `modules.infotext_utils`.

## Referencias

- Spec img2prompt (hallazgos loader): `docs/spec-forge-neo-img2prompt_25-09-2026.md` (este repo)
- Contexto docker-neo: seed `extensions/README.md`, builtins, ejemplos Moodboard / Identity Edit / Depth / IIB

Plantillas mínimas: [examples.md](examples.md).
Detalle metadata/callbacks/JS: [reference.md](reference.md).
