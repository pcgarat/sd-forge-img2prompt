# Referencia — extensiones Forge Neo / A1111

Última modificación: 2026-09-25

Fuente base: [Developing extensions (A1111 wiki)](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Developing-extensions).
Validado contra loader Neo y extensiones de `docker-neo`.

## Ciclo de carga

1. `preload.py` (si existe) — antes de CLI
2. `install.py` (si existe y no `--skip-install`) — proceso aparte
3. Import de `scripts/*.py` (y equivalentes en builtins)
4. `javascript/*` inyectado en la página
5. `style.css` de la raíz de la extensión
6. `localizations/*` → Settings (mismo nombre = uno pisa al otro, no merge)

## `scripts.basedir()`

```python
# scripts/example.py  (durante import)
from modules import scripts
EXT_DIR = scripts.basedir()  # .../extensions/mi_ext

class Example(scripts.Script):
    def show(self, is_img2img):
        scripts.basedir()  # → root WebUI, NO la extensión
        return scripts.AlwaysVisible
```

Usar `EXT_DIR` guardado para paths a assets, DB, embeddings, etc.

## Métodos de `scripts.Script`

| Método | Rol |
|--------|-----|
| `title()` | Nombre UI |
| `show(is_img2img)` | `True`/`False`/`is_img2img` → dropdown; `AlwaysVisible` → siempre |
| `ui(is_img2img)` | Componentes Gradio; return lista → args de `run`/`process` |
| `run(p, *ui_args)` | Dropdown: sustituye/envuelve generación |
| `process(p, *ui_args)` | AlwaysVisible: antes/durante el pipeline |
| `before_process` / `postprocess` | Hooks adicionales AlwaysVisible |
| `elem_id(name)` | Prefijo estable para `elem_id` Gradio |
| `sorting_priority` | Orden entre scripts AlwaysVisible (entero) |

Clase base: `modules/scripts.py` del WebUI.

## `install.py` + `launch`

```python
import launch

if not launch.is_installed("aitextgen"):
    launch.run_pip("install aitextgen==0.6.0", "requirements for MagicPrompt")
```

Patrón Neo (evitar romper deps del host):

```python
launch.run_pip("install --no-deps easy-dwpose==1.0.2", "easy-dwpose (no-deps)")
```

Con `--skip-install` (docker-neo): este archivo no se ejecuta.

## `metadata.ini`

Keys **case-sensitive**. Canonical `Name`: solo `a-z0-9_-`.

```ini
[Extension]
Name = demo-extension
Requires = ext-a|ext-b, ext-c

[scripts]
Requires = another-extension
Before = another-extension
After = yet-another-extension

[scripts/another-script.py]
Requires = another-extension, yet-another-extension/another-script.py
Before = xyz_grid.py
After = yet-another-extension/another-script.py

[callbacks/mi-ext/mi_script.py/ui_settings]
Before = otra/otra.py/ui_settings
After = hypertile/hypertile_script.py/ui_settings
```

- `Requires` con `|`: satisfecho por cualquiera.
- Secciones `[scripts]`, `[javascript]`, `[localization]` controlan orden del folder.
- `[callbacks/…]` reordena callbacks concretos (ids visibles en Settings).

## Embeddings extra

```python
import os
from modules import scripts, sd_hijack

path = os.path.join(EXT_DIR, "embeddings")  # EXT_DIR de basedir() en import
sd_hijack.model_hijack.embedding_db.add_embedding_dir(path)
```

## JavaScript / CSS

- Archivos en `javascript/` se cargan en la página.
- API habitual del frontend WebUI: `gradioApp()`, `updateInput(el)`.
- Selectores de prompt: `#txt2img_prompt textarea`, `#img2img_prompt textarea`.
- `style.css` en la raíz de la extensión (no dentro de `scripts/`).

## Send-to / paste params (Neo)

Forge Classic renombró el módulo:

```python
# A1111 antiguo
# from modules import generation_parameters_copypaste as send

# Forge Neo / Classic
from modules import infotext_utils as send

send.register_paste_params_button(
    send.ParamBinding(
        paste_button=btn,
        tabname="txt2img",  # txt2img | img2img | inpaint | extras
        source_image_component=img,
        source_text_component=info_box,
    )
)
```

## Callbacks frecuentes (`modules.script_callbacks`)

| Nombre | Firma típica | Uso |
|--------|--------------|-----|
| `on_ui_tabs` | `() -> list[(Blocks, title, id)]` | Pestaña |
| `on_ui_settings` | `()` | `shared.opts.add_option(...)` |
| `on_app_started` | `(Blocks, FastAPI)` | Rutas API |
| `on_before_ui` | `()` | Antes de construir UI |
| `on_after_component` | `(component, **kwargs)` | Enganchar a un `elem_id` |
| `on_image_saved` | `(ImageSaveParams)` | Post-guardado |

Registrar en import del script: `script_callbacks.on_ui_tabs(fn)`.

## Localizaciones

```text
extensions/webui-localization-xx_XX/
  localizations/
    xx_XX.json
```

Opcional: JS/CSS/Python en la misma extensión.

## docker-neo — flags relevantes

```text
COMMANDLINE_ARGS=... --skip-prepare-environment --skip-install ...
```

| Flag | Efecto en extensiones |
|------|------------------------|
| `--skip-install` | No ejecuta `install.py` |
| `--enable-insecure-extension-access` | Permite instalar/gestionar desde UI con `--listen` |
| `--data-dir /data` | Extensiones de usuario en `/data/extensions` |

## Qué no portar ciegamente de la wiki A1111

- Índice oficial de extensiones / pedir alta en wiki de localizaciones: proceso A1111, no Neo.
- APIs internas de SD1.5-only pueden no existir o diferir en backends Krea/FLUX de Neo.
- Extensiones que pinchan versiones viejas de `numpy` / `torch` / ORT suelen romper la imagen Neo (Python moderno + CUDA reciente).
