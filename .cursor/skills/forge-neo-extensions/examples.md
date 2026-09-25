# Ejemplos mínimos — Forge Neo

Última modificación: 2026-09-25

Plantillas alineadas con Moodboard / Identity Edit / IIB en `docker-neo`.

## 1. AlwaysVisible + InputAccordion + process

```text
sd-forge-demo/
  README.md
  scripts/
    demo.py
  forge_demo/
    __init__.py
    logic.py
```

```python
# scripts/demo.py
from modules import scripts
from modules.processing import StableDiffusionProcessing
from modules.ui_components import InputAccordion
import gradio as gr

from forge_demo.logic import apply

class DemoScript(scripts.Script):
    sorting_priority = 600

    def title(self):
        return "Demo Neo"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enable:
            strength = gr.Slider(0, 1, value=0.5, label="Strength",
                                 elem_id=self.elem_id("strength"))
        return [enable, strength]

    def process(self, p: StableDiffusionProcessing, enable: bool, strength: float):
        if not enable:
            return
        apply(p, strength)
```

## 2. Pestaña top-level + API

```python
# scripts/demo_tab.py
from modules import script_callbacks
from fastapi import FastAPI
import gradio as gr

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.Markdown("Demo tab")
    return [(demo, "Demo", "demo_neo_tab")]

def on_app_started(_blocks: gr.Blocks, app: FastAPI):
    @app.get("/demo-neo/health")
    def health():
        return {"ok": True}

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_app_started(on_app_started)
```

## 3. basedir + asset path

```python
from modules import scripts
import os

EXT_DIR = scripts.basedir()
ASSETS = os.path.join(EXT_DIR, "assets")
```

## 4. install.py (solo si se hornea en imagen)

```python
import launch

if not launch.is_installed("example-pkg"):
    launch.run_pip("install example-pkg==1.0.0", "demo-neo deps")
```

En docker-neo: añadir el mismo `pip install` al Dockerfile; no esperar a runtime.

## 5. metadata.ini mínimo

```ini
[Extension]
Name = sd-forge-demo

[scripts]
After = sd-forge-krea2-moodboard
```

## 6. JS: escribir en el prompt activo

```javascript
// javascript/send_prompt.js
function demoSendPrompt(text) {
    const tab = gradioApp().querySelector("#tabs > .tabitem:not([style*='display: none'])");
    const scope = tab || gradioApp();
    const box =
        scope.querySelector("#txt2img_prompt textarea") ||
        scope.querySelector("#img2img_prompt textarea");
    if (!box) return;
    box.value = text;
    updateInput(box);
}
```

Preferible en tools con `on_ui_tabs`. En AlwaysVisible, a menudo basta devolver el string a un `gr.Textbox` y un botón que copie, o usar callbacks Gradio nativos.

## Referencias vivas en el repo

| Extensión | Patrón |
|-----------|--------|
| `extensions/sd-forge-krea2-moodboard` | AlwaysVisible + `process` |
| `extensions/sd-forge-krea2-edit` | AlwaysVisible + `process`/`postprocess` |
| `builtin-extensions/sd-forge-krea2-depth-controlnet` | AlwaysVisible + paquete `forge_krea2_depth/` + `install.py` |
| `extensions/sd-webui-infinite-image-browsing` | `on_ui_tabs` + `on_app_started` + `infotext_utils` |
| `builtin-extensions/sd-civitai-browser-neo` | `on_ui_tabs` + Settings |
| `builtin-extensions/forge-neo-lama-cleaner` | `metadata.ini` + postprocessing script |
