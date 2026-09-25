---
name: forge-neo-scripts
description: >-
  Desarrolla custom scripts para Forge Neo / A1111 (clase modules.scripts.Script,
  dropdown vs AlwaysVisible, hooks process/run/postprocess, ScriptPostprocessing
  en Extras). Usar al crear o modificar scripts Gradio de txt2img/img2img,
  acordeones AlwaysVisible, o scripts de postprocesado.
---

# Custom scripts Forge Neo

Compatible con [Developing custom scripts (A1111)](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Developing-custom-scripts).
API real leída de Forge Classic `neo` (`modules/scripts.py`).

Para empaquetar como extensión (install.py, metadata, tabs): skill **forge-neo-extensions**.

## Dónde vive el fichero

| Ubicación | `scripts.basedir()` en import | Uso |
|-----------|-------------------------------|-----|
| `<webui>/scripts/*.py` | root WebUI | Script suelto / builtin |
| `<ext>/scripts/*.py` | raíz de la extensión | Lo habitual en Neo |

El loader importa cada `.py` y registra subclases de `Script` o `ScriptPostprocessing`.

## Decisión: dropdown vs AlwaysVisible

| `show(is_img2img)` | UI | Lógica principal |
|--------------------|----|--------------------|
| `True` | Dropdown "Scripts" (ambos tabs) | **`run(p, *args)`** — tú llamas `process_images` |
| `False` | Oculto | — |
| `is_img2img` / `not is_img2img` | Solo un tab | `run` |
| `scripts.AlwaysVisible` | Siempre visible (accordion) | **`process` / hooks**, no `run` |

En Neo, UI AlwaysVisible típica: `InputAccordion` + `sorting_priority` (mayor = más abajo).

```python
from modules import scripts
from modules.ui_components import InputAccordion
from modules.processing import StableDiffusionProcessing, process_images, Processed
import gradio as gr

class MyScript(scripts.Script):
    sorting_priority = 500

    def title(self):
        return "My Script"

    def show(self, is_img2img):
        return scripts.AlwaysVisible  # o True / is_img2img

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enable:
            angle = gr.Slider(0, 360, value=0, label="Angle",
                              elem_id=self.elem_id("angle"))
        return [enable, angle]

    def process(self, p: StableDiffusionProcessing, enable, angle):
        if not enable:
            return
        # mutar p, armar hooks, caches…
```

## Contrato `ui()` → args

1. `ui()` crea componentes Gradio y **devuelve una lista** (orden = orden de args).
2. Esos valores se pasan a `run` / `process` / `before_process` / `postprocess` / etc. **en el mismo orden**.
3. Primer arg de procesamiento siempre es `p` (o `processed` en postprocess); luego los de la UI.

```python
def ui(self, is_img2img):
    a = gr.Slider(...)
    b = gr.Checkbox(...)
    return [a, b]

def process(self, p, a, b):  # mismos nombres/orden
    ...
```

## Dropdown: `run` debe devolver `Processed`

```python
def show(self, is_img2img):
    return True

def run(self, p, angle, hflip, vflip):
    proc = process_images(p)
    # transformar proc.images…
    return proc
```

Imports habituales: `modules.processing` (`process_images`, `Processed`, `StableDiffusionProcessing*`), `modules.images`, `modules.shared` (`opts`, `state`).

## Ciclo AlwaysVisible (orden útil)

```
setup(p, *args)
before_process(p, *args)
process(p, *args)
before_process_batch(p, *args, batch_number=, prompts=, seeds=, subseeds=)
after_extra_networks_activate(...)
process_batch(...)
process_before_every_sampling(...)   # 2× con hires fix
# … denoise …
post_sample(p, PostSampleArgs)       # latentes, pre-VAE
postprocess_batch / postprocess_batch_list
postprocess_image / postprocess_maskoverlay / postprocess_image_after_composite
before_hr(p, *args)                  # si hires
postprocess(p, processed, *args)
```

Elegir el hook más temprano que baste:

| Necesitas… | Hook |
|------------|------|
| Mutar `p` (pasos, size, flags) | `before_process` / `process` |
| Reescribir prompts/seeds del batch | `before_process_batch` |
| Tras LoRAs/extra networks | `after_extra_networks_activate` |
| Por cada sample (tb. hires) | `process_before_every_sampling` |
| Latentes antes del VAE | `post_sample` |
| Cada PIL generada | `postprocess_image` |
| Cleanup / UI final | `postprocess` |

Detalle de firmas: [hooks.md](hooks.md). Plantillas: [examples.md](examples.md).

## Atributos útiles de `Script`

| Atributo | Rol |
|----------|-----|
| `sorting_priority` | Orden UI AlwaysVisible (mayor → más abajo) |
| `section` | Agrupar controles en sección UI |
| `create_group` | `False` = sin `gr.Group` automático |
| `infotext_fields` | `[(component, "Label")]` para PNG info / paste |
| `paste_field_names` | Campos que viajan en "Send to …" |
| `setup_for_ui_only` | No correr setup vía API |
| `elem_id(item_id)` | ID HTML estable `script_{tab}{title}_{item}` |

## Postprocessing (pestaña Extras)

Subclase de `modules.scripts_postprocessing.ScriptPostprocessing`:

- `ui()` → **dict** nombre → componente (no lista).
- `process(pp: PostprocessedImage, **args)` / `process_firstpass`.
- `order` controla orden en Extras.

## Infotext

```python
def ui(self, is_img2img):
    strength = gr.Slider(...)
    self.infotext_fields = [(strength, "My Script Strength")]
    self.paste_field_names = ["My Script Strength"]
    return [strength]
```

## Anti-patrones

- Implementar `run` en AlwaysVisible (no se llama) o solo `process` en dropdown (no se llama).
- Cambiar la **longitud** de `prompts`/`seeds` en `before_process_batch` sin actualizar el resto.
- Usar `scripts.basedir()` fuera del import de una extensión.
- Nombres de fichero/módulo genéricos (`utils.py`) en `scripts/` → colisiones globales.
- Asumir SD1.5: en Neo comprueba el backend (Krea/FLUX/Wan) antes de hijackear UNet.

## Relación con extensiones

Script suelto = un `.py` en `scripts/`.  
Producto repartible = extensión con `scripts/` + paquete hermano → skill **forge-neo-extensions**.

Ejemplos vivos en `docker-neo`: Moodboard, Identity Edit, Depth (`process`); Lama Cleaner (`ScriptPostprocessing` / masked content).
