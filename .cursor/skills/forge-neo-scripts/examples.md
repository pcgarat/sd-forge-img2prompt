# Ejemplos — custom scripts Forge Neo

Última modificación: 2026-09-25

## 1. Dropdown: flip/rotate (patrón wiki A1111)

```python
import modules.scripts as scripts
import gradio as gr
from modules import images
from modules.processing import process_images
from modules.shared import opts
from PIL import Image


class Script(scripts.Script):
    def title(self):
        return "Flip/Rotate Output"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        angle = gr.Slider(0.0, 360.0, step=1, value=0, label="Angle")
        hflip = gr.Checkbox(False, label="Horizontal flip")
        vflip = gr.Checkbox(False, label="Vertical flip")
        overwrite = gr.Checkbox(False, label="Overwrite existing files")
        return [angle, hflip, vflip, overwrite]

    def run(self, p, angle, hflip, vflip, overwrite):
        def rotate_and_flip(im):
            out = im
            if angle:
                out = out.rotate(angle, expand=True)
            if hflip:
                out = out.transpose(Image.FLIP_LEFT_RIGHT)
            if vflip:
                out = out.transpose(Image.FLIP_TOP_BOTTOM)
            return out

        basename = ""
        if not overwrite:
            if angle:
                basename += f"rotated_{angle}"
            if hflip:
                basename += "_hflip"
            if vflip:
                basename += "_vflip"
        else:
            p.do_not_save_samples = True

        proc = process_images(p)
        for i, im in enumerate(proc.images):
            proc.images[i] = rotate_and_flip(im)
            images.save_image(
                proc.images[i], p.outpath_samples, basename,
                proc.seed + i, proc.prompt, opts.samples_format,
                info=proc.info, p=p,
            )
        return proc
```

## 2. AlwaysVisible + InputAccordion (patrón Neo)

```python
from modules import scripts
from modules.processing import StableDiffusionProcessing
from modules.ui_components import InputAccordion
import gradio as gr


class SeedSuffix(scripts.Script):
    sorting_priority = 400

    def title(self):
        return "Seed Suffix"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enable:
            suffix = gr.Number(value=0, label="Add to seed", precision=0,
                               elem_id=self.elem_id("suffix"))
        return [enable, suffix]

    def before_process(self, p: StableDiffusionProcessing, enable, suffix):
        if enable:
            p.seed = int(p.seed) + int(suffix)
```

## 3. Reescribir prompts del batch

```python
def before_process_batch(self, p, enable, tag, **kwargs):
    if not enable:
        return
    prompts = kwargs.get("prompts")
    if not prompts:
        return
    for i, prompt in enumerate(prompts):
        prompts[i] = f"{prompt}, {tag}"
```

## 4. Postprocess por imagen

```python
from modules.scripts import PostprocessImageArgs

def postprocess_image(self, p, pp: PostprocessImageArgs, enable, *args):
    if not enable:
        return
    # pp.image: PIL.Image; pp.index: int
    pp.image = pp.image.convert("RGB")
```

## 5. Infotext / paste

```python
def ui(self, is_img2img):
    with InputAccordion(False, label=self.title()) as enable:
        strength = gr.Slider(0, 1, value=0.5, label="Strength",
                             elem_id=self.elem_id("strength"))
    self.infotext_fields = [
        (enable, "MyScript Enabled"),
        (strength, "MyScript Strength"),
    ]
    self.paste_field_names = ["MyScript Enabled", "MyScript Strength"]
    return [enable, strength]
```

## 6. ScriptPostprocessing (Extras)

```python
import gradio as gr
from modules import scripts_postprocessing
from PIL import ImageOps


class ScriptPostprocessingGrayscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Grayscale"
    order = 2000

    def ui(self):
        enable = gr.Checkbox(False, label="Convert to grayscale")
        return {"enable": enable}

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable=False):
        if not enable:
            return
        pp.image = ImageOps.grayscale(pp.image).convert("RGB")
        pp.nametags.append("gray")
```

## Referencias vivas (`docker-neo`)

| Script | Tipo |
|--------|------|
| `extensions/sd-forge-krea2-moodboard` | AlwaysVisible + `process` |
| `extensions/sd-forge-krea2-edit` | AlwaysVisible + `process`/`postprocess` |
| `builtin-extensions/sd-forge-krea2-depth-controlnet` | AlwaysVisible + sampling hooks |
| `builtin-extensions/forge-neo-lama-cleaner` | Postprocessing + masked content |
| `builtin-extensions/ADetailer-Neo` | AlwaysVisible + callbacks |
