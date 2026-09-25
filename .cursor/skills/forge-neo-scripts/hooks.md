# Hooks de `scripts.Script` (Forge Neo)

Última modificación: 2026-09-25

Fuente: `modules/scripts.py` (Forge Classic rama `neo`).

## Visibilidad — `show(is_img2img)`

| Retorno | Comportamiento |
|---------|----------------|
| `False` | No aparece |
| `True` | Dropdown Scripts (txt2img y img2img) |
| `is_img2img` | Solo img2img |
| `not is_img2img` / `True` condicional | Solo txt2img |
| `scripts.AlwaysVisible` | Siempre; hooks AlwaysVisible |

## Dropdown — `run(p, *ui_args) -> Processed`

- Obligatorio si el script está seleccionado en el dropdown.
- Debe ejecutar (o reemplazar) la generación y **devolver** `Processed`.
- Patrón: `proc = process_images(p)` → transformar `proc.images` → `return proc`.
- Útil: `p.do_not_save_samples = True` si guardas tú con `images.save_image`.

## AlwaysVisible — setup / process

| Método | Cuándo |
|--------|--------|
| `setup(p, *args)` | Al preparar `p`, antes de procesar |
| `before_process(p, *args)` | Muy temprano; mutar `p`, inyectar hooks |
| `process(p, *args)` | Antes de empezar el pipeline |

## Batch

kwargs habituales: `batch_number`, `prompts`, `seeds`, `subseeds`.

| Método | Notas |
|--------|-------|
| `before_process_batch(p, *args, **kwargs)` | Antes de parsear extra networks; puedes editar prompts |
| `after_extra_networks_activate(p, *args, **kwargs)` | Tras LoRAs etc.; también `extra_network_data`. No si `p.disable_extra_networks` |
| `process_batch(p, *args, **kwargs)` | Por cada batch |
| `process_before_every_sampling(p, *args, **kwargs)` | Antes de cada sample; **2 veces** con hires fix |
| `postprocess_batch(p, *args, **kwargs)` | Tras batch; `images` = tensor 4D `[0,1]` |
| `postprocess_batch_list(p, pp: PostprocessBatchListArgs, *args, **kwargs)` | Lista de tensores 3D; si cambias el nº de imágenes, actualiza `p.prompts/seeds/...` |

## Latentes / imagen / final

| Método | Notas |
|--------|--------|
| `on_mask_blend(p, mba: MaskBlendArgs, *args)` | Inpaint: cada step + blend final (`mba.is_final_blend`) |
| `post_sample(p, ps: PostSampleArgs, *args)` | Tras samples, antes VAE; `getattr(ps.samples, 'already_decoded', False)` |
| `postprocess_image(p, pp: PostprocessImageArgs, *args)` | Por imagen (`pp.image`, `pp.index`) |
| `postprocess_maskoverlay(p, ppmo: PostProcessMaskOverlayArgs, *args)` | Overlay máscara |
| `postprocess_image_after_composite(p, pp, *args)` | Tras composite inpaint_full_res (imagen completa) |
| `before_hr(p, *args)` | Justo antes de hires fix |
| `postprocess(p, processed, *args)` | Fin de AlwaysVisible |

## UI injection

| Método | Notas |
|--------|--------|
| `before_component(component, **kwargs)` | Antes de crear componente; mira `elem_id`/`label` en kwargs |
| `after_component(component, **kwargs)` | Después |
| `on_before_component(cb, *, elem_id=)` | Callback puntual; llamar desde `show()` preferible |
| `on_after_component(cb, *, elem_id=)` | Idem; arg `OnComponent` |

## Helpers

```python
self.elem_id("strength")
# → script_txt2img_my_script_strength  (si visible en ambos tabs)
# → script_my_script_strength          (si solo un tab)
```

`sorting_priority`: entero; **mayor = más abajo** en la lista AlwaysVisible.

## Tipos auxiliares

```python
MaskBlendArgs(current_latent, nmask, init_latent, mask, blended_latent, denoiser=None, sigma=None)
PostSampleArgs(samples)
PostprocessImageArgs(image, index)
PostProcessMaskOverlayArgs(index, mask_for_overlay, overlay_image)
PostprocessBatchListArgs(images)
```

## `ScriptPostprocessing` (Extras)

```python
from modules import scripts_postprocessing

class MyPost(scripts_postprocessing.ScriptPostprocessing):
    name = "My Post"
    order = 1000

    def ui(self):
        enable = gr.Checkbox(False, label="Enable")
        return {"enable": enable}  # dict, no list

    def process_firstpass(self, pp, enable=False):
        ...

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable=False):
        if not enable:
            return
        pp.image = ...
        pp.nametags.append("mypost")
```

`PostprocessedImage`: `.image`, `.info`, `.extra_images`, `.nametags`, `.disable_processing`, `.caption`, `.shared`.
