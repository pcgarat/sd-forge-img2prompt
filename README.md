# sd-forge-img2prompt

Extensión para **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** que te ayuda a **reproducir una imagen con un prompt** pensado para el modelo del **UI Preset** activo.

## ¿Qué hace?

1. Abres la pestaña **Image → Prompt**.
2. Subes una imagen o la **pegas desde el portapapeles**.
3. (Opcional) Escribes unas **notas** sobre lo que quieres conservar (sujeto, luz, estilo…).
4. Pulsas **Generate**: lee el checkpoint/TE del **preset Forge** (`forge_checkpoint_<preset>` + módulos), no solo `sd_model_checkpoint` (a veces queda desfasado), y genera un **prompt en prosa** (no tags booru).
5. **Send to txt2img** / **img2img** vuelca el prompt.
6. Muestra **hints de sampler** (Krea Turbo/RAW o Klein distilled/base) sin cambiar settings.

Soportado: **Krea 2** y **FLUX.2 Klein 9B**. Otros stacks → aviso, sin fingir optimización.

### Estado actual (v1)

- Stub sin visión: el prompt sale de tus **notas** + perfil del stack (`StubProvider`).
- `PromptProvider` listo para un VL/API después.

## Requisitos

- Forge Neo
- Preset / checkpoint **Krea 2** (+ Qwen3-VL) o **Klein 9B** (+ Qwen3 8B)

Sin `install.py` (compatible con `--skip-install`).

## Instalación

### Desde la WebUI (recomendado)

1. **Extensions** → **Install from URL**
2. Pega:

```text
https://github.com/pcgarat/sd-forge-img2prompt
```

3. **Install** → **Apply and restart UI**

### Por línea de comandos

```bash
git clone https://github.com/pcgarat/sd-forge-img2prompt.git \
  "$EXTENSIONS_PATH/sd-forge-img2prompt"
```

Reinicia Forge Neo (o `Apply and restart UI`).

## Uso rápido

| Paso | Acción |
|------|--------|
| 1 | Carga Krea 2 + TE Qwen3-VL |
| 2 | Pestaña **Image → Prompt** |
| 3 | Imagen + notas |
| 4 | **Generate** → revisa prompt e hints |
| 5 | **Send to txt2img** (o img2img) → genera |

## Desarrollo

```bash
git clone https://github.com/pcgarat/sd-forge-img2prompt.git
cd sd-forge-img2prompt
python -m pytest tests -q
```

Con **docker-neo** en la misma máquina: monta el repo vía `IMG2PROMPT_EXT_PATH` (compose) y `make restart`. No uses symlink dentro de `EXTENSIONS_PATH` — el contenedor no ve rutas fuera de `/data`.

| Ruta | Rol |
|------|-----|
| `docs/spec-forge-neo-img2prompt_25-09-2026.md` | Spec v1 |
| `tasks/plan.md` / `tasks/todo.md` | Plan y checklist |
| `scripts/img2prompt.py` | Pestaña Gradio (`on_ui_tabs`) + send-to |
| `forge_img2prompt/stack.py` | Detección Krea 2 / Klein / turbo / RAW |
| `forge_img2prompt/provider.py` | `PromptProvider` + stub v1 |

Para enchufar un backend real: implementa `PromptProvider` y sustituye `_PROVIDER` en `scripts/img2prompt.py`.

## Licencia

MIT
