# sd-forge-img2prompt

Extensión para **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** que te ayuda a **reproducir una imagen con un prompt** pensado para el modelo que tienes cargado.

## ¿Qué hace?

1. Abres la pestaña **Image → Prompt**.
2. Subes una imagen o la **pegas desde el portapapeles**.
3. (Opcional) Escribes unas **notas** sobre lo que quieres conservar (sujeto, luz, estilo…).
4. Pulsas **Generate**: la extensión mira el **checkpoint y text encoder** seleccionados en Forge y genera un **prompt en prosa natural** (estilo Krea 2 / Qwen3-VL, no lista de tags booru).
5. Con **Send to txt2img** / **Send to img2img** vuelcas ese prompt al cuadro de generación.
6. Además muestra **pistas de sampler** según si el checkpoint parece Turbo (~8 steps, CFG bajo) o RAW (~28 steps, CFG ~4.5), sin cambiar tus settings solos.

Si el stack **no es Krea 2**, avisa y **no** presenta el resultado como “optimizado” para ese modelo.

### Estado actual (v1)

- Enfocado en **Krea 2** (Turbo / RAW).
- El “cerebro” de visión aún no está: la imagen se acepta en la UI, pero el prompt se construye sobre todo a partir de tus **notas** + el perfil Krea 2 (**StubProvider**).
- La arquitectura deja un `PromptProvider` listo para enchufar después un modelo de visión o una API, sin rehacer la pestaña.

## Requisitos

- Forge Neo
- Checkpoint **Krea 2** y text encoder **Qwen3-VL 4B** (`qwen3vl_4b_…`) seleccionados en la UI

No instala dependencias extra (`install.py` ausente): compatible con entornos que usan `--skip-install` (p. ej. docker-neo).

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

| Ruta | Rol |
|------|-----|
| `scripts/img2prompt.py` | Pestaña Gradio (`on_ui_tabs`) + send-to |
| `forge_img2prompt/stack.py` | Detección Krea 2 / turbo / RAW |
| `forge_img2prompt/provider.py` | `PromptProvider` + stub v1 |

## Licencia

MIT
