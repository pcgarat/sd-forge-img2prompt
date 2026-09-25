# sd-forge-img2prompt

Extensión para **Forge Neo**: imagen → prompt en prosa afinado al stack **Krea 2** (checkpoint + text encoder + turbo/RAW).

v1 sin backend de visión: stub que parte de tus notas + perfil Krea 2. Interfaz `PromptProvider` lista para un VL/API después.

## Requisitos

- Forge Neo (Gradio 4.x)
- Checkpoint Krea 2 + TE `qwen3vl_4b` (visión) seleccionados en la UI

## Instalación

**Extensions → Install from URL**

```text
https://github.com/pcgarat/sd-forge-img2prompt
```

Luego **Apply and restart UI**.

O por CLI:

```bash
git clone https://github.com/pcgarat/sd-forge-img2prompt.git "$EXTENSIONS_PATH/sd-forge-img2prompt"
```

Sin `install.py` ni deps extra (compatible con docker-neo `--skip-install`).

## Uso (v1)

1. Abre la pestaña **Image → Prompt**.
2. Sube o pega una imagen (en v1 la imagen se guarda para el provider futuro; el stub usa las **notas**).
3. Escribe una descripción breve en notas (recomendado).
4. **Generate** → revisa el prompt y los hints Turbo/RAW.
5. **Send to txt2img** / **img2img**.

Si el stack no es Krea 2, la UI avisa y no finge optimización.

## Desarrollo

```bash
python -m pytest tests -q
```

Lógica en `forge_img2prompt/`; UI en `scripts/img2prompt.py` (`on_ui_tabs`).

Para enchufar un backend real: implementa `PromptProvider` y sustituye el stub en el script (un solo punto de wiring).

## Licencia

MIT
