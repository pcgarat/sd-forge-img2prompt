# sd-forge-img2prompt

Extensión para **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** que te ayuda a **reproducir una imagen con un prompt** pensado para el modelo del **UI Preset** activo.

## ¿Qué hace?

1. Abres la pestaña **Image → Prompt**.
2. Subes una imagen o la **pegas desde el portapapeles**.
3. (Opcional) Escribes unas **notas** sobre lo que quieres conservar (sujeto, luz, estilo…).
4. Pulsas **Generate**: lee el checkpoint/TE del **preset Forge** (`forge_checkpoint_<preset>` + módulos), no solo `sd_model_checkpoint` (a veces queda desfasado), y genera un **prompt en prosa** (no tags booru).
5. (Opcional) Pinta una zona con el **pincel** y pulsa **Añadir detalle**: el VL describe el crop en **Prompt de la zona** (sin mezclarlo al prompt general).
6. **Send to txt2img** / **img2img** vuelca el prompt.
7. Muestra **hints de sampler** (Krea Turbo/RAW o Klein distilled/base) sin cambiar settings.

Soportado (perfil prosa/hints): **Krea 2** y **FLUX.2 Klein 9B**. Otros stacks → aviso.

### Estado actual

- Backend **Ollama** (recomendado) vía `/api/chat` + imágenes.
  El **modelo se elige en el dropdown** de la pestaña (todos los tags con
  `vision` de tu Ollama, local y cloud). Conexión en
  **Settings → Image → Prompt / Ollama**: URL, timeout y **API key**
  (cloud / `https://ollama.com`). Botón ↻ junto al dropdown refresca la lista.
  No carga el VL en el proceso Forge → no pelea VRAM con el checkpoint.
  Ejemplos típicos (según lo que tengas pullado):
  - `qwen3-vl:8b-instruct` — local (preferido)
  - `huihui-qwen3vl-8b-abl:q6` — uncensor local (GGUF Q6_K + `mmproj`)
  - Cloud con visión: `kimi-k3`, `kimi-k2.6`, `mistral-large-3:675b-cloud`,
    `glm-5.3-flash`, `gemma4:31b-cloud` (API key o `ollama signin`)
  En 8 GB VRAM el Huihui 8B irá con offload a RAM; si quieres más holgura usa Q4 o el Huihui 4B (transformers).
- Backend **Transformers** (opcional), descarga a `TextEncoders/<nombre>/`:
  - **Huihui 2B abliterated** (uncensor; ~5 GB VRAM)
  - **Qwen3-VL-2B-Instruct** oficial
  - **Huihui 4B abliterated** (mejor calidad; riesgo OOM en 8 GB)
- Sin imagen: stub con **Notas**.
- **Añadir detalle**: máscara con pincel → crop del bbox → texto solo en «Prompt de la zona».

#### Ollama + Docker (forge-neo)

Si Forge corre en contenedor y Ollama en el host:

1. En el host, Ollama debe escuchar fuera de localhost:
   ```bash
   # /etc/systemd/system/ollama.service.d/override.conf
   [Service]
   Environment="OLLAMA_HOST=0.0.0.0:11434"
   ```
   Luego `sudo systemctl daemon-reload && sudo systemctl restart ollama`.
2. En Settings de la extensión: URL `http://172.17.0.1:11434` (o la gateway de tu red Docker).
3. Misma convención que **chatBot**: env `OLLAMA_HOST` si aplica.

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
| 5 | (Opcional) Pinta zona → **Añadir detalle** |
| 6 | **Send to txt2img** (o img2img) → genera |

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
| `docs/guia_desarrollador_img2prompt_26-09-2026.md` | Guía de desarrollador (flujos + Mermaid) |
| `tasks/plan.md` / `tasks/todo.md` | Plan y checklist |
| `scripts/img2prompt.py` | Pestaña Gradio (`on_ui_tabs`) + send-to |
| `forge_img2prompt/stack.py` | Detección Krea 2 / Klein / turbo / RAW |
| `forge_img2prompt/provider.py` | Stub + `PromptRequest` / `DetailRequest` / `append_detail` |
| `forge_img2prompt/mask_crop.py` | ImageEditor brush → crop del bbox |
| `forge_img2prompt/vl_catalog.py` | Catálogo VL (Ollama + Instruct HF) |
| `forge_img2prompt/vl_provider.py` | Caption + detalle Transformers + composite |
| `forge_img2prompt/ollama_settings.py` | Settings Forge + config Ollama |
| `forge_img2prompt/ollama_client.py` | Cliente `/api/chat` + images |
| `forge_img2prompt/ollama_vl.py` | Provider generate/detail vía Ollama |

## Licencia

MIT
