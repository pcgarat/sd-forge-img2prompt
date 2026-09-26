# Última modificación: 2026-09-26

# Spec: Backend Ollama para Image → Prompt

## Objective

Permitir que la extensión use **Ollama** (local o cloud) vía API nativa `/api/chat` + `images` (mismo estilo que chatBot), sin cargar el VL en el proceso Forge. Conexión en **Settings** de Forge Neo; **modelo en el dropdown de la pestaña**.

## Architecture (implementada)

- Settings (solo conexión): `img2prompt_ollama_*` (`base_url`, `api_key`, `timeout`)
- Dropdown pestaña: una entrada `ollama:{tag}` por modelo con capability `vision`
  - Descubrimiento: `GET /api/tags` + filtro `vision` + dedupe por digest
  - Fallback curado si Ollama no responde (`OLLAMA_VISION_FALLBACK`)
  - Botón ↻ refresca el catálogo sin reiniciar Forge
- Cliente: `forge_img2prompt/ollama_client.py` → `POST /api/chat` (+ Bearer si hay API key)
- Provider: `OllamaVLProvider` usa el tag del `VlModelChoice` seleccionado
- Docker: default URL `http://172.17.0.1:11434` si hay `/.dockerenv`; host necesita `OLLAMA_HOST=0.0.0.0:11434`
- Cloud directo: URL `https://ollama.com` + API key (también env `OLLAMA_API_KEY`)

## Acceptance

Ver README + `tests/test_ollama_client.py` + `tests/test_vl_catalog.py`.
