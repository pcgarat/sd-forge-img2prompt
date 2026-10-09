# Última modificación: 2026-10-09

# Spec: Backend NaN (nan.builders) para Image → Prompt

## Objective

Permitir que la extensión use **NaN** ([nan.builders](https://nan.builders/es), clúster
comunitario de GPUs que sirve modelos abiertos) como backend de visión, sin cargar
ningún VL en el proceso Forge. NaN expone una **API compatible con OpenAI**
(`https://api.nan.builders/v1`) y Bearer key `sk-…`; el transporte es
`POST /v1/chat/completions` con `image_url` (data URL) por imagen. Conexión en
**Settings** de Forge Neo; **modelo en el dropdown de la pestaña**.

## Modelos de visión del clúster (entrada de imagen)

| id | Perfil | Contexto | Cuota / notas |
|----|--------|----------|---------------|
| `deepseek-v4-flash` | DeepSeek 305B MoE (Vision-Exp) | 1M | 3B tok/mes · **default** |
| `glm5.3-flash` | GLM 320B-18B MoE multimodal | 1M | 2B tok/mes · MIT |
| `qwen3.8-flash` | Qwen 125B-6B MoE | 1M | 500M tok/mes |
| `mimo-v2.6-flash` | Xiaomi MiMo omnimodal | 1M | 1B tok/mes |
| `gemma4` | Gemma 26B-A4B MoE | 262K | sin contador |
| `qwen3.6` | Qwen 35B-A3B MoE | 262K | sin contador |
| `glm5.3` | GLM 753B MoE (premium) | 1M | 3B tok/periodo · **tier premium** |

Todos piensan antes de responder; la traza va en `message.reasoning_content`. El cliente
pide `reasoning_effort: "none"` para no pagar latencia de razonamiento (NaN lo acepta
siempre y lo ignora donde no aplica). El catálogo vivo es `GET /v1/models`; la extensión
lo interseca con el registro curado y cae al registro si no responde.

## Architecture (implementada)

- Settings (solo conexión): `img2prompt_nan_*` (`base_url`, `api_key`, `timeout`).
  También lee `NAN_API_KEY` / `NAN_BASE_URL` del entorno como alternativa.
- Dropdown pestaña: una entrada `nan:{id}` por modelo con visión.
  - Descubrimiento: `GET /v1/models` + filtro contra el registro curado.
  - Fallback curado si NaN no responde (`NAN_VISION_MODELS`).
  - Botón ↻ refresca el catálogo sin reiniciar Forge.
- Cliente: `forge_img2prompt/nan_client.py` → `POST /v1/chat/completions` (+ Bearer).
- Provider: `NanVLProvider` reutiliza todo el flujo de `OllamaVLProvider`
  (prompts, rangos, refs, filtros post-VL) cambiando solo config y transporte
  (`/api/chat`+`images` → `/chat/completions`+`image_url`).
- Sin descarga HF ni `is_local_ready`: NaN es backend remoto (`choice.is_remote`).
- Errores: 401/403 → revisa key; 402 → cuota agotada (cambia de modelo); premium.

## Acceptance

Ver `tests/test_nan_client.py` + `tests/test_vl_catalog.py` y el README.
