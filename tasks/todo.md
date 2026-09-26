# Última modificación: 2026-09-26

# Todo: sd-forge-img2prompt

Repo: https://github.com/pcgarat/sd-forge-img2prompt

- [x] Task 1: Esqueleto + stack + stub + tests
- [x] Task 2: Pestaña UI + paste_params
- [x] Smoke Forge Neo stub: Generate → Send txt2img
- [x] Montaje dev docker-neo (`IMG2PROMPT_EXT_PATH`)
- [x] Backend VL: catálogo Instruct (Huihui 2B/4B abliterated + Qwen 2B) + descarga TextEncoders + checklist/progreso
- [x] Backend Ollama (`/api/chat` + Settings conexión + modelos vision en dropdown pestaña)
- [x] Selector pestaña: descubrimiento `/api/tags` (vision) + refresh ↻; sin DeepSeek
- [ ] Smoke VL real (Huihui 2B abliterated + caption con imagen)
- [ ] Smoke Ollama desde pestaña (Generate con `ollama:…`)
- [ ] Smoke máscara: Generate → pintar → Añadir detalle → Send
- [ ] (Ask first) make target `img2prompt-ext` en docker-neo
- [ ] Host: `OLLAMA_HOST=0.0.0.0:11434` (sudo) para Docker → Ollama
- [x] ~~GGUF / llama.cpp (opcional)~~ → cubierto vía Ollama Q4
- [x] ~~Task 3 seed extensions/~~ cancelada — repo dedicado

## Fuera de ciclo

- Auto sampler/CFG, AlwaysVisible
- Rewrite del prompt completo desde máscara / imagen atenuada
- Multi-máscaras en paralelo
