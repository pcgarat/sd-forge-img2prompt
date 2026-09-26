# Última modificación: 2026-09-26

# Todo: sd-forge-img2prompt

Repo: https://github.com/pcgarat/sd-forge-img2prompt

- [x] Task 1: Esqueleto + stack + stub + tests
- [x] Task 2: Pestaña UI + paste_params
- [x] Smoke Forge Neo stub: Generate → Send txt2img
- [x] Montaje dev docker-neo (`IMG2PROMPT_EXT_PATH`)
- [x] Backend VL: catálogo Instruct (Huihui 2B/4B abliterated + Qwen 2B) + descarga TextEncoders + checklist/progreso
- [x] Log progresivo en terminal Forge (`[img2prompt] …`)
- [x] Máscara → detalle (crop bbox + append al prompt) · rama `feat/mask-add-detail`
- [ ] Smoke VL real (Huihui 2B abliterated + caption con imagen)
- [ ] Smoke máscara: Generate → pintar → Añadir detalle → Send
- [ ] (Ask first) make target `img2prompt-ext` en docker-neo
- [ ] GGUF / llama.cpp (opcional)
- [x] ~~Task 3 seed extensions/~~ cancelada — repo dedicado

## Fuera de ciclo

- Auto sampler/CFG, AlwaysVisible
- Rewrite del prompt completo desde máscara / imagen atenuada
- Multi-máscaras en paralelo
