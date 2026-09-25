# Última modificación: 2026-09-26

# Plan: sd-forge-img2prompt (Forge Neo / Krea 2 + Klein v1)

Spec: [`docs/spec-forge-neo-img2prompt_25-09-2026.md`](../docs/spec-forge-neo-img2prompt_25-09-2026.md)  
Guía prompts: [`docs/guia_img_prompts_23-09-2026.md`](../docs/guia_img_prompts_23-09-2026.md)  
Todo: [`tasks/todo.md`](todo.md)  
Skill: `.cursor/skills/forge-neo-extensions`

## Enfoque

- Repo **dedicado** (no semilla en `docker-neo/extensions/`).
- UI: `on_ui_tabs` + send-to con `infotext_utils`.
- v1: `StubProvider` (notas → prosa); imagen reservada para backend futuro.
- Sin `install.py` / deps (compatible `--skip-install`).
- Perfiles: **Krea 2** (turbo/RAW) y **Klein 9B** (distilled/base).

## Estado

| Ítem | Estado |
|------|--------|
| Repo GitHub + scaffold | Hecho |
| Tests unitarios stack/stub | Hecho (16 passed) |
| Pestaña + paste_params + Refresh stack | Hecho |
| Smoke Forge Neo (Krea 2) | Hecho 2026-09-26 |
| Dev mount docker-neo | Hecho (`IMG2PROMPT_EXT_PATH`) |
| Seed en docker-neo | No (install por URL / mount) |

## Pendiente

1. (Ask first) `make img2prompt-ext` en docker-neo.
2. Provider VL/API real (post-v1).

## Fuera de ciclo

Auto sampler/CFG, AlwaysVisible.
