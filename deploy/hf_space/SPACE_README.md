---
title: LogLens
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Log triage with a fine-tuned 1.5B model on CPU
---

# LogLens demo

Paste system or network log lines; a QLoRA fine-tuned Qwen2.5-1.5B model, served with Ollama on
the free CPU tier, returns severity, category, likely root cause and a suggested action.

This Space runs a trimmed version of the gateway: Redis cache and rate limiting, SQLite request log,
no paid fallback. The full stack (PostgreSQL, MLflow, Jaeger, dashboard, Claude fallback) runs with
Docker Compose; see the GitHub repository.

API: `POST /v1/triage` with `Authorization: Bearer public-demo`. Interactive docs at `/docs`.
