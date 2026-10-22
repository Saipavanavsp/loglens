# LogLens

LogLens triages system and network logs. Given a raw log line it returns a severity, a category,
a short root-cause explanation and a suggested action as validated JSON.

It is built around three pieces:

- **A fine-tuned small model.** Qwen2.5-1.5B-Instruct, QLoRA fine-tuned on Loghub logs labeled by
  Claude (teacher-student distillation) and partly hand-reviewed.
- **A gateway.** A FastAPI service that sends requests to the local model first and falls back to
  Claude when the local answer is invalid, low-confidence or slow. It handles auth, rate limits,
  budgets, PII redaction, an exact + semantic cache and per-request logging.
- **Evals and LLMOps.** A golden test set, comparison of base vs fine-tuned vs Claude, versioned
  prompts and models, canary routing, tracing, a dashboard and a CI regression gate.

> Status: under construction. See [docs/PLAN.md](docs/PLAN.md).

## Results

All numbers come from real runs. Rows marked as pending have not been run yet.

| Model | Severity macro F1 | Category macro F1 | Valid JSON | p95 latency | Cost / 1k req |
|---|---|---|---|---|---|
| Qwen2.5-1.5B base + prompt | [TO BE FILLED AFTER RUN] | | | | |
| Qwen2.5-1.5B fine-tuned | [TO BE FILLED AFTER RUN] | | | | |
| Claude | [TO BE FILLED AFTER RUN] | | | | |

## Repository layout

```
src/loglens/      Python package: data pipeline, gateway, evals, shared schema/config
training/         QLoRA notebook + script (runs on Colab/Kaggle GPU)
prompts/          Versioned prompt files
dashboard/        Streamlit dashboard and label review tool
infra/            Docker Compose, Dockerfiles, load tests
docs/             Plan, design, fine-tuning notes
tests/            pytest suite
learning/         Small from-scratch transformer (separate learning exercise)
data/             Gitignored; created by the data scripts
```

## Quick start (development)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
./tasks.ps1 install
copy .env.example .env   # then edit .env
./tasks.ps1 check
```

On WSL/Linux use `make install` and `make check`.

## License

MIT
