# LLMOps

## What is versioned, and where

| Thing | Where it lives | How a change ships |
|---|---|---|
| Prompts | `prompts/<name>/<version>.yaml` in git | add a new version file, point `configs/gateway.yaml` `prompt.canary` at it |
| Models | MLflow registry `loglens-triage` (versions + `champion` / `challenger` aliases) | `registry log-training` then `registry promote` |
| Active versions | `configs/gateway.yaml` (`models.stable`, `models.canary`, `prompt.*`) | written by `registry promote`, reviewed as a git diff |
| Eval results | `reports/eval/<run>/`, PostgreSQL `eval_metric`, MLflow `loglens-eval` | `python -m loglens.evals.run` |

Every request log row and every eval prediction records the model version and prompt version that
produced it, so any number can be traced back to exactly what generated it.

The gateway reads versions from its config file, not from MLflow at request time. A registry outage
can't break serving, and a deploy is a reviewable diff.

## Release flow for a new model or prompt

1. Train on Colab, then `registry log-training ...`. This creates version N.
2. Offline: `python -m loglens.evals.run --systems finetuned --finetuned-tag <tag>` on the golden set.
   Compare against the current champion's report.
3. `registry promote --version N --alias challenger`. This sets the canary slot.
   Set `canary.percent` (e.g. 10) and restart the gateway.
4. Online: after enough traffic, `python -m loglens.llmops.canary --hours 24`. It compares fallback
   rate, invalid-JSON rate, confidence and p95 latency between the slots and prints PROMOTE or HOLD
   with reasons.
5. `registry promote --version N --alias champion`, then set `canary.percent: 0`.
6. Refresh the CI fixture from the new eval run (`python -m loglens.evals.make_ci_fixture`) and commit.

Canary assignment hashes the request id. Production has no labels, so the online signals are
proxies. Accuracy is only measured offline on the golden set, which is why step 2 is required.

## Tracing

OpenTelemetry. Set `OTEL_EXPORTER_OTLP_ENDPOINT` (e.g. `http://jaeger:4318`) and the gateway exports:

- one server span per HTTP request (FastAPI instrumentation);
- `triage` with slot, source and prompt version, and child spans `cache.lookup`, `provider.local`
  (model, confidence, fallback_reason) and `provider.fallback` (model, number of redactions).

Jaeger's UI (port 16686) shows where time goes, for example how much of a slow request was the
local model vs Claude. Log text is never put on spans.

## CI

`.github/workflows/ci.yml`:

- **lint**: ruff and black.
- **test**: the pytest suite. All providers are fakes, Redis is fakeredis, the DB is SQLite.
  `ANTHROPIC_API_KEY` is set to empty so nothing can spend money.
- **eval-regression**: replays recorded fine-tuned model outputs on a fixed 40-line golden subset
  (`evals/fixtures/ci_predictions.jsonl`). Metrics are recomputed through the real parser and
  metric code, and the build fails if valid-JSON rate or accuracy is below `evals/thresholds.json`.
  This catches parser, schema and metric regressions, and stops anyone committing a worse model's
  outputs as the new baseline.
- **eval-live** (manual): runs the same subset against the real GGUF in an Ollama service
  container on the CPU runner.

Why not call the model or Claude on every push: Claude costs money and isn't deterministic, and a
1.5B model on a shared CPU runner is slow. Replay gives a fast, deterministic, free gate; the live
job is there when the model itself changes.
