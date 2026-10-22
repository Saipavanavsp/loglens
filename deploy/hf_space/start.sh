#!/bin/sh
# Space entrypoint: start Redis and Ollama, load the model, then the gateway on :7860.
#
# Space settings:
#   Variables: LOGLENS_GGUF_REPO  (e.g. your-name/loglens-qwen2.5-1.5b), LOGLENS_GGUF_FILE (default loglens-q4_k_m.gguf)
#   Secrets:   HF_TOKEN (read access, only needed if the model repo is private)
set -eu

redis-server --daemonize yes --save "" --maxmemory 256mb --maxmemory-policy allkeys-lru
ollama serve > /tmp/ollama.log 2>&1 &
for i in $(seq 1 60); do ollama list > /dev/null 2>&1 && break; sleep 1; done

if [ -n "${LOGLENS_GGUF_REPO:-}" ]; then
  echo "downloading ${LOGLENS_GGUF_REPO}/${LOGLENS_GGUF_FILE:-loglens-q4_k_m.gguf}"
  GGUF=$(python -c "import os; from huggingface_hub import hf_hub_download; print(hf_hub_download(os.environ['LOGLENS_GGUF_REPO'], os.environ.get('LOGLENS_GGUF_FILE', 'loglens-q4_k_m.gguf')))")
  sed "s#^FROM .*#FROM ${GGUF}#" Modelfile > /tmp/Modelfile
  ollama create loglens-demo -f /tmp/Modelfile
  export LOGLENS_DEMO_MODEL_LABEL="Qwen2.5-1.5B QLoRA fine-tuned (${LOGLENS_GGUF_REPO})"
else
  echo "LOGLENS_GGUF_REPO not set: serving the BASE model (not fine-tuned)"
  ollama pull qwen2.5:1.5b-instruct
  ollama cp qwen2.5:1.5b-instruct loglens-demo
  export LOGLENS_DEMO_MODEL_LABEL="Qwen2.5-1.5B-Instruct base (not fine-tuned yet)"
fi

# Public demo key: rate limited, fallback disabled, so it can't spend money.
DEMO_KEY="${LOGLENS_DEMO_KEY:-public-demo}"
export LOGLENS_DEMO_KEY="$DEMO_KEY"
export LOGLENS_API_KEYS="${LOGLENS_API_KEYS:-$DEMO_KEY}"
export LOGLENS_GATEWAY_CONFIG=configs/gateway.space.yaml
export REDIS_URL=redis://127.0.0.1:6379/0
export DATABASE_URL="sqlite+aiosqlite:////home/user/loglens.db"
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export LOGLENS_OLLAMA_THREADS="${LOGLENS_OLLAMA_THREADS:-2}"
export ANTHROPIC_API_KEY=""

exec uvicorn loglens.gateway.app:app --host 0.0.0.0 --port 7860 --timeout-graceful-shutdown 20 --no-access-log
