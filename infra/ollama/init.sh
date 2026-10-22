#!/bin/sh
# One-shot init for the Ollama container: pull the base model and, if an exported
# GGUF is present, register the fine-tuned model under LOGLENS_LOCAL_MODEL.
set -eu

ollama pull qwen2.5:1.5b-instruct

if [ -f /models/loglens-q4_k_m.gguf ]; then
  cd /models
  ollama create "${LOGLENS_LOCAL_MODEL:-loglens-qwen2.5-1.5b}" -f Modelfile
  echo "registered fine-tuned model ${LOGLENS_LOCAL_MODEL:-loglens-qwen2.5-1.5b}"
else
  echo "no fine-tuned GGUF in infra/ollama/; serving the base model only"
fi
