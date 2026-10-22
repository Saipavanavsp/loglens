# Hugging Face Space (Docker SDK, free CPU tier): Redis + Ollama + gateway in one container.
FROM ollama/ollama:0.40.8 AS ollama

FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends redis-server ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Ollama binary and its CPU runner libraries.
COPY --from=ollama /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama /usr/lib/ollama /usr/lib/ollama

# Spaces run as uid 1000 with a writable home.
RUN useradd --create-home --uid 1000 user
ENV HOME=/home/user \
    OLLAMA_MODELS=/home/user/.ollama/models \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_KEEP_ALIVE=24h \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /home/user/app
COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -e ".[gateway,embeddings]" "aiosqlite>=0.20" "huggingface_hub>=1.0"
COPY --chown=user prompts ./prompts
COPY --chown=user configs ./configs
COPY --chown=user Modelfile start.sh ./

USER user
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 7860
CMD ["sh", "start.sh"]
