# Gateway image. Build context: repository root.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

# sentence-transformers for the semantic cache pulls in torch; the CPU wheel keeps it ~200 MB.
ARG WITH_EMBEDDINGS=true

WORKDIR /app
# Dependencies first so code changes don't invalidate this layer.
COPY pyproject.toml README.md ./
COPY src/loglens/__init__.py src/loglens/__init__.py
RUN pip install --upgrade pip \
 && if [ "$WITH_EMBEDDINGS" = "true" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install -e ".[gateway,embeddings]"; \
    else pip install -e ".[gateway]"; fi

COPY src ./src
COPY prompts ./prompts
COPY configs ./configs

# Download the embedding model at build time so startup doesn't need the internet.
RUN if [ "$WITH_EMBEDDINGS" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"; \
    fi

RUN useradd --create-home --uid 10001 app && chown -R app /opt/hf 2>/dev/null || true
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=2)"

# Uvicorn waits up to 20 s for in-flight requests on SIGTERM; the app then flushes logs.
CMD ["uvicorn", "loglens.gateway.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-graceful-shutdown", "20", "--no-access-log"]
