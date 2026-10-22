"""Central settings, read from environment variables and an optional .env file.

Secrets are typed as SecretStr so they never show up in repr() or logs by accident.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    teacher_model: str = Field(default="claude-opus-5-5", alias="LOGLENS_TEACHER_MODEL")
    fallback_model: str = Field(default="claude-opus-5-5", alias="LOGLENS_FALLBACK_MODEL")
    judge_model: str = Field(default="claude-opus-5-5", alias="LOGLENS_JUDGE_MODEL")

    max_requests: int = Field(default=50, ge=0, alias="LOGLENS_MAX_REQUESTS")
    dry_run: bool = Field(default=True, alias="LOGLENS_DRY_RUN")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    local_model: str = Field(default="loglens-qwen2.5-1.5b", alias="LOGLENS_LOCAL_MODEL")
    # CPU threads for the local model; use the physical core count (see providers/ollama.py).
    ollama_num_thread: int | None = Field(default=8, alias="LOGLENS_OLLAMA_THREADS")

    api_keys: str = Field(default="", alias="LOGLENS_API_KEYS")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    database_url: str = Field(
        default="postgresql+psycopg://loglens:loglens@localhost:5432/loglens",
        alias="DATABASE_URL",
    )

    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    otel_endpoint: str | None = Field(default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")

    @property
    def api_key_set(self) -> set[str]:
        """Gateway API keys as a set; empty entries are dropped."""
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
