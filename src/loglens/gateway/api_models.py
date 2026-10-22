"""Request and response bodies for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

MAX_LINE_CHARS = 4000


class LogItem(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_LINE_CHARS)
    source: str = Field(default="unknown", max_length=64)


class TriageRequest(BaseModel):
    """Either a single `message` or a batch in `logs`."""

    message: str | None = Field(default=None, min_length=1, max_length=MAX_LINE_CHARS)
    source: str = Field(default="unknown", max_length=64)
    logs: list[LogItem] | None = None

    @model_validator(mode="after")
    def _one_of(self):
        if (self.message is None) == (self.logs is None):
            raise ValueError("provide exactly one of 'message' or 'logs'")
        if self.logs is not None and not self.logs:
            raise ValueError("'logs' must not be empty")
        return self

    def items(self) -> list[LogItem]:
        return (
            self.logs
            if self.logs is not None
            else [LogItem(message=self.message, source=self.source)]
        )


class TriageResult(BaseModel):
    index: int
    severity: str | None = None
    category: str | None = None
    root_cause: str | None = None
    suggested_action: str | None = None
    provider: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    confidence: float | None = None
    cache: str | None = None
    fallback_reason: str | None = None
    degraded: bool = False
    latency_ms: float | None = None
    error: str | None = None


class TriageResponse(BaseModel):
    request_id: str
    results: list[TriageResult]


# --- OpenAI-compatible subset -------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=32_000)


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0, le=2)
