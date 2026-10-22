"""Provider interface: anything that can answer a chat request."""

from dataclasses import dataclass, field
from typing import Protocol


class ProviderError(RuntimeError):
    """A provider failed in a way that should trigger fallback (not a bug)."""


@dataclass
class TokenLogprob:
    token: str
    logprob: float


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    logprobs: list[TokenLogprob] | None = None
    raw: dict = field(default_factory=dict)


class Provider(Protocol):
    name: str  # stable id used in config and logs, e.g. "local", "claude"
    model: str  # model version string recorded with every request
    external: bool  # True if requests leave our network (redaction applies)

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        want_logprobs: bool = False,
        temperature: float | None = None,
        json_schema: dict | None = None,
    ) -> Completion: ...

    async def health(self) -> bool: ...

    async def aclose(self) -> None: ...
