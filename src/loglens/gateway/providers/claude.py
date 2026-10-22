"""Claude via the official Anthropic SDK."""

import time

import anthropic

from loglens.gateway.providers.base import Completion, ProviderError
from loglens.pricing import cost_usd


class ClaudeProvider:
    external = True

    def __init__(
        self,
        api_key: str,
        model: str,
        name: str = "claude",
        timeout_s: float = 30.0,
        effort: str = "low",
    ):
        self.name = name
        self.model = model
        self.effort = effort
        # Retries are handled by the gateway (with the circuit breaker), not the SDK.
        self._client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=0, timeout=timeout_s)

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        want_logprobs: bool = False,
        temperature: float | None = None,  # not supported by current Claude models; ignored
        json_schema: dict | None = None,
    ) -> Completion:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        output_config: dict = {"effort": self.effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        started = time.perf_counter()
        try:
            resp = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system or anthropic.NOT_GIVEN,
                messages=chat,
                output_config=output_config,
            )
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            raise ConnectionError(str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise ConnectionError("claude rate limited") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ConnectionError(f"claude {exc.status_code}") from exc
            raise ProviderError(f"claude {exc.status_code}: {exc.message}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        if resp.stop_reason == "refusal":
            raise ProviderError("claude refused the request")
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return Completion(
            text=text,
            model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd(self.model, resp.usage.input_tokens, resp.usage.output_tokens),
        )

    async def health(self) -> bool:
        # No free health endpoint; the circuit breaker tracks real call outcomes.
        return True

    async def aclose(self) -> None:
        await self._client.close()
