"""Any server that speaks the OpenAI chat completions API.

This is how a GPU deployment would plug in: vLLM, TGI and llama.cpp's server
all expose /v1/chat/completions. Not used in the default local setup (Ollama
is used there), but it shows that adding a provider is one small class.
"""

import time

import httpx

from loglens.gateway.providers.base import Completion, ProviderError, TokenLogprob


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        name: str = "openai_compat",
        api_key: str | None = None,
        external: bool = False,
        timeout_s: float = 30.0,
    ):
        self.name = name
        self.model = model
        self.external = external
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout_s)

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        want_logprobs: bool = False,
        temperature: float | None = 0.0,
        json_schema: dict | None = None,
    ) -> Completion:
        body: dict = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        if want_logprobs:
            body["logprobs"] = True
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "triage", "schema": json_schema},
            }
        started = time.perf_counter()
        try:
            resp = await self._client.post("/v1/chat/completions", json=body)
        except httpx.TransportError as exc:
            raise ConnectionError(str(exc)) from exc
        if resp.status_code >= 500 or resp.status_code == 429:
            raise ConnectionError(f"{self.name} {resp.status_code}")
        if resp.status_code >= 400:
            raise ProviderError(f"{self.name} {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        choice = data["choices"][0]
        lp = None
        if choice.get("logprobs") and choice["logprobs"].get("content"):
            lp = [TokenLogprob(t["token"], t["logprob"]) for t in choice["logprobs"]["content"]]
        usage = data.get("usage", {})
        return Completion(
            text=choice["message"]["content"] or "",
            model=self.model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=(time.perf_counter() - started) * 1000,
            logprobs=lp,
        )

    async def health(self) -> bool:
        try:
            return (await self._client.get("/v1/models")).status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
