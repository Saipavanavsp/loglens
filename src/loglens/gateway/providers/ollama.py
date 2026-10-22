"""Local model served by Ollama (llama.cpp under the hood, runs on CPU)."""

import time

import httpx

from loglens.gateway.providers.base import Completion, ProviderError, TokenLogprob


class OllamaProvider:
    external = False

    def __init__(
        self,
        base_url: str,
        model: str,
        name: str = "local",
        timeout_s: float = 60.0,
        constrained_json: bool = False,
        num_ctx: int = 1024,
        num_thread: int | None = None,
    ):
        """
        constrained_json: pass the JSON schema as Ollama's `format`, which uses
        grammar-constrained decoding so the output is always schema-valid. Off by
        default so evals measure what the model itself produces.

        num_thread: CPU threads for llama.cpp. Set it to the physical core count.
        On a 16-vCPU (8-core) WSL2 host, Ollama's default of one thread per vCPU
        generated ~0.5 tok/s; 8 threads gave ~32 tok/s.
        """
        self.name = name
        self.model = model
        self.constrained_json = constrained_json
        self.num_ctx = num_ctx
        self.num_thread = num_thread
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        want_logprobs: bool = False,
        temperature: float | None = 0.0,
        json_schema: dict | None = None,
    ) -> Completion:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.0 if temperature is None else temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        if self.num_thread:
            body["options"]["num_thread"] = self.num_thread
        if want_logprobs:
            body["logprobs"] = True
        if json_schema is not None and self.constrained_json:
            body["format"] = json_schema
        started = time.perf_counter()
        try:
            resp = await self._client.post("/api/chat", json=body)
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"ollama timeout: {exc}") from exc
        except httpx.TransportError as exc:
            raise ConnectionError(f"ollama unreachable: {exc}") from exc
        if resp.status_code >= 500:
            raise ConnectionError(f"ollama {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise ProviderError(f"ollama {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        logprobs = None
        if isinstance(data.get("logprobs"), list):
            logprobs = [
                TokenLogprob(item.get("token", ""), float(item.get("logprob", 0.0)))
                for item in data["logprobs"]
            ]
        return Completion(
            text=data.get("message", {}).get("content", ""),
            model=self.model,
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            latency_ms=(time.perf_counter() - started) * 1000,
            logprobs=logprobs,
            raw={k: data.get(k) for k in ("total_duration", "load_duration", "eval_duration")},
        )

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/api/tags")
        except httpx.HTTPError:
            return False
        if resp.status_code != 200:
            return False
        names = {m.get("name") for m in resp.json().get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    async def aclose(self) -> None:
        await self._client.aclose()
