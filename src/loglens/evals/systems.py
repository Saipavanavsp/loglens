"""The systems compared in the eval and how to run each one on a log line."""

import asyncio
import time
from dataclasses import dataclass

from loglens.config import Settings
from loglens.gateway.providers.base import Provider
from loglens.jsonparse import parse_label
from loglens.prompts import load_prompt, triage_messages
from loglens.schema import triage_json_schema

BASE_OLLAMA_TAG = "qwen2.5:1.5b-instruct"


@dataclass
class System:
    name: str
    provider: Provider
    prompt_version: str = "v1"
    concurrency: int = 1  # 1 for CPU-bound local models so latency isn't inflated by contention

    @property
    def is_paid(self) -> bool:
        return self.provider.external


def build_systems(names: list[str], settings: Settings, finetuned_tag: str) -> list[System]:
    from loglens.gateway.providers.ollama import OllamaProvider

    systems = []
    for name in names:
        if name == "base":
            systems.append(
                System(
                    "base",
                    OllamaProvider(
                        settings.ollama_base_url,
                        BASE_OLLAMA_TAG,
                        "base",
                        num_thread=settings.ollama_num_thread,
                    ),
                )
            )
        elif name == "finetuned":
            systems.append(
                System(
                    "finetuned",
                    OllamaProvider(
                        settings.ollama_base_url,
                        finetuned_tag,
                        "finetuned",
                        num_thread=settings.ollama_num_thread,
                    ),
                )
            )
        elif name == "claude":
            from loglens.gateway.providers.claude import ClaudeProvider

            if settings.anthropic_api_key is None:
                raise SystemExit("ANTHROPIC_API_KEY is not set; cannot evaluate 'claude'")
            key = settings.anthropic_api_key.get_secret_value()
            systems.append(
                System("claude", ClaudeProvider(key, settings.fallback_model), concurrency=4)
            )
        else:
            raise SystemExit(f"unknown system {name!r}; choose from base, finetuned, claude")
    return systems


async def predict_one(system: System, row: dict, timeout_s: float = 120.0) -> dict:
    prompt = load_prompt("triage", system.prompt_version)
    messages = triage_messages(prompt, row["source"], row["message"])
    out = {
        "id": row["id"],
        "source": row["source"],
        "message": row["message"],
        "template": row.get("template"),
        "seen_template": row.get("seen_template"),
        "gold": row["label"],
        "system": system.name,
        "model": system.provider.model,
        "prompt_version": system.prompt_version,
        "pred": None,
        "error": None,
        "parse_error": None,
        "raw": None,
        "latency_ms": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    started = time.perf_counter()
    try:
        comp = await asyncio.wait_for(
            system.provider.complete(
                messages=messages,
                max_tokens=1024 if system.is_paid else 256,
                json_schema=triage_json_schema(),
            ),
            timeout=timeout_s,
        )
    except Exception as exc:  # recorded per row; one failure must not stop the run
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        out["latency_ms"] = (time.perf_counter() - started) * 1000
        return out
    out["latency_ms"] = (time.perf_counter() - started) * 1000
    out["raw"] = comp.text[:2000]
    out["input_tokens"], out["output_tokens"], out["cost_usd"] = (
        comp.input_tokens,
        comp.output_tokens,
        comp.cost_usd,
    )
    parsed = parse_label(comp.text)
    if parsed.ok:
        out["pred"] = parsed.label.model_dump(mode="json")
    else:
        out["parse_error"] = parsed.error
    return out


async def run_system(system: System, rows: list[dict], progress: bool = True) -> list[dict]:
    sem = asyncio.Semaphore(system.concurrency)
    done = 0

    async def one(row):
        nonlocal done
        async with sem:
            res = await predict_one(system, row)
        done += 1
        if progress and (done % 20 == 0 or done == len(rows)):
            print(f"  [{system.name}] {done}/{len(rows)}")
        return res

    return await asyncio.gather(*(one(r) for r in rows))
