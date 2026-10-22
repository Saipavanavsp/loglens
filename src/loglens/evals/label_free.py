"""Label-free benchmark of a local model: valid-JSON rate, latency, confidence.

Needs no gold labels, so it can run before the golden set exists. It does NOT
measure accuracy.

  python -m loglens.evals.label_free --n 100 --tag qwen2.5:1.5b-instruct
"""

import argparse
import asyncio
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

from loglens.config import get_settings
from loglens.data.io import read_jsonl, write_jsonl
from loglens.data.paths import REPORTS, SAMPLE
from loglens.evals.metrics import percentile
from loglens.gateway.confidence import logprob_confidence
from loglens.gateway.providers.ollama import OllamaProvider
from loglens.jsonparse import parse_label
from loglens.prompts import load_prompt, triage_messages


async def bench(provider: OllamaProvider, rows: list[dict], prompt_version: str) -> list[dict]:
    prompt = load_prompt("triage", prompt_version)
    out = []
    # Warm-up request so model load time isn't counted.
    await provider.complete(triage_messages(prompt, "linux", "warm up"), max_tokens=8)
    for i, row in enumerate(rows, 1):
        started = time.perf_counter()
        try:
            c = await provider.complete(
                triage_messages(prompt, row["source"], row["message"]), want_logprobs=True
            )
            parsed = parse_label(c.text)
            out.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "valid": parsed.ok,
                    "parse_error": parsed.error,
                    "confidence": logprob_confidence(c.text, c.logprobs),
                    "output_tokens": c.output_tokens,
                    "severity": parsed.label.severity.value if parsed.ok else None,
                    "category": parsed.label.category.value if parsed.ok else None,
                    "fenced": c.text.lstrip().startswith("```"),
                }
            )
        except Exception as exc:
            out.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                }
            )
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    return out


def summarise(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    lat = [r["latency_ms"] for r in ok]
    conf = [r["confidence"] for r in ok if r.get("confidence") is not None]
    toks = [r["output_tokens"] for r in ok]
    return {
        "n": len(results),
        "request_errors": len(results) - len(ok),
        "valid_json_rate": sum(r["valid"] for r in ok) / max(1, len(ok)),
        "markdown_fenced_rate": sum(r["fenced"] for r in ok) / max(1, len(ok)),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
        "output_tokens_mean": statistics.mean(toks) if toks else None,
        "confidence_p10": percentile(conf, 10),
        "confidence_median": percentile(conf, 50),
        "share_below_0_7": sum(c < 0.7 for c in conf) / max(1, len(conf)),
        "parse_errors": dict(Counter(r["parse_error"] for r in ok if not r["valid"])),
        "predicted_severity": dict(Counter(r["severity"] for r in ok if r["valid"])),
    }


def main() -> None:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--tag", default="qwen2.5:1.5b-instruct")
    p.add_argument("--prompt-version", default="v1")
    p.add_argument("--inp", type=Path, default=SAMPLE)
    args = p.parse_args()
    rows = list(read_jsonl(args.inp))[: args.n]
    provider = OllamaProvider(
        s.ollama_base_url, args.tag, num_thread=s.ollama_num_thread, timeout_s=300
    )
    results = asyncio.run(bench(provider, rows, args.prompt_version))
    summary = {
        "model": args.tag,
        "prompt": f"triage:{args.prompt_version}",
        "machine": f"{platform.system()} host, Ollama in Docker, CPU only, num_thread={s.ollama_num_thread}",
        "date": time.strftime("%Y-%m-%d"),
        **summarise(results),
    }
    out_dir = REPORTS / "label_free"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.tag.replace(":", "_").replace("/", "_")
    write_jsonl(out_dir / f"{slug}.jsonl", results)
    (out_dir / f"{slug}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
