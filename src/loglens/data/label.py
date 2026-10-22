"""Label sampled log lines with Claude as the teacher model.

Safety rails (all on by default):
- dry run unless --no-dry-run is passed; dry run prints the plan and the first prompt
- hard cap on API requests (--max-requests / LOGLENS_MAX_REQUESTS)
- cost estimate and a y/N prompt before any request (skip prompt with --yes)
- disk cache keyed by (model, prompt version, line): reruns never pay twice

Several lines go into one request (--batch-size) to amortise the system prompt.
The response is constrained to a JSON schema, then every label is validated
again with Pydantic. Lines missing from a batch answer are retried one by one.
Transport errors (429, 5xx, timeouts) are retried by the SDK with backoff.

Usage:
  python -m loglens.data.label --limit 50                 # dry run
  python -m loglens.data.label --limit 50 --no-dry-run    # real run, asks first
"""

import argparse
import asyncio
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from loglens.config import get_settings
from loglens.data.io import append_jsonl, read_jsonl, write_jsonl
from loglens.data.paths import LABEL_CACHE, REPORTS, SAMPLE, TEACHER_LABELS
from loglens.pricing import cost_usd, estimate_tokens
from loglens.prompts import Prompt, load_prompt
from loglens.schema import TriageLabel, triage_json_schema
from loglens.spend import SpendPlan, confirm_spend

# Rough per-item output size (JSON label) and per-request overhead (thinking, wrapper).
EST_OUT_PER_ITEM = 130
EST_OUT_PER_REQUEST = 300


def batch_schema() -> dict:
    item = triage_json_schema()
    item = {
        **item,
        "properties": {"id": {"type": "string"}, **item["properties"]},
        "required": ["id", *item["required"]],
    }
    return {
        "type": "object",
        "properties": {"labels": {"type": "array", "items": item}},
        "required": ["labels"],
        "additionalProperties": False,
    }


def cache_key(model: str, prompt: Prompt, row: dict) -> str:
    raw = f"{model}|{prompt.ref}|{row['id']}|{row['message']}"
    return hashlib.sha256(raw.encode()).hexdigest()


def render_batch(prompt: Prompt, rows: list[dict]) -> str:
    if prompt.item_template is None:
        raise ValueError(f"prompt {prompt.ref} has no item_template")
    items = "\n".join(
        prompt.item_template.format(id=r["id"], source=r["source"], message=r["message"])
        for r in rows
    )
    return prompt.render_user(items=items)


@dataclass
class RunStats:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    labeled: int = 0
    invalid: int = 0
    errors: list[str] = field(default_factory=list)


class RequestBudget:
    """Counts requests and refuses once the cap is reached."""

    def __init__(self, cap: int):
        self.cap = cap
        self.used = 0

    def take(self) -> bool:
        if self.used >= self.cap:
            return False
        self.used += 1
        return True


def parse_batch_response(text: str, expected_ids: set[str]) -> tuple[dict[str, TriageLabel], int]:
    """Validate a batch answer. Returns (labels by id, number of invalid items)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}, len(expected_ids)
    labels: dict[str, TriageLabel] = {}
    invalid = 0
    for item in data.get("labels", []) if isinstance(data, dict) else []:
        if not isinstance(item, dict) or item.get("id") not in expected_ids:
            invalid += 1
            continue
        item_id = item.pop("id")
        try:
            labels[item_id] = TriageLabel.model_validate(item)
        except ValidationError:
            invalid += 1
    return labels, invalid


async def label_batch(
    client,
    model: str,
    prompt: Prompt,
    rows: list[dict],
    budget: RequestBudget,
    stats: RunStats,
    max_tokens: int = 8000,
) -> dict[str, TriageLabel]:
    """One API request for a batch. Never raises for API-level problems."""
    if not budget.take():
        stats.errors.append("request cap reached")
        return {}
    stats.requests += 1
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=prompt.system,
            messages=[{"role": "user", "content": render_batch(prompt, rows)}],
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": batch_schema()},
            },
        )
    except Exception as exc:  # SDK already retried transient errors
        stats.errors.append(f"{type(exc).__name__}: {exc}"[:300])
        return {}
    stats.input_tokens += resp.usage.input_tokens
    stats.output_tokens += resp.usage.output_tokens
    if resp.stop_reason in {"refusal", "max_tokens"}:
        stats.errors.append(f"stop_reason={resp.stop_reason} for batch of {len(rows)}")
        return {}
    text = next((b.text for b in resp.content if b.type == "text"), "")
    labels, invalid = parse_batch_response(text, {r["id"] for r in rows})
    stats.invalid += invalid
    return labels


async def run_labeling(
    client,
    model: str,
    prompt: Prompt,
    todo: list[dict],
    batch_size: int,
    concurrency: int,
    budget: RequestBudget,
    on_label,
) -> RunStats:
    stats = RunStats()
    sem = asyncio.Semaphore(concurrency)

    async def handle(rows: list[dict]) -> None:
        async with sem:
            got = await label_batch(client, model, prompt, rows, budget, stats)
        missing = [r for r in rows if r["id"] not in got]
        # Retry lines the batch answer dropped or got wrong, one at a time.
        if len(rows) > 1:
            for r in missing:
                async with sem:
                    got.update(await label_batch(client, model, prompt, [r], budget, stats))
        for r in rows:
            if r["id"] in got:
                on_label(r, got[r["id"]])
                stats.labeled += 1

    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    await asyncio.gather(*(handle(b) for b in batches))
    return stats


def load_cache(path: Path) -> dict[str, dict]:
    return {row["cache_key"]: row for row in read_jsonl(path)}


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    p = argparse.ArgumentParser(description="Label log lines with the Claude teacher model")
    p.add_argument("--inp", type=Path, default=SAMPLE)
    p.add_argument("--out", type=Path, default=TEACHER_LABELS)
    p.add_argument("--limit", type=int, default=50, help="label the first N sampled lines")
    p.add_argument("--model", default=settings.teacher_model)
    p.add_argument("--prompt-version", default="v1")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-requests", type=int, default=settings.max_requests)
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=settings.dry_run)
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = p.parse_args(argv)

    prompt = load_prompt("teacher", args.prompt_version)
    rows = list(read_jsonl(args.inp))[: args.limit]
    if not rows:
        print(f"No rows in {args.inp}. Run: python -m loglens.data.sample --n {args.limit}")
        return 1

    cache = load_cache(LABEL_CACHE)
    todo = [r for r in rows if cache_key(args.model, prompt, r) not in cache]
    print(
        f"{len(rows)} lines selected, {len(rows) - len(todo)} already cached, {len(todo)} to label"
    )

    n_batches = -(-len(todo) // args.batch_size)
    sys_tokens = estimate_tokens(prompt.system)
    est_in = n_batches * sys_tokens + sum(estimate_tokens(r["message"]) + 15 for r in todo)
    est_out = len(todo) * EST_OUT_PER_ITEM + n_batches * EST_OUT_PER_REQUEST
    # Allow ~10% headroom in the estimate for per-line retries.
    plan = SpendPlan(
        model=args.model,
        n_requests=n_batches,
        est_input_tokens=int(est_in * 1.1),
        est_output_tokens=int(est_out * 1.1),
        est_cost_usd=cost_usd(args.model, int(est_in * 1.1), int(est_out * 1.1)),
    )
    if args.dry_run and todo:
        print("\n--- first request (user message) ---")
        print(render_batch(prompt, todo[: args.batch_size])[:1500])
        print("--- end ---\n")
    # The cap covers per-line retries too, so give it headroom over the batch count.
    if not confirm_spend(
        plan,
        max_requests=args.max_requests,
        dry_run=args.dry_run,
        assume_yes=args.yes,
    ):
        _write_output(rows, cache, args.model, prompt, args.out)
        return 0

    import anthropic

    if settings.anthropic_api_key is None:
        print("ANTHROPIC_API_KEY is not set (put it in .env).", file=sys.stderr)
        return 2
    client = anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value(), max_retries=4, timeout=120
    )

    def on_label(row: dict, label: TriageLabel) -> None:
        entry = {
            "cache_key": cache_key(args.model, prompt, row),
            "id": row["id"],
            "model": args.model,
            "prompt": prompt.ref,
            "label": label.model_dump(mode="json"),
        }
        cache[entry["cache_key"]] = entry
        append_jsonl(LABEL_CACHE, entry)

    started = time.perf_counter()
    budget = RequestBudget(args.max_requests)
    stats = asyncio.run(
        run_labeling(
            client, args.model, prompt, todo, args.batch_size, args.concurrency, budget, on_label
        )
    )
    elapsed = time.perf_counter() - started
    actual = cost_usd(args.model, stats.input_tokens, stats.output_tokens)
    print(
        f"labeled {stats.labeled}/{len(todo)} in {elapsed:.1f}s | requests {stats.requests} | "
        f"tokens {stats.input_tokens:,} in / {stats.output_tokens:,} out | cost ${actual:.4f} | "
        f"invalid items {stats.invalid}"
    )
    for err in stats.errors[:10]:
        print(f"  error: {err}")
    append_jsonl(
        REPORTS / "labeling_runs.jsonl",
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": args.model,
            "prompt": prompt.ref,
            "lines": len(todo),
            "labeled": stats.labeled,
            "requests": stats.requests,
            "input_tokens": stats.input_tokens,
            "output_tokens": stats.output_tokens,
            "cost_usd": round(actual, 6),
            "seconds": round(elapsed, 1),
            "invalid_items": stats.invalid,
        },
    )
    _write_output(rows, cache, args.model, prompt, args.out)
    return 0


def _write_output(rows: list[dict], cache: dict, model: str, prompt: Prompt, out: Path) -> None:
    labeled = []
    for r in rows:
        hit = cache.get(cache_key(model, prompt, r))
        if hit:
            labeled.append(
                {**r, "label": hit["label"], "label_source": f"teacher:{model}:{prompt.version}"}
            )
    if labeled:
        write_jsonl(out, labeled)
        print(f"{len(labeled)} labeled lines written to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
