"""Compare canary vs stable traffic from the request log before promoting.

Production signals come from request_log rows grouped by slot (stable/canary)
over a time window. Cache hits are excluded (they don't exercise the model).
There are no labels in production, so the signals are proxies:

  fallback_rate      share of requests the local model couldn't answer confidently
  invalid_json_rate  share of local answers that failed schema validation
  mean_confidence    average local confidence on answered requests
  latency_p95_ms     end-to-end p95
  degraded_rate      share returned without a trusted answer

Decision rule (simple and explicit): recommend promotion only if the canary has
at least --min-requests requests, its fallback and invalid-JSON rates are no
worse than stable + tolerance, and its p95 latency is no more than 20% worse.
Accuracy is checked separately by running the golden-set eval on both models;
this script prints that command.

  python -m loglens.llmops.canary --hours 24
"""

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from loglens.config import get_settings
from loglens.db import RequestLog, make_engine, sessionmaker
from loglens.evals.metrics import percentile


@dataclass
class SlotStats:
    slot: str
    n: int
    fallback_rate: float
    invalid_json_rate: float
    degraded_rate: float
    mean_confidence: float | None
    latency_p95_ms: float | None
    models: list[str]


def summarise(rows: list[dict], slot: str) -> SlotStats:
    rs = [r for r in rows if r["slot"] == slot and r["cache"] == "miss" and r["status_code"] == 200]
    n = len(rs)
    conf = [r["confidence"] for r in rs if r["confidence"] is not None]
    return SlotStats(
        slot=slot,
        n=n,
        fallback_rate=sum(r["fallback_reason"] is not None for r in rs) / n if n else 0.0,
        invalid_json_rate=(
            sum((r["fallback_reason"] or "").startswith("invalid_json") for r in rs) / n
            if n
            else 0.0
        ),
        degraded_rate=sum(bool(r.get("degraded")) for r in rs) / n if n else 0.0,
        mean_confidence=sum(conf) / len(conf) if conf else None,
        latency_p95_ms=percentile([r["latency_ms"] for r in rs], 95),
        models=sorted({r["model_version"] for r in rs if r["model_version"]}),
    )


def decide(
    stable: SlotStats, canary: SlotStats, min_requests: int, tolerance: float
) -> tuple[bool, list[str]]:
    reasons = []
    if canary.n < min_requests:
        reasons.append(f"only {canary.n} canary requests (need {min_requests})")
    if canary.fallback_rate > stable.fallback_rate + tolerance:
        reasons.append(
            f"fallback rate {canary.fallback_rate:.1%} vs stable {stable.fallback_rate:.1%}"
        )
    if canary.invalid_json_rate > stable.invalid_json_rate + tolerance:
        reasons.append(
            f"invalid JSON {canary.invalid_json_rate:.1%} vs stable {stable.invalid_json_rate:.1%}"
        )
    if (
        stable.latency_p95_ms
        and canary.latency_p95_ms
        and canary.latency_p95_ms > 1.2 * stable.latency_p95_ms
    ):
        reasons.append(
            f"p95 latency {canary.latency_p95_ms:.0f} ms vs stable {stable.latency_p95_ms:.0f} ms"
        )
    return not reasons, reasons


async def load_rows(database_url: str, since: datetime) -> list[dict]:
    engine = make_engine(database_url)
    try:
        async with sessionmaker(engine)() as s:
            result = await s.execute(
                select(RequestLog).where(RequestLog.ts >= since, RequestLog.endpoint == "triage")
            )
            cols = (
                "slot",
                "cache",
                "status_code",
                "confidence",
                "fallback_reason",
                "latency_ms",
                "model_version",
                "degraded",
            )
            return [{c: getattr(r, c) for c in cols} for r in result.scalars()]
    finally:
        await engine.dispose()


def main() -> None:
    p = argparse.ArgumentParser(description="Canary vs stable comparison")
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--min-requests", type=int, default=200)
    p.add_argument(
        "--tolerance", type=float, default=0.02, help="allowed absolute increase in rates"
    )
    args = p.parse_args()
    since = datetime.now(UTC) - timedelta(hours=args.hours)
    rows = asyncio.run(load_rows(get_settings().database_url, since))
    stable, canary = summarise(rows, "stable"), summarise(rows, "canary")
    ok, reasons = decide(stable, canary, args.min_requests, args.tolerance)
    print(json.dumps({"stable": asdict(stable), "canary": asdict(canary)}, indent=2))
    print("\nRecommendation:", "PROMOTE" if ok else "HOLD")
    for r in reasons:
        print(f"  - {r}")
    if canary.models:
        print(
            "\nAlso compare accuracy on the golden set before promoting:\n"
            f"  python -m loglens.evals.run --systems finetuned --finetuned-tag {canary.models[0]}"
        )


if __name__ == "__main__":
    main()
