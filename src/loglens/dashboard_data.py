"""Aggregations behind the Streamlit dashboard, kept free of Streamlit so they can be tested."""

import json
from pathlib import Path

import pandas as pd

# Fixed color per entity (reference palette slots 1-3, then 4); color follows the
# entity, so a filter that hides one provider never repaints the others.
PROVIDER_COLORS = {
    "local": "#2a78d6",
    "claude": "#eb6834",
    "cache": "#1baf7a",
    "local_canary": "#eda100",
}


def load_requests(engine, hours: float) -> pd.DataFrame:
    query = (
        "SELECT ts, provider, slot, cache, fallback_reason, latency_ms, cost_usd, input_tokens, "
        "output_tokens, confidence, degraded, status_code, model_version "
        "FROM request_log WHERE endpoint = 'triage' AND ts >= :since"
    )
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    from sqlalchemy import text

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"since": since.to_pydatetime()})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def route_label(df: pd.DataFrame) -> pd.Series:
    """Who actually served each request: cache, local model or the external fallback."""
    return df["provider"].where(df["cache"] == "miss", "cache").fillna("error")


def kpis(df: pd.DataFrame) -> dict:
    ok = df[df["status_code"] == 200]
    misses = ok[ok["cache"] == "miss"]
    return {
        "requests": int(len(df)),
        "errors": int((df["status_code"] != 200).sum()),
        "cost_usd": float(df["cost_usd"].sum()),
        "latency_p95_ms": float(ok["latency_ms"].quantile(0.95)) if len(ok) else None,
        "cache_hit_rate": float((ok["cache"] != "miss").mean()) if len(ok) else None,
        "fallback_rate": float(misses["fallback_reason"].notna().mean()) if len(misses) else None,
        "degraded_rate": (
            float(ok["degraded"].fillna(False).astype(bool).mean()) if len(ok) else None
        ),
    }


def over_time(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Per time bucket and route: request count, p95 latency, cost."""
    if df.empty:
        return pd.DataFrame(columns=["bucket", "route", "requests", "p95_ms", "cost_usd"])
    d = df.assign(route=route_label(df), bucket=df["ts"].dt.floor(freq))
    g = d.groupby(["bucket", "route"])
    return pd.DataFrame(
        {
            "requests": g.size(),
            "p95_ms": g["latency_ms"].quantile(0.95),
            "cost_usd": g["cost_usd"].sum(),
        }
    ).reset_index()


def fallback_reasons(df: pd.DataFrame) -> pd.DataFrame:
    reasons = df.loc[df["cache"] == "miss", "fallback_reason"].dropna()
    # "low_confidence+budget_exhausted" -> count under the primary reason.
    counts = reasons.str.split("+").str[0].value_counts()
    return counts.rename_axis("reason").reset_index(name="requests")


def latest_eval(reports_dir: Path) -> dict | None:
    runs = sorted(p for p in (reports_dir / "eval").glob("*/metrics.json"))
    if not runs:
        return None
    return json.loads(runs[-1].read_text(encoding="utf-8"))


def label_free_results(reports_dir: Path) -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((reports_dir / "label_free").glob("*.json"))
    ]
