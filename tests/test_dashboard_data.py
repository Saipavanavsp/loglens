import pandas as pd
import pytest

from loglens.dashboard_data import fallback_reasons, kpis, over_time, route_label


def frame():
    ts = pd.Timestamp("2026-10-22T10:00:00Z")
    rows = [
        # provider, cache, fallback, latency, cost, status, degraded
        ("local", "miss", None, 2000, 0.0, 200, False),
        ("claude", "miss", "low_confidence", 4000, 0.003, 200, False),
        ("local", "exact", None, 5, 0.0, 200, False),
        ("local", "miss", "low_confidence+budget_exhausted", 2100, 0.0, 200, True),
        (None, "miss", "local_unavailable", 10, 0.0, 503, False),
    ]
    return pd.DataFrame(
        [
            {
                "ts": ts + pd.Timedelta(minutes=10 * i),
                "provider": p,
                "cache": c,
                "fallback_reason": f,
                "latency_ms": lat,
                "cost_usd": cost,
                "status_code": st,
                "degraded": deg,
            }
            for i, (p, c, f, lat, cost, st, deg) in enumerate(rows)
        ]
    )


def test_kpis():
    k = kpis(frame())
    assert k["requests"] == 5 and k["errors"] == 1
    assert k["cost_usd"] == pytest.approx(0.003)
    assert k["cache_hit_rate"] == pytest.approx(1 / 4)
    assert k["fallback_rate"] == pytest.approx(2 / 3)  # of the 3 successful cache misses
    assert k["degraded_rate"] == pytest.approx(1 / 4)


def test_route_label_and_over_time():
    df = frame()
    assert list(route_label(df)) == ["local", "claude", "cache", "local", "error"]
    t = over_time(df, "1h")
    assert t.groupby("route")["requests"].sum().to_dict() == {
        "cache": 1,
        "claude": 1,
        "error": 1,
        "local": 2,
    }


def test_fallback_reasons_use_primary_reason():
    fr = fallback_reasons(frame()).set_index("reason")["requests"].to_dict()
    assert fr == {"low_confidence": 2, "local_unavailable": 1}
