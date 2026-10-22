from loglens.llmops.canary import decide, summarise


def row(slot, fallback=None, conf=0.9, latency=100.0, cache="miss", status=200, model="m"):
    return {
        "slot": slot,
        "cache": cache,
        "status_code": status,
        "confidence": conf,
        "fallback_reason": fallback,
        "latency_ms": latency,
        "model_version": model,
    }


def test_summarise_excludes_cache_hits_and_errors():
    rows = [
        row("stable"),
        row("stable", fallback="low_confidence"),
        row("stable", cache="exact"),
        row("stable", status=503),
    ]
    s = summarise(rows, "stable")
    assert s.n == 2
    assert s.fallback_rate == 0.5


def test_promote_when_canary_is_not_worse():
    rows = [row("stable", fallback="low_confidence" if i % 10 == 0 else None) for i in range(300)]
    rows += [
        row("canary", fallback="low_confidence" if i % 20 == 0 else None, model="v2")
        for i in range(300)
    ]
    ok, reasons = decide(summarise(rows, "stable"), summarise(rows, "canary"), 200, 0.02)
    assert ok and reasons == []


def test_hold_when_canary_worse_or_too_small():
    rows = [row("stable") for _ in range(300)]
    rows += [row("canary", fallback="invalid_json", latency=500) for _ in range(50)]
    ok, reasons = decide(summarise(rows, "stable"), summarise(rows, "canary"), 200, 0.02)
    assert not ok
    joined = " ".join(reasons)
    assert (
        "only 50" in joined
        and "fallback rate" in joined
        and "invalid JSON" in joined
        and "latency" in joined
    )
