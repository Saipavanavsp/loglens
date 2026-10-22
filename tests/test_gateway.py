import asyncio

import pytest
from fastapi.testclient import TestClient

from gateway_fakes import (
    GOOD,
    HEADERS,
    FakeProvider,
    build,
    good_claude,
    good_local,
    label_text,
    make_cfg,
)
from loglens.gateway.providers.base import Completion

LINE = "Invalid user oracle from 10.0.0.5"


def post(client, json, headers=HEADERS):
    return client.post("/v1/triage", json=json, headers=headers)


# --- routing and fallback -----------------------------------------------------


def test_confident_local_answer_is_used():
    claude = FakeProvider("claude", "claude-test", external=True, default=good_claude())
    app, _ = build(fallback=claude)
    with TestClient(app) as c:
        res = post(c, {"message": LINE, "source": "openssh"}).json()["results"][0]
    assert res["provider"] == "local"
    assert res["fallback_reason"] is None
    assert res["confidence"] == pytest.approx(0.95)
    assert claude.calls == []


@pytest.mark.parametrize(
    ("local_item", "reason"),
    [
        (Completion(text="Sure! The severity is high.", model="m"), "invalid_json"),
        (good_local(p=0.3), "low_confidence"),
        (ConnectionError("down"), "local_unavailable"),
        ("sleep", "timeout"),
    ],
)
def test_fallback_reasons(local_item, reason):
    local = FakeProvider(default=local_item)
    claude = FakeProvider("claude", "claude-test", external=True, default=good_claude())
    app, _ = build(local=local, fallback=claude)
    with TestClient(app) as c:
        res = post(c, {"message": LINE}).json()["results"][0]
    assert res["provider"] == "claude"
    assert res["fallback_reason"] == reason
    assert len(claude.calls) == 1


def test_fallback_receives_redacted_text_and_answer_is_restored():
    claude_answer = {**GOOD, "root_cause": "User <USER_1> failed from <IP_1>."}
    local = FakeProvider(default=Completion(text="not json", model="m"))
    claude = FakeProvider(
        "claude", "claude-test", external=True, default=good_claude(label_text(claude_answer))
    )
    app, _ = build(local=local, fallback=claude)
    with TestClient(app) as c:
        res = post(c, {"message": LINE}).json()["results"][0]
    sent = " ".join(m["content"] for m in claude.calls[0])
    assert "10.0.0.5" not in sent and "oracle" not in sent
    assert "<IP_1>" in sent and "<USER_1>" in sent
    assert res["root_cause"] == "User oracle failed from 10.0.0.5."


def test_degraded_local_answer_when_fallback_fails():
    local = FakeProvider(default=good_local(p=0.2))
    claude = FakeProvider("claude", "claude-test", external=True, default=ConnectionError("x"))
    app, _ = build(local=local, fallback=claude)
    with TestClient(app) as c:
        res = post(c, {"message": LINE}).json()["results"][0]
    assert res["provider"] == "local"
    assert res["degraded"] is True
    assert res["fallback_reason"] == "low_confidence"


def test_503_when_nothing_can_answer():
    local = FakeProvider(default=ConnectionError("down"))
    app, _ = build(local=local, fallback=None)
    with TestClient(app) as c:
        r = post(c, {"message": LINE})
    assert r.status_code == 503


def test_fallback_disabled_by_config():
    claude = FakeProvider("claude", "claude-test", external=True, default=good_claude())
    cfg = make_cfg(routing={"fallback_enabled": False})
    app, _ = build(local=FakeProvider(default=good_local(p=0.1)), fallback=claude, cfg=cfg)
    with TestClient(app) as c:
        res = post(c, {"message": LINE}).json()["results"][0]
    assert res["degraded"] is True and claude.calls == []


def test_circuit_breaker_opens_after_repeated_failures():
    local = FakeProvider(default=ConnectionError("down"))
    claude = FakeProvider("claude", "claude-test", external=True, default=good_claude())
    cfg = make_cfg(circuit_breaker={"failure_threshold": 2, "reset_after_s": 60})
    app, _ = build(local=local, fallback=claude, cfg=cfg, use_cache=False)
    with TestClient(app) as c:
        reasons = [
            post(c, {"message": f"{LINE} {i}"}).json()["results"][0]["fallback_reason"]
            for i in range(4)
        ]
    assert reasons[:2] == ["local_unavailable", "local_unavailable"]
    assert reasons[2:] == ["circuit_open", "circuit_open"]
    assert len(local.calls) == 2  # breaker stopped further calls


def test_canary_routing_split():
    stable = FakeProvider("local", "stable-model", default=good_local())
    canary = FakeProvider("local_canary", "canary-model", default=good_local())
    cfg = make_cfg(
        models={"canary": {"ollama_tag": "canary-model", "registry_version": "2"}},
        canary={"percent": 30},
    )
    app, _ = build(local=stable, canary=canary, cfg=cfg, use_cache=False)
    with TestClient(app) as c:
        slots = [
            post(c, {"message": f"line {i}"}).json()["results"][0]["model_version"]
            for i in range(200)
        ]
    share = slots.count("canary-model") / len(slots)
    assert 0.15 < share < 0.45


# --- cache ---------------------------------------------------------------------


def test_exact_then_semantic_cache_hits():
    local = FakeProvider(default=good_local())
    app, deps = build(local=local)
    with TestClient(app) as c:
        first = post(c, {"message": LINE}).json()["results"][0]
        second = post(c, {"message": LINE}).json()["results"][0]
        third = post(c, {"message": "Invalid user oracle from 192.168.1.77"}).json()["results"][0]
        stats = c.get("/v1/stats", headers=HEADERS).json()["cache"]
    assert first["cache"] == "miss" and second["cache"] == "exact" and third["cache"] == "semantic"
    assert len(local.calls) == 1
    # The semantic hit must not leak the first request's IP.
    assert "10.0.0.5" not in third["root_cause"] + third["suggested_action"]
    assert stats["exact_hit"] == 1 and stats["semantic_hit"] == 1 and stats["miss"] == 1


def test_low_confidence_degraded_answers_are_not_cached():
    local = FakeProvider(default=good_local(p=0.1))
    app, _ = build(local=local, fallback=None)
    with TestClient(app) as c:
        post(c, {"message": LINE})
        res = post(c, {"message": LINE}).json()["results"][0]
    assert res["cache"] == "miss"
    assert len(local.calls) == 2


# --- auth, rate limits, budgets ---------------------------------------------------


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"X-API-Key": "nope"}])
def test_auth_required(headers):
    app, _ = build()
    with TestClient(app) as c:
        assert post(c, {"message": LINE}, headers=headers).status_code == 401


def test_x_api_key_header_works():
    app, _ = build()
    with TestClient(app) as c:
        assert post(c, {"message": LINE}, headers={"X-API-Key": "test-key-123"}).status_code == 200


def test_rate_limit_returns_429_with_retry_after():
    cfg = make_cfg(rate_limit={"capacity": 2, "refill_per_s": 0.01})
    app, _ = build(cfg=cfg)
    with TestClient(app) as c:
        codes = [post(c, {"message": f"{LINE} {i}"}).status_code for i in range(3)]
        last = post(c, {"message": LINE})
    assert codes == [200, 200, 429]
    assert int(last.headers["Retry-After"]) >= 1


def test_batch_counts_against_rate_limit():
    cfg = make_cfg(rate_limit={"capacity": 3, "refill_per_s": 0.01})
    app, _ = build(cfg=cfg)
    with TestClient(app) as c:
        r = post(c, {"logs": [{"message": f"l{i}"} for i in range(4)]})
    assert r.status_code == 429


def test_cost_budget_blocks_fallback_but_returns_degraded():
    local = FakeProvider(default=good_local(p=0.2))
    claude = FakeProvider("claude", "claude-test", external=True, default=good_claude())
    cfg = make_cfg(budget={"daily_cost_usd": 0.0})
    app, _ = build(local=local, fallback=claude, cfg=cfg)
    with TestClient(app) as c:
        res = post(c, {"message": LINE}).json()["results"][0]
    assert claude.calls == []
    assert res["degraded"] is True
    assert "budget_exhausted" in res["fallback_reason"]


def test_token_budget_exhausted_returns_429():
    cfg = make_cfg(budget={"daily_tokens": 100})
    app, _ = build(cfg=cfg, use_cache=False)
    with TestClient(app) as c:
        assert post(c, {"message": "a"}).status_code == 200  # uses 160 tokens
        assert post(c, {"message": "b"}).status_code == 429


def test_spend_is_tracked_per_key():
    local = FakeProvider(
        default=Completion(text="bad", model="m", input_tokens=10, output_tokens=5)
    )
    claude = FakeProvider("claude", "claude-test", external=True, default=good_claude())
    app, _ = build(local=local, fallback=claude)
    with TestClient(app) as c:
        post(c, {"message": LINE})
        budget = c.get("/v1/stats", headers=HEADERS).json()["budget_today"]
    assert budget["tokens"] == 10 + 5 + 300 + 80
    assert budget["cost_usd"] == pytest.approx(0.002)


# --- API surface -------------------------------------------------------------------


def test_batch_results_keep_order():
    app, _ = build()
    with TestClient(app) as c:
        body = {"logs": [{"message": f"line {i}", "source": "linux"} for i in range(5)]}
        results = post(c, body).json()["results"]
    assert [r["index"] for r in results] == list(range(5))


def test_batch_too_large():
    cfg = make_cfg(batch={"max_items": 2})
    app, _ = build(cfg=cfg)
    with TestClient(app) as c:
        assert post(c, {"logs": [{"message": "a"}] * 3}).status_code == 413


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"message": "a", "logs": [{"message": "b"}]},
        {"logs": []},
        {"message": ""},
        {"message": "x" * 5000},
    ],
)
def test_invalid_bodies_rejected(body):
    app, _ = build()
    with TestClient(app) as c:
        assert post(c, body).status_code == 422


def test_openai_compatible_triage_model():
    app, _ = build()
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "loglens-triage", "messages": [{"role": "user", "content": LINE}]},
            headers=HEADERS,
        )
    body = r.json()
    assert r.status_code == 200
    assert body["object"] == "chat.completion"
    import json

    content = json.loads(body["choices"][0]["message"]["content"])
    assert set(content) == {"severity", "category", "root_cause", "suggested_action"}
    assert body["loglens"]["provider"] == "local"


def test_openai_direct_external_provider_is_redacted():
    claude = FakeProvider("claude", "claude-test", external=True, default=good_claude("hello"))
    app, _ = build(fallback=claude)
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "claude",
                "messages": [{"role": "user", "content": "ping 10.1.2.3 password=abc"}],
            },
            headers=HEADERS,
        )
    assert r.status_code == 200
    sent = claude.calls[0][0]["content"]
    assert "10.1.2.3" not in sent and "abc" not in sent
    assert r.json()["loglens"]["redactions"] == 2


def test_unknown_model_404():
    app, _ = build()
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "gpt-x", "messages": [{"role": "user", "content": "x"}]},
            headers=HEADERS,
        )
    assert r.status_code == 404


def test_health_ready_metrics():
    app, deps = build()
    with TestClient(app) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/readyz").status_code == 200
        post(c, {"message": LINE})
        text = c.get("/metrics").text
    assert "loglens_requests_total" in text
    deps.service.local.stable.healthy = False
    with TestClient(app) as c:
        assert c.get("/readyz").status_code == 503


# --- request logging -----------------------------------------------------------------


def test_requests_are_logged_to_database_and_flushed_on_shutdown(tmp_path):
    from sqlalchemy import select

    from loglens.db import RequestLog, create_tables, make_engine, sessionmaker

    # A file, not :memory:, because the test client runs its own event loop/connection.
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'log.db'}")
    asyncio.run(create_tables(engine))
    sessions = sessionmaker(engine)
    local = FakeProvider(default=Completion(text="bad", model="m"))
    claude = FakeProvider("claude", "claude-test", external=True, default=good_claude())
    app, _ = build(local=local, fallback=claude, sessions=sessions)
    with TestClient(app) as c:
        post(c, {"message": LINE})

    async def rows():
        async with sessions() as s:
            return (await s.execute(select(RequestLog))).scalars().all()

    logged = asyncio.run(rows())
    assert len(logged) == 1
    row = logged[0]
    assert row.provider == "claude" and row.fallback_reason == "invalid_json"
    assert row.valid_json is True  # final answer was valid
    assert row.degraded is False
    assert row.redactions == 2 and row.cost_usd == pytest.approx(0.002)
    assert row.key_id != "test-key-123"  # only the hash is stored
