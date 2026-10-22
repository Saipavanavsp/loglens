import asyncio
import math

import pytest

from loglens.gateway.confidence import agreement_confidence, logprob_confidence
from loglens.gateway.providers.base import TokenLogprob
from loglens.gateway.resilience import (
    BreakerState,
    CircuitBreaker,
    CircuitOpenError,
    call_with_retries,
)
from loglens.schema import TriageLabel


async def _no_sleep(_):
    return None


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_breaker_opens_then_half_opens_then_closes():
    clock = Clock()
    b = CircuitBreaker("x", failure_threshold=2, reset_after_s=10, clock=clock)
    b.record_failure()
    assert b.allow()
    b.record_failure()
    assert b.state == BreakerState.OPEN and not b.allow()
    clock.t = 10
    assert b.allow()  # one trial call
    assert not b.allow()  # second concurrent call is rejected
    b.record_success()
    assert b.state == BreakerState.CLOSED and b.allow()


def test_breaker_half_open_failure_reopens():
    clock = Clock()
    b = CircuitBreaker("x", failure_threshold=1, reset_after_s=5, clock=clock)
    b.record_failure()
    clock.t = 5
    assert b.allow()
    b.record_failure()
    assert b.state == BreakerState.OPEN
    assert not b.allow()


def test_retries_transient_errors_then_succeeds():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert asyncio.run(call_with_retries(flaky, attempts=3, sleep=_no_sleep)) == "ok"
    assert calls["n"] == 3


def test_does_not_retry_permanent_errors():
    calls = {"n": 0}

    async def bad():
        calls["n"] += 1
        raise ValueError("bug")

    with pytest.raises(ValueError):
        asyncio.run(call_with_retries(bad, attempts=3, sleep=_no_sleep))
    assert calls["n"] == 1


def test_timeout_counts_as_failure_and_opens_breaker():
    async def slow():
        await asyncio.sleep(1)

    b = CircuitBreaker("slow", failure_threshold=2)
    with pytest.raises(TimeoutError):
        asyncio.run(call_with_retries(slow, attempts=2, timeout_s=0.01, breaker=b, sleep=_no_sleep))
    assert b.state == BreakerState.OPEN
    with pytest.raises(CircuitOpenError):
        asyncio.run(call_with_retries(slow, attempts=1, breaker=b, sleep=_no_sleep))


def _lp(tokens):
    return [TokenLogprob(t, math.log(p)) for t, p in tokens]


def test_logprob_confidence_uses_first_value_token_and_min():
    tokens = _lp(
        [
            ('{"', 1.0),
            ("severity", 1.0),
            ('":', 1.0),
            (' "', 1.0),
            ("warn", 0.6),
            ("ing", 0.99),
            ('",', 1.0),
            (' "', 1.0),
            ("category", 1.0),
            ('":', 1.0),
            (' "', 1.0),
            ("auth", 0.9),
            ('"}', 1.0),
        ]
    )
    text = "".join(t.token for t in tokens)
    assert logprob_confidence(text, tokens) == pytest.approx(0.6)


def test_logprob_confidence_handles_token_spanning_quote():
    tokens = _lp(
        [
            ('{"severity": "', 1.0),
            ("error", 0.8),
            ('", "category": "', 1.0),
            ("disk", 0.95),
            ('"}', 1.0),
        ]
    )
    assert logprob_confidence("".join(t.token for t in tokens), tokens) == pytest.approx(0.8)


def test_logprob_confidence_none_without_signal():
    assert logprob_confidence("{}", None) is None
    assert logprob_confidence('{"severity": "info"}', _lp([('{"severity": "info"}', 0.9)])) is None


def test_agreement_confidence():
    base = {"root_cause": "x happened", "suggested_action": "do y"}
    g = TriageLabel(severity="error", category="disk", **base)
    same = TriageLabel(severity="error", category="disk", **base)
    diff = TriageLabel(severity="warning", category="disk", **base)
    assert agreement_confidence(g, [same, diff, None, same]) == 0.5
