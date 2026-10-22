"""Test doubles for the gateway: scripted providers and an app builder."""

import asyncio
import json
import math

import fakeredis
from opentelemetry import trace

from loglens.gateway.app import Deps, create_app
from loglens.gateway.auth import ApiKeyAuth
from loglens.gateway.cache import HashingEmbedder, TriageCache
from loglens.gateway.limits import BudgetTracker, RateLimiter
from loglens.gateway.observability import Metrics
from loglens.gateway.providers.base import Completion, TokenLogprob
from loglens.gateway.request_log import RequestLogger
from loglens.gateway.router import SlotProviders, TriageService
from loglens.gateway.settings import GatewayConfig

API_KEY = "test-key-123"

GOOD = {
    "severity": "warning",
    "category": "security",
    "root_cause": "An unknown user tried to log in from 10.0.0.5.",
    "suggested_action": "Block 10.0.0.5 if attempts continue.",
}


def label_text(label: dict = GOOD) -> str:
    return json.dumps(label)


def logprobs_for(label: dict, p_sev: float, p_cat: float) -> list[TokenLogprob]:
    """Token stream for a JSON label with chosen probabilities on the value tokens."""
    parts = [
        ('{"severity": "', 1.0),
        (label["severity"], p_sev),
        ('", "category": "', 1.0),
        (label["category"], p_cat),
        (
            '", "root_cause": '
            + json.dumps(label["root_cause"])
            + ', "suggested_action": '
            + json.dumps(label["suggested_action"])
            + "}",
            1.0,
        ),
    ]
    return [TokenLogprob(t, math.log(p)) for t, p in parts]


class FakeProvider:
    """Returns scripted completions; each item may be a Completion, an exception, or a callable."""

    def __init__(self, name="local", model="fake-local", external=False, script=None, default=None):
        self.name = name
        self.model = model
        self.external = external
        self.script = list(script or [])
        self.default = default
        self.calls: list[list[dict]] = []
        self.healthy = True

    async def complete(
        self, messages, *, max_tokens=256, want_logprobs=False, temperature=None, json_schema=None
    ):
        self.calls.append(messages)
        item = self.script.pop(0) if self.script else self.default
        if callable(item) and not isinstance(item, Completion):
            item = item(messages)
        if isinstance(item, BaseException):
            raise item
        if item == "sleep":
            await asyncio.sleep(10)
        return item

    async def health(self):
        return self.healthy

    async def aclose(self):
        return None


def good_local(p: float = 0.95) -> Completion:
    return Completion(
        text=label_text(),
        model="fake-local",
        input_tokens=100,
        output_tokens=60,
        logprobs=logprobs_for(GOOD, p, p),
    )


def good_claude(text: str | None = None) -> Completion:
    return Completion(
        text=text or label_text(),
        model="claude-test",
        input_tokens=300,
        output_tokens=80,
        cost_usd=0.002,
    )


def make_cfg(**overrides) -> GatewayConfig:
    base = {
        "models": {"stable": {"ollama_tag": "fake-local", "registry_version": "t"}},
        "retries": {"attempts": 1, "base_delay_s": 0, "max_delay_s": 0},
        "timeouts": {"local_s": 0.2, "fallback_s": 0.2},
        "rate_limit": {"capacity": 1000, "refill_per_s": 1000},
    }
    for k, v in overrides.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return GatewayConfig.model_validate(base)


def build(
    local: FakeProvider | None = None,
    fallback: FakeProvider | None = None,
    cfg: GatewayConfig | None = None,
    canary: FakeProvider | None = None,
    sessions=None,
    use_cache: bool = True,
):
    cfg = cfg or make_cfg()
    local = local or FakeProvider(default=good_local())
    r = fakeredis.FakeAsyncRedis()
    metrics = Metrics()
    cache = (
        TriageCache(r, HashingEmbedder(), semantic_threshold=cfg.cache.semantic_threshold)
        if use_cache
        else None
    )
    budget = BudgetTracker(r, cfg.budget.daily_tokens, cfg.budget.daily_cost_usd)
    limiter = RateLimiter(r, cfg.rate_limit.capacity, cfg.rate_limit.refill_per_s)
    service = TriageService(
        cfg,
        SlotProviders(local, canary),
        fallback,
        cache,
        budget,
        metrics,
        trace.get_tracer("test"),
    )
    deps = Deps(
        cfg=cfg,
        service=service,
        limiter=limiter,
        budget=budget,
        cache=cache,
        request_logger=RequestLogger(sessions),
        metrics=metrics,
        auth=ApiKeyAuth({API_KEY}),
        redis=r,
    )
    return create_app(deps), deps


HEADERS = {"Authorization": f"Bearer {API_KEY}"}
