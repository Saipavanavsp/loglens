"""Triage routing: cache -> local model -> (maybe) Claude.

For one log line:

1. Pick a slot. A configurable share of traffic goes to the canary model or
   prompt, chosen by hashing the request id so the choice is reproducible.
2. Cache lookup (exact, then semantic) in that slot's namespace.
3. Ask the local model, with timeout, retries and a circuit breaker.
4. Decide whether to trust it. Fall back to Claude if:
     invalid_json     - no parseable, schema-valid answer
     low_confidence   - confidence below the threshold
     timeout / local_unavailable / circuit_open / local_error
5. Before anything goes to Claude: check the key's daily budget and redact
   PII and secrets. Placeholders in Claude's answer are mapped back to the
   original values for the caller.
6. If the fallback is disabled, over budget or failing, return the local
   answer marked degraded=True when there is one; otherwise raise.
7. Cache only trusted, non-degraded answers.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field

from loglens.gateway.cache import TriageCache
from loglens.gateway.confidence import agreement_confidence, logprob_confidence
from loglens.gateway.limits import BudgetTracker
from loglens.gateway.observability import Metrics, span
from loglens.gateway.providers.base import Completion, Provider, ProviderError
from loglens.gateway.redaction import redact, restore
from loglens.gateway.resilience import CircuitBreaker, CircuitOpenError, call_with_retries
from loglens.gateway.settings import GatewayConfig
from loglens.jsonparse import parse_label
from loglens.pricing import budget_price_for, estimate_tokens
from loglens.prompts import load_prompt, triage_messages
from loglens.schema import TriageLabel, triage_json_schema

log = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    pass


class TriageUnavailableError(RuntimeError):
    pass


@dataclass
class TriageOutcome:
    label: TriageLabel | None
    provider: str | None
    model_version: str | None
    prompt_version: str
    slot: str
    cache: str = "miss"
    confidence: float | None = None
    fallback_reason: str | None = None
    degraded: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    redactions: int = 0
    valid_json: bool = True
    calls: list[str] = field(default_factory=list)


@dataclass
class SlotProviders:
    stable: Provider
    canary: Provider | None = None


class TriageService:
    def __init__(
        self,
        cfg: GatewayConfig,
        local: SlotProviders,
        fallback: Provider | None,
        cache: TriageCache | None,
        budget: BudgetTracker | None,
        metrics: Metrics,
        tracer,
    ):
        self.cfg = cfg
        self.local = local
        self.fallback = fallback
        self.cache = cache
        self.budget = budget
        self.metrics = metrics
        self.tracer = tracer
        b = cfg.circuit_breaker
        self.breakers = {
            name: CircuitBreaker(name, b.failure_threshold, b.reset_after_s)
            for name in ("local_stable", "local_canary", "fallback")
        }
        self.schema = triage_json_schema()

    # --- helpers -------------------------------------------------------------

    def choose_slot(self, request_id: str) -> str:
        pct = self.cfg.canary_percent
        if pct <= 0 or self.local.canary is None:
            return "stable"
        bucket = int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) % 10_000
        return "canary" if bucket < pct * 100 else "stable"

    def _slot(self, slot: str) -> tuple[Provider, str, CircuitBreaker]:
        if slot == "canary" and self.local.canary is not None:
            return (
                self.local.canary,
                self.cfg.prompt.get("canary", "v1"),
                self.breakers["local_canary"],
            )
        return self.local.stable, self.cfg.prompt.get("stable", "v1"), self.breakers["local_stable"]

    async def _call(
        self, provider: Provider, breaker: CircuitBreaker, timeout_s: float, **kwargs
    ) -> Completion:
        r = self.cfg.retries
        try:
            return await call_with_retries(
                lambda: provider.complete(**kwargs),
                attempts=r.attempts,
                timeout_s=timeout_s,
                base_delay_s=r.base_delay_s,
                max_delay_s=r.max_delay_s,
                breaker=breaker,
            )
        finally:
            self.metrics.breaker_open.labels(provider.name).set(int(breaker.state == "open"))

    async def _confidence(
        self, provider: Provider, messages, completion: Completion, label: TriageLabel
    ):
        method = self.cfg.routing.confidence_method
        if method == "logprobs":
            return logprob_confidence(completion.text, completion.logprobs)
        if method == "self_consistency":
            n = self.cfg.routing.self_consistency_samples

            async def sample():
                try:
                    c = await provider.complete(messages=messages, max_tokens=256, temperature=0.7)
                    return parse_label(c.text).label
                except (ConnectionError, TimeoutError, ProviderError):
                    return None

            samples = await asyncio.gather(*(sample() for _ in range(n)))
            return agreement_confidence(label, samples)
        return None

    # --- main entry point ------------------------------------------------------

    async def triage(
        self, message: str, source: str, key_id: str, request_id: str
    ) -> TriageOutcome:
        slot = self.choose_slot(request_id)
        provider, prompt_version, breaker = self._slot(slot)
        prompt = load_prompt("triage", prompt_version)
        out = TriageOutcome(None, None, None, prompt_version, slot)
        namespace = f"{prompt_version}:{provider.model}"

        with span(
            self.tracer, "triage", slot=slot, source=source, prompt_version=prompt_version
        ) as root:
            if self.cache is not None:
                with span(self.tracer, "cache.lookup"):
                    hit = await self.cache.get(namespace, message)
                if hit is not None:
                    v = hit.value
                    out.label = TriageLabel.model_validate(v["label"])
                    out.provider, out.model_version = v["provider"], v["model_version"]
                    out.confidence, out.cache = v.get("confidence"), hit.kind
                    root.set_attribute("cache", hit.kind)
                    return out

            if self.budget is not None and not await self.budget.can_spend_tokens(key_id):
                raise BudgetExceededError("daily token budget exhausted")

            # 1. local model
            messages = triage_messages(prompt, source, message)
            local_label: TriageLabel | None = None
            local_conf: float | None = None
            reason: str | None = None
            with span(self.tracer, "provider.local", model=provider.model) as s:
                try:
                    comp = await self._call(
                        provider,
                        breaker,
                        self.cfg.timeouts.local_s,
                        messages=messages,
                        max_tokens=256,
                        want_logprobs=self.cfg.routing.confidence_method == "logprobs",
                        json_schema=self.schema,
                    )
                    out.calls.append(provider.name)
                    out.input_tokens += comp.input_tokens
                    out.output_tokens += comp.output_tokens
                    parsed = parse_label(comp.text)
                    if not parsed.ok:
                        reason = "invalid_json"
                        out.valid_json = False
                    else:
                        local_label = parsed.label
                        local_conf = await self._confidence(provider, messages, comp, local_label)
                        thr = self.cfg.routing.confidence_threshold
                        if local_conf is None:
                            if self.cfg.routing.unknown_confidence_falls_back:
                                reason = "unknown_confidence"
                        elif local_conf < thr:
                            reason = "low_confidence"
                except TimeoutError:
                    reason = "timeout"
                except CircuitOpenError:
                    reason = "circuit_open"
                except ConnectionError:
                    reason = "local_unavailable"
                except ProviderError as exc:
                    reason = "local_error"
                    log.warning("local provider error", extra={"error": str(exc)[:200]})
                s.set_attribute("fallback_reason", reason or "")
                if local_conf is not None:
                    s.set_attribute("confidence", local_conf)

            if reason is None and local_label is not None:
                self._finish(out, local_label, provider.name, provider.model, local_conf)
                await self._account(out, key_id, external=False)
                await self._store(namespace, message, out)
                return out

            # 2. fallback
            out.fallback_reason = reason
            self.metrics.fallbacks.labels(reason).inc()
            fb = await self._fallback(message, source, prompt, key_id, out)
            if fb is not None:
                self._finish(out, fb, self.fallback.name, self.fallback.model, None)
                await self._account(out, key_id, external=True)
                await self._store(namespace, message, out)
                return out

            await self._account(out, key_id, external=out.cost_usd > 0)
            if local_label is not None:
                # Better a flagged low-confidence answer than none at all.
                self._finish(out, local_label, provider.name, provider.model, local_conf)
                out.degraded = True
                return out
            raise TriageUnavailableError(f"no provider could answer (reason: {reason})")

    async def _fallback(
        self, message, source, prompt, key_id, out: TriageOutcome
    ) -> TriageLabel | None:
        if not self.cfg.routing.fallback_enabled or self.fallback is None:
            return None
        red = redact(message)
        out.redactions = red.count
        messages = triage_messages(prompt, source, red.text)
        if self.budget is not None:
            pin, pout = budget_price_for(self.fallback.model)
            est = (estimate_tokens(prompt.system + red.text) * pin + 300 * pout) / 1e6
            if not await self.budget.can_spend_external(key_id, est):
                out.fallback_reason = f"{out.fallback_reason}+budget_exhausted"
                return None
        with span(
            self.tracer, "provider.fallback", model=self.fallback.model, redactions=red.count
        ):
            try:
                comp = await self._call(
                    self.fallback,
                    self.breakers["fallback"],
                    self.cfg.timeouts.fallback_s,
                    messages=messages,
                    max_tokens=1024,
                    json_schema=self.schema,
                )
            except (TimeoutError, ConnectionError, CircuitOpenError, ProviderError) as exc:
                log.warning("fallback failed", extra={"error": type(exc).__name__})
                return None
        out.calls.append(self.fallback.name)
        out.input_tokens += comp.input_tokens
        out.output_tokens += comp.output_tokens
        out.cost_usd += comp.cost_usd
        parsed = parse_label(comp.text)
        if not parsed.ok:
            return None
        label = parsed.label
        if red.mapping:
            label = label.model_copy(
                update={
                    "root_cause": restore(label.root_cause, red.mapping),
                    "suggested_action": restore(label.suggested_action, red.mapping),
                }
            )
        return label

    def _finish(
        self, out: TriageOutcome, label, provider: str, model: str, conf: float | None
    ) -> None:
        out.label = label
        out.provider = provider
        out.model_version = model
        out.confidence = conf
        out.valid_json = True

    async def _account(self, out: TriageOutcome, key_id: str, external: bool) -> None:
        if self.budget is not None and (out.input_tokens or out.output_tokens):
            await self.budget.record(key_id, out.input_tokens + out.output_tokens, out.cost_usd)
        if out.cost_usd:
            self.metrics.cost.inc(out.cost_usd)

    async def _store(self, namespace: str, message: str, out: TriageOutcome) -> None:
        if self.cache is None or out.label is None or out.degraded:
            return
        value = {
            "label": out.label.model_dump(mode="json"),
            "provider": out.provider,
            "model_version": out.model_version,
            "confidence": out.confidence,
        }
        # Semantic hits serve *other* lines, so scrub this line's PII from the text.
        scrubbed = out.label.model_copy(
            update={
                "root_cause": redact(out.label.root_cause).text,
                "suggested_action": redact(out.label.suggested_action).text,
            }
        )
        semantic_value = {**value, "label": scrubbed.model_dump(mode="json")}
        await self.cache.put(namespace, message, value, semantic_value)
