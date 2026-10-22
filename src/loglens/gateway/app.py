"""FastAPI application.

Endpoints:
  POST /v1/triage            one log line or a batch -> validated triage JSON
  POST /v1/chat/completions  OpenAI-compatible; model "loglens-triage" routes,
                             "local" / "claude" call one provider directly
  GET  /v1/models            models accepted by /v1/chat/completions
  GET  /v1/stats             cache hit rates and today's spend for the caller
  GET  /healthz              liveness (process is up)
  GET  /readyz               readiness (Redis, DB, local model reachable)
  GET  /metrics              Prometheus metrics

Run: uvicorn loglens.gateway.app:app --timeout-graceful-shutdown 20
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from loglens.config import Settings, get_settings
from loglens.gateway.api_models import ChatRequest, TriageRequest, TriageResponse, TriageResult
from loglens.gateway.auth import ApiKeyAuth
from loglens.gateway.cache import TriageCache, make_embedder
from loglens.gateway.limits import BudgetTracker, RateLimiter
from loglens.gateway.observability import Metrics, setup_logging, setup_tracing
from loglens.gateway.providers.base import Provider, ProviderError
from loglens.gateway.redaction import redact
from loglens.gateway.request_log import RequestLogger
from loglens.gateway.resilience import CircuitOpenError
from loglens.gateway.router import (
    BudgetExceededError,
    SlotProviders,
    TriageOutcome,
    TriageService,
    TriageUnavailableError,
)
from loglens.gateway.settings import GatewayConfig, load_gateway_config

log = logging.getLogger("loglens.gateway")


@dataclass
class Deps:
    cfg: GatewayConfig
    service: TriageService
    limiter: RateLimiter | None
    budget: BudgetTracker | None
    cache: TriageCache | None
    request_logger: RequestLogger
    metrics: Metrics
    auth: ApiKeyAuth
    redis: object | None = None
    engine: object | None = None
    closers: tuple = ()


async def build_deps(settings: Settings, cfg: GatewayConfig) -> Deps:
    """Wire real dependencies from environment + config."""
    import os

    import redis.asyncio as aioredis

    from loglens.db import create_tables, make_engine, sessionmaker
    from loglens.gateway.providers.claude import ClaudeProvider
    from loglens.gateway.providers.ollama import OllamaProvider

    metrics = Metrics()
    tracer = setup_tracing("loglens-gateway", settings.otel_endpoint)
    r = aioredis.from_url(settings.redis_url)
    engine = make_engine(settings.database_url)
    try:
        await create_tables(engine)
        sessions = sessionmaker(engine)
    except Exception:
        log.exception("database unavailable at startup; request logging disabled")
        sessions = None

    constrained = os.environ.get("LOGLENS_LOCAL_CONSTRAINED_JSON", "false").lower() == "true"
    stable = cfg.models["stable"]
    local = SlotProviders(
        stable=OllamaProvider(
            settings.ollama_base_url,
            stable.ollama_tag,
            "local",
            constrained_json=constrained,
            num_thread=settings.ollama_num_thread,
        )
    )
    if cfg.models.get("canary"):
        local.canary = OllamaProvider(
            settings.ollama_base_url,
            cfg.models["canary"].ollama_tag,
            "local_canary",
            constrained_json=constrained,
            num_thread=settings.ollama_num_thread,
        )
    fallback = None
    if cfg.routing.fallback_enabled and settings.anthropic_api_key is not None:
        fallback = ClaudeProvider(
            settings.anthropic_api_key.get_secret_value(), settings.fallback_model
        )

    embedder = None
    if cfg.cache.semantic_enabled:
        embedder = await asyncio.to_thread(
            make_embedder, os.environ.get("LOGLENS_EMBEDDER", "sentence-transformers")
        )
    c = cfg.cache
    cache = TriageCache(
        r,
        embedder,
        exact_ttl_s=c.exact_ttl_s,
        semantic_ttl_s=c.semantic_ttl_s,
        semantic_threshold=c.semantic_threshold,
        semantic_max_entries=c.semantic_max_entries,
    )
    budget = BudgetTracker(r, cfg.budget.daily_tokens, cfg.budget.daily_cost_usd)
    limiter = RateLimiter(r, cfg.rate_limit.capacity, cfg.rate_limit.refill_per_s)
    service = TriageService(cfg, local, fallback, cache, budget, metrics, tracer)
    providers = [p for p in (local.stable, local.canary, fallback) if p is not None]
    return Deps(
        cfg=cfg,
        service=service,
        limiter=limiter,
        budget=budget,
        cache=cache,
        request_logger=RequestLogger(sessions),
        metrics=metrics,
        auth=ApiKeyAuth(settings.api_key_set),
        redis=r,
        engine=engine,
        closers=tuple(p.aclose for p in providers),
    )


def create_app(deps: Deps | None = None) -> FastAPI:
    """App factory. Tests pass ready-made deps; production builds them at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        if deps is None:
            setup_logging()
            if not settings.api_key_set:
                raise RuntimeError("LOGLENS_API_KEYS is empty; refusing to start without auth")
            app.state.deps = await build_deps(settings, load_gateway_config())
        else:
            app.state.deps = deps
        d: Deps = app.state.deps
        d.request_logger.start()
        log.info("gateway started")
        try:
            yield
        finally:
            # Graceful shutdown: uvicorn has stopped accepting requests and waited
            # for in-flight ones; flush logs, then close connections.
            await d.request_logger.close()
            for close in d.closers:
                await close()
            if d.redis is not None:
                await d.redis.aclose()
            if d.engine is not None:
                await d.engine.dispose()
            log.info("gateway stopped")

    app = FastAPI(title="LogLens gateway", version="0.1.0", lifespan=lifespan)
    if get_settings().otel_endpoint:
        # One server span per HTTP request; router spans (cache, provider calls) nest under it.
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics")

    def get_deps(request: Request) -> Deps:
        return request.app.state.deps

    async def authenticated(request: Request) -> str:
        return await request.app.state.deps.auth(request)

    async def rate_limit(d: Deps, key_id: str, cost: int) -> None:
        if d.limiter is None:
            return
        decision = await d.limiter.check(key_id, cost)
        if not decision.allowed:
            d.metrics.rate_limited.inc()
            raise HTTPException(
                429,
                "rate limit exceeded",
                headers={"Retry-After": str(max(1, round(decision.retry_after_s)))},
            )

    def log_row(
        d: Deps, request_id: str, key_id: str, endpoint: str, out: TriageOutcome | None, **extra
    ) -> None:
        row = {"request_id": request_id, "key_id": key_id, "endpoint": endpoint, **extra}
        if out is not None:
            row.update(
                provider=out.provider,
                model_version=out.model_version,
                prompt_version=out.prompt_version,
                slot=out.slot,
                input_tokens=out.input_tokens,
                output_tokens=out.output_tokens,
                latency_ms=out.latency_ms,
                cost_usd=out.cost_usd,
                cache=out.cache,
                fallback_reason=out.fallback_reason,
                confidence=out.confidence,
                valid_json=out.valid_json,
                degraded=out.degraded,
                severity=out.label.severity.value if out.label else None,
                category=out.label.category.value if out.label else None,
                redactions=out.redactions,
            )
        d.request_logger.submit(row)

    async def run_one(
        d: Deps, index: int, message: str, source: str, key_id: str, request_id: str
    ) -> TriageResult:
        started = time.perf_counter()
        sub_id = f"{request_id}:{index}"
        try:
            out = await d.service.triage(message, source, key_id, sub_id)
        except BudgetExceededError as exc:
            log_row(d, sub_id, key_id, "triage", None, status_code=429, error=str(exc))
            return TriageResult(index=index, error="budget_exceeded")
        except TriageUnavailableError as exc:
            log_row(d, sub_id, key_id, "triage", None, status_code=503, error=str(exc))
            return TriageResult(index=index, error="unavailable")
        out.latency_ms = (time.perf_counter() - started) * 1000
        d.metrics.requests.labels(
            out.provider or "none", out.slot, out.cache, "degraded" if out.degraded else "ok"
        ).inc()
        d.metrics.latency.labels(out.provider or "none").observe(out.latency_ms / 1000)
        d.metrics.tokens.labels(out.provider or "none", "in").inc(out.input_tokens)
        d.metrics.tokens.labels(out.provider or "none", "out").inc(out.output_tokens)
        log_row(d, sub_id, key_id, "triage", out, status_code=200)
        return TriageResult(
            index=index,
            **out.label.model_dump(mode="json"),
            provider=out.provider,
            model_version=out.model_version,
            prompt_version=out.prompt_version,
            confidence=out.confidence,
            cache=out.cache,
            fallback_reason=out.fallback_reason,
            degraded=out.degraded,
            latency_ms=round(out.latency_ms, 1),
        )

    @app.post("/v1/triage", response_model=TriageResponse)
    async def triage(
        body: TriageRequest,
        request: Request,
        key_id: str = Depends(authenticated),
        d: Deps = Depends(get_deps),
    ):
        items = body.items()
        if len(items) > d.cfg.batch.max_items:
            raise HTTPException(413, f"batch too large (max {d.cfg.batch.max_items})")
        await rate_limit(d, key_id, cost=len(items))
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        sem = asyncio.Semaphore(d.cfg.batch.concurrency)

        async def guarded(i, item):
            async with sem:
                return await run_one(d, i, item.message, item.source, key_id, request_id)

        results = await asyncio.gather(*(guarded(i, it) for i, it in enumerate(items)))
        if len(results) == 1 and results[0].error:
            code = 429 if results[0].error == "budget_exceeded" else 503
            raise HTTPException(code, results[0].error)
        return TriageResponse(request_id=request_id, results=list(results))

    @app.get("/v1/models")
    async def models(d: Deps = Depends(get_deps), _: str = Depends(authenticated)):
        names = ["loglens-triage", d.service.local.stable.name]
        if d.service.fallback is not None:
            names.append(d.service.fallback.name)
        return {
            "object": "list",
            "data": [{"id": n, "object": "model", "owned_by": "loglens"} for n in names],
        }

    @app.post("/v1/chat/completions")
    async def chat(
        body: ChatRequest,
        request: Request,
        key_id: str = Depends(authenticated),
        d: Deps = Depends(get_deps),
    ):
        await rate_limit(d, key_id, cost=1)
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        created = int(time.time())
        if body.model == "loglens-triage":
            user_msgs = [m.content for m in body.messages if m.role == "user"]
            if not user_msgs:
                raise HTTPException(400, "need a user message containing the log line")
            res = await run_one(d, 0, user_msgs[-1], "unknown", key_id, request_id)
            if res.error:
                raise HTTPException(429 if res.error == "budget_exceeded" else 503, res.error)
            content = TriageResult.model_validate(res).model_dump_json(
                include={"severity", "category", "root_cause", "suggested_action"}
            )
            meta = res.model_dump(
                exclude={"severity", "category", "root_cause", "suggested_action", "index"}
            )
            return _chat_response(request_id, body.model, created, content, 0, 0, meta)

        provider = _direct_provider(d, body.model)
        if provider is None:
            raise HTTPException(404, f"unknown model '{body.model}'; see /v1/models")
        messages = [m.model_dump() for m in body.messages]
        redactions = 0
        if provider.external:
            if d.budget is not None and not await d.budget.can_spend_external(key_id, 0.01):
                raise HTTPException(429, "budget_exceeded")
            reds = [redact(m["content"]) for m in messages]
            redactions = sum(r.count for r in reds)
            messages = [{**m, "content": r.text} for m, r in zip(messages, reds, strict=True)]
        started = time.perf_counter()
        try:
            comp = await provider.complete(
                messages=messages, max_tokens=body.max_tokens, temperature=body.temperature
            )
        except (TimeoutError, ConnectionError, CircuitOpenError) as exc:
            raise HTTPException(503, f"{provider.name} unavailable") from exc
        except ProviderError as exc:
            raise HTTPException(502, str(exc)) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        if d.budget is not None:
            await d.budget.record(key_id, comp.input_tokens + comp.output_tokens, comp.cost_usd)
        d.request_logger.submit(
            {
                "request_id": request_id,
                "key_id": key_id,
                "endpoint": "chat",
                "provider": provider.name,
                "model_version": provider.model,
                "input_tokens": comp.input_tokens,
                "output_tokens": comp.output_tokens,
                "latency_ms": latency_ms,
                "cost_usd": comp.cost_usd,
                "redactions": redactions,
                "status_code": 200,
            }
        )
        return _chat_response(
            request_id,
            body.model,
            created,
            comp.text,
            comp.input_tokens,
            comp.output_tokens,
            {"redactions": redactions},
        )

    @app.get("/v1/stats")
    async def stats(key_id: str = Depends(authenticated), d: Deps = Depends(get_deps)):
        out: dict = {}
        if d.cache is not None:
            out["cache"] = await d.cache.stats()
        if d.budget is not None:
            tokens, usd = await d.budget.usage(key_id)
            out["budget_today"] = {
                "tokens": tokens,
                "tokens_limit": d.budget.daily_tokens,
                "cost_usd": round(usd, 6),
                "cost_limit_usd": d.budget.daily_cost_usd,
            }
        out["request_log"] = {
            "written": d.request_logger.written,
            "dropped": d.request_logger.dropped,
        }
        return out

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(d: Deps = Depends(get_deps)):
        checks: dict[str, bool] = {}
        if d.redis is not None:
            try:
                checks["redis"] = bool(await d.redis.ping())
            except Exception:
                checks["redis"] = False
        if d.engine is not None:
            try:
                async with d.engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                checks["database"] = True
            except Exception:
                checks["database"] = False
        checks["local_model"] = await d.service.local.stable.health()
        ok = checks.get("redis", True) and checks["local_model"]
        return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)

    @app.get("/metrics")
    async def metrics(d: Deps = Depends(get_deps)):
        return Response(generate_latest(d.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def _direct_provider(d: Deps, name: str) -> Provider | None:
    for p in (d.service.local.stable, d.service.local.canary, d.service.fallback):
        if p is not None and p.name == name:
            return p
    return None


def _chat_response(request_id, model, created, content, pin, pout, meta) -> dict:
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout},
        "loglens": meta,
    }


app = create_app()
