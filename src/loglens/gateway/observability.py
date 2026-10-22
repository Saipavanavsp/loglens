"""Structured JSON logging, Prometheus metrics and OpenTelemetry tracing."""

import json
import logging
import sys
import time
from contextlib import contextmanager

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra fields passed via `extra={...}` are included."""

    _skip = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message"}

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in self._skip:
                out[k] = v
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # Library loggers that would otherwise log request bodies/URLs at INFO.
    # (anthropic 2.x and huggingface_hub 1.x log through "httpx2".)
    for noisy in (
        "httpx",
        "httpx2",
        "httpcore",
        "anthropic",
        "huggingface_hub",
        "sentence_transformers",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Metrics:
    """Prometheus metrics, on a private registry so tests can create many apps."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "loglens_requests_total",
            "Triage requests by final provider and outcome",
            ["provider", "slot", "cache", "outcome"],
            registry=self.registry,
        )
        self.fallbacks = Counter(
            "loglens_fallbacks_total", "Fallbacks to the external provider by reason", ["reason"],
            registry=self.registry,
        )  # fmt: skip
        self.latency = Histogram(
            "loglens_request_latency_seconds",
            "End-to-end triage latency",
            ["provider"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
            registry=self.registry,
        )
        self.cost = Counter(
            "loglens_cost_usd_total", "External provider spend", registry=self.registry
        )
        self.tokens = Counter(
            "loglens_tokens_total", "Tokens by provider and direction", ["provider", "direction"],
            registry=self.registry,
        )  # fmt: skip
        self.breaker_open = Gauge(
            "loglens_circuit_open", "1 if the provider's breaker is open", ["provider"],
            registry=self.registry,
        )  # fmt: skip
        self.rate_limited = Counter(
            "loglens_rate_limited_total", "Rejected by rate limit", registry=self.registry
        )


def setup_tracing(service_name: str, endpoint: str | None):
    """Return a tracer. With no OTLP endpoint configured, spans are no-ops."""
    from opentelemetry import trace

    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
        )
        trace.set_tracer_provider(provider)
    return trace.get_tracer("loglens.gateway")


@contextmanager
def span(tracer, name: str, **attrs):
    """Start a span and set attributes, skipping None values."""
    with tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            if v is not None:
                s.set_attribute(k, v)
        yield s
