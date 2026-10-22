"""Typed view of configs/gateway.yaml."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "configs" / "gateway.yaml"


class ModelSlot(BaseModel):
    ollama_tag: str
    registry_version: str = "unknown"


class Routing(BaseModel):
    confidence_method: str = "logprobs"
    confidence_threshold: float = Field(0.7, ge=0, le=1)
    self_consistency_samples: int = 3
    unknown_confidence_falls_back: bool = False
    fallback_enabled: bool = True


class Timeouts(BaseModel):
    local_s: float = 20.0
    fallback_s: float = 30.0


class Retries(BaseModel):
    attempts: int = 2
    base_delay_s: float = 0.2
    max_delay_s: float = 2.0


class Breaker(BaseModel):
    failure_threshold: int = 5
    reset_after_s: float = 30.0


class CacheCfg(BaseModel):
    exact_ttl_s: int = 86400
    semantic_enabled: bool = True
    semantic_threshold: float = 0.95
    semantic_ttl_s: int = 21600
    semantic_max_entries: int = 5000


class RateLimit(BaseModel):
    capacity: int = 60
    refill_per_s: float = 1.0


class Budget(BaseModel):
    daily_tokens: int = 2_000_000
    daily_cost_usd: float = 5.0


class Batch(BaseModel):
    max_items: int = 100
    concurrency: int = 4


class GatewayConfig(BaseModel):
    prompt: dict[str, str] = {"stable": "v1", "canary": "v1"}
    models: dict[str, ModelSlot | None] = {}
    canary: dict[str, float] = {"percent": 0}
    routing: Routing = Routing()
    timeouts: Timeouts = Timeouts()
    retries: Retries = Retries()
    circuit_breaker: Breaker = Breaker()
    cache: CacheCfg = CacheCfg()
    rate_limit: RateLimit = RateLimit()
    budget: Budget = Budget()
    batch: Batch = Batch()

    @property
    def canary_percent(self) -> float:
        return float(self.canary.get("percent", 0)) if self.models.get("canary") else 0.0


def load_gateway_config(path: Path = DEFAULT_PATH) -> GatewayConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return GatewayConfig.model_validate(data)
