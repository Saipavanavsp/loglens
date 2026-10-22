"""Per-API-key rate limiting (token bucket) and daily budgets, both in Redis."""

import time
from dataclasses import dataclass

# Atomic token bucket. KEYS[1] = bucket hash. ARGV = capacity, refill/s, now, cost.
# Stores tokens + last refill time; refills lazily on each call.
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil then
  tokens = capacity
  ts = now
end
tokens = math.min(capacity, tokens + math.max(0, now - ts) * rate)
local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end
redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, math.ceil(capacity / rate) + 60)
local retry_after = 0
if allowed == 0 then
  retry_after = (cost - tokens) / rate
end
return {allowed, tostring(tokens), tostring(retry_after)}
"""  # noqa: S105 - Lua source, not a secret


@dataclass
class RateDecision:
    allowed: bool
    remaining: float
    retry_after_s: float


class RateLimiter:
    """Token bucket: `capacity` burst, refilled at `refill_per_s` tokens per second.

    Done in a Lua script so check-and-decrement is atomic across gateway replicas.
    """

    def __init__(self, redis, capacity: int, refill_per_s: float, prefix: str = "rl"):
        self.r = redis
        self.capacity = capacity
        self.rate = refill_per_s
        self.prefix = prefix
        self._script = redis.register_script(_TOKEN_BUCKET_LUA)

    async def check(self, key_id: str, cost: int = 1, now: float | None = None) -> RateDecision:
        now = time.time() if now is None else now
        allowed, remaining, retry = await self._script(
            keys=[f"{self.prefix}:{key_id}"], args=[self.capacity, self.rate, now, cost]
        )
        return RateDecision(bool(int(allowed)), float(remaining), float(retry))


class BudgetTracker:
    """Daily token and USD budgets per key. Counters reset by date in the key name."""

    def __init__(self, redis, daily_tokens: int, daily_cost_usd: float, prefix: str = "budget"):
        self.r = redis
        self.daily_tokens = daily_tokens
        self.daily_cost_usd = daily_cost_usd
        self.prefix = prefix

    def _keys(self, key_id: str, day: str | None) -> tuple[str, str]:
        day = day or time.strftime("%Y-%m-%d", time.gmtime())
        return f"{self.prefix}:{key_id}:{day}:tokens", f"{self.prefix}:{key_id}:{day}:usd"

    async def usage(self, key_id: str, day: str | None = None) -> tuple[int, float]:
        tk, ck = self._keys(key_id, day)
        tokens, usd = await self.r.mget([tk, ck])
        return int(float(tokens or 0)), float(usd or 0.0)

    async def can_spend_tokens(self, key_id: str, day: str | None = None) -> bool:
        tokens, _ = await self.usage(key_id, day)
        return tokens < self.daily_tokens

    async def can_spend_external(
        self, key_id: str, est_cost_usd: float, day: str | None = None
    ) -> bool:
        """Checked before calling a paid provider."""
        tokens, usd = await self.usage(key_id, day)
        # Strictly below the limit: a limit of 0 means "no external calls at all".
        return (
            tokens < self.daily_tokens
            and usd < self.daily_cost_usd
            and usd + est_cost_usd <= self.daily_cost_usd
        )

    async def record(
        self, key_id: str, tokens: int, cost_usd: float, day: str | None = None
    ) -> None:
        tk, ck = self._keys(key_id, day)
        pipe = self.r.pipeline()
        pipe.incrby(tk, int(tokens))
        pipe.incrbyfloat(ck, float(cost_usd))
        pipe.expire(tk, 2 * 86400)
        pipe.expire(ck, 2 * 86400)
        await pipe.execute()
