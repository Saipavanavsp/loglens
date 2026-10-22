"""Retries with backoff, timeouts and a per-provider circuit breaker."""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum


class CircuitOpenError(RuntimeError):
    """Raised instead of calling a provider whose breaker is open."""


class BreakerState(StrEnum):
    CLOSED = "closed"  # normal: calls go through
    OPEN = "open"  # failing: calls are rejected immediately
    HALF_OPEN = "half_open"  # cooling off done: let one trial call through


@dataclass
class CircuitBreaker:
    """Stops hammering a provider that keeps failing.

    After `failure_threshold` consecutive failures the breaker opens and calls
    fail fast for `reset_after_s`. Then one trial call is allowed (half-open):
    success closes the breaker, failure opens it again.
    """

    name: str
    failure_threshold: int = 5
    reset_after_s: float = 30.0
    clock: Callable[[], float] = time.monotonic
    state: BreakerState = BreakerState.CLOSED
    failures: int = 0
    opened_at: float = 0.0
    _trial_in_flight: bool = field(default=False, repr=False)

    def allow(self) -> bool:
        if self.state == BreakerState.CLOSED:
            return True
        if self.state == BreakerState.OPEN and self.clock() - self.opened_at >= self.reset_after_s:
            self.state = BreakerState.HALF_OPEN
            self._trial_in_flight = False
        if self.state == BreakerState.HALF_OPEN and not self._trial_in_flight:
            self._trial_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self.state = BreakerState.CLOSED
        self.failures = 0
        self._trial_in_flight = False

    def record_failure(self) -> None:
        self.failures += 1
        if self.state == BreakerState.HALF_OPEN or self.failures >= self.failure_threshold:
            self.state = BreakerState.OPEN
            self.opened_at = self.clock()
        self._trial_in_flight = False


async def call_with_retries[T](
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    timeout_s: float = 10.0,
    base_delay_s: float = 0.2,
    max_delay_s: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError),
    breaker: CircuitBreaker | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run fn with a per-attempt timeout and exponential backoff with full jitter.

    Only errors in `retry_on` (plus timeouts) are retried; anything else is a
    bug or a permanent error and is raised straight away.
    """
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        if breaker is not None and not breaker.allow():
            raise CircuitOpenError(f"circuit open for {breaker.name}")
        try:
            result = await asyncio.wait_for(fn(), timeout=timeout_s)
        except retry_on as exc:
            last_exc = exc
            if breaker is not None:
                breaker.record_failure()
            if attempt + 1 < attempts:
                delay = min(max_delay_s, base_delay_s * 2**attempt)
                await sleep(random.uniform(0, delay))
            continue
        except Exception:
            if breaker is not None:
                breaker.record_failure()
            raise
        if breaker is not None:
            breaker.record_success()
        return result
    raise last_exc if last_exc else RuntimeError("no attempts made")
