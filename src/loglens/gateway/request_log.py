"""Write request logs to PostgreSQL off the request path.

Handlers put a dict on an in-memory queue and return immediately. A background
task drains the queue in batches. On shutdown, `close()` enqueues a stop marker
and waits for the worker to write everything queued before it, so no
in-flight batch is lost. If the database is down, rows are dropped (and
counted) rather than blocking traffic; logging must never take the gateway down.
"""

import asyncio
import logging

from sqlalchemy.ext.asyncio import async_sessionmaker

from loglens.db import RequestLog

log = logging.getLogger(__name__)

_STOP = object()


class RequestLogger:
    def __init__(
        self,
        sessions: async_sessionmaker | None,
        batch_size: int = 50,
        max_queue: int = 10_000,
        close_timeout_s: float = 10.0,
    ):
        self._sessions = sessions
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._batch_size = batch_size
        self._close_timeout_s = close_timeout_s
        self._task: asyncio.Task | None = None
        self.dropped = 0
        self.written = 0

    def start(self) -> None:
        if self._sessions is not None and self._task is None:
            self._task = asyncio.create_task(self._run(), name="request-logger")

    def submit(self, row: dict) -> None:
        if self._sessions is None:
            return
        try:
            self._queue.put_nowait(row)
        except asyncio.QueueFull:
            self.dropped += 1

    async def _write(self, rows: list[dict]) -> None:
        try:
            async with self._sessions() as session, session.begin():
                session.add_all(RequestLog(**r) for r in rows)
            self.written += len(rows)
        except Exception:  # database problems must not crash the logger
            self.dropped += len(rows)
            log.exception("request log write failed; dropped %d rows", len(rows))

    async def _run(self) -> None:
        stop = False
        while not stop:
            item = await self._queue.get()
            if item is _STOP:
                break
            rows = [item]
            while len(rows) < self._batch_size and not self._queue.empty():
                nxt = self._queue.get_nowait()
                if nxt is _STOP:
                    stop = True
                    break
                rows.append(nxt)
            await self._write(rows)

    async def close(self) -> None:
        if self._task is None:
            return
        await self._queue.put(_STOP)
        try:
            await asyncio.wait_for(self._task, timeout=self._close_timeout_s)
        except TimeoutError:
            self._task.cancel()
            log.warning("request logger did not flush within %.0fs", self._close_timeout_s)
        self._task = None
