"""Bounded, observable delivery queue for lab flow observations."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from src.application.collection.transport import ServiceRequestError, post_json


@dataclass
class DeliveryCounters:
    enqueued: int = 0
    delivered: int = 0
    retries: int = 0
    terminal_failures: int = 0
    queue_dropped: int = 0


@dataclass(frozen=True)
class DeliveryItem:
    document: Mapping[str, Any]
    run_id: str | None


class ObservationDispatcher:
    """Serialize delivery to protect the single-CPU inference boundary."""

    def __init__(
        self,
        *,
        endpoint: str,
        queue_size: int = 1000,
        workers: int = 1,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 0.25,
        sender: Callable[..., dict[str, Any]] = post_json,
    ) -> None:
        if queue_size < 1 or workers < 1 or retry_attempts < 1:
            raise ValueError("Queue size, workers, and retry attempts must be positive.")
        if retry_backoff_seconds < 0:
            raise ValueError("Retry backoff cannot be negative.")
        self.endpoint = endpoint
        self.workers = workers
        self.retry_attempts = retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.sender = sender
        self.queue: asyncio.Queue[DeliveryItem | None] = asyncio.Queue(queue_size)
        self.total = DeliveryCounters()
        self.by_run: defaultdict[str, DeliveryCounters] = defaultdict(DeliveryCounters)
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        if self._tasks:
            return
        self._tasks = [
            asyncio.create_task(self._worker(), name=f"observation-delivery-{index}")
            for index in range(self.workers)
        ]

    async def stop(self) -> None:
        if not self._tasks:
            return
        await self.queue.join()
        for _ in self._tasks:
            await self.queue.put(None)
        await asyncio.gather(*self._tasks)
        self._tasks.clear()

    def enqueue(self, document: Mapping[str, Any], *, run_id: str | None) -> bool:
        counters = self._counters(run_id)
        try:
            self.queue.put_nowait(DeliveryItem(dict(document), run_id))
        except asyncio.QueueFull:
            self.total.queue_dropped += 1
            if counters is not self.total:
                counters.queue_dropped += 1
            return False
        self.total.enqueued += 1
        if counters is not self.total:
            counters.enqueued += 1
        return True

    def metrics(self, run_id: str | None = None) -> dict[str, Any]:
        counters = self.total if run_id is None else self.by_run[run_id]
        return {
            **asdict(counters),
            "queue_depth": self.queue.qsize(),
            "queue_capacity": self.queue.maxsize,
        }

    def _counters(self, run_id: str | None) -> DeliveryCounters:
        return self.total if run_id is None else self.by_run[run_id]

    def _increment(self, field: str, run_id: str | None) -> None:
        setattr(self.total, field, getattr(self.total, field) + 1)
        if run_id is not None:
            counters = self.by_run[run_id]
            setattr(counters, field, getattr(counters, field) + 1)

    async def _worker(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    return
                await self._deliver(item)
            finally:
                self.queue.task_done()

    async def _deliver(self, item: DeliveryItem) -> None:
        for attempt in range(1, self.retry_attempts + 1):
            try:
                await asyncio.to_thread(self.sender, self.endpoint, item.document)
                self._increment("delivered", item.run_id)
                return
            except ServiceRequestError:
                if attempt >= self.retry_attempts:
                    self._increment("terminal_failures", item.run_id)
                    return
                self._increment("retries", item.run_id)
                await asyncio.sleep(self.retry_backoff_seconds * attempt)
