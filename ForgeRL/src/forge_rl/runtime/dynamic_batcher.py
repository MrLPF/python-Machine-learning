from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class BatchEnvelope(Generic[T]):
    payload: T
    item_count: int
    enqueued_at: float
    deadline: float


class DeadlineBatcher(Generic[T]):
    """Bounded, overflow-safe dynamic batch queue.

    The oldest request sets the latency deadline. A batch is released when it reaches
    `max_items`, reaches `min_items`, or the oldest deadline expires. A request that would
    overflow the current batch remains queued for the next batch.
    """

    def __init__(
        self,
        *,
        max_items: int,
        min_items: int = 1,
        max_wait_ms: float = 2.0,
        max_queued_items: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        if min_items <= 0 or min_items > max_items:
            raise ValueError("min_items must be in [1, max_items]")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms cannot be negative")
        self.max_items = int(max_items)
        self.min_items = int(min_items)
        self.max_wait_seconds = float(max_wait_ms) / 1000.0
        self.max_queued_items = int(max_queued_items or max_items * 16)
        self._clock = clock
        self._queue: deque[BatchEnvelope[T]] = deque()
        self._queued_items = 0
        self._closed = False
        self._condition = threading.Condition()

    @property
    def queued_items(self) -> int:
        with self._condition:
            return self._queued_items

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def submit(
        self,
        payload: T,
        *,
        item_count: int = 1,
        timeout: float | None = None,
    ) -> BatchEnvelope[T]:
        if item_count <= 0 or item_count > self.max_items:
            raise ValueError("item_count must be in [1, max_items]")
        end = None if timeout is None else self._clock() + max(0.0, timeout)
        with self._condition:
            while self._queued_items + item_count > self.max_queued_items:
                if self._closed:
                    raise RuntimeError("batcher is closed")
                if timeout == 0:
                    raise TimeoutError("batcher queue is full")
                remaining = None if end is None else end - self._clock()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for batcher capacity")
                self._condition.wait(remaining)
            if self._closed:
                raise RuntimeError("batcher is closed")
            now = self._clock()
            envelope = BatchEnvelope(
                payload=payload,
                item_count=int(item_count),
                enqueued_at=now,
                deadline=now + self.max_wait_seconds,
            )
            self._queue.append(envelope)
            self._queued_items += int(item_count)
            self._condition.notify_all()
            return envelope

    def pop(self, *, timeout: float | None = None) -> list[BatchEnvelope[T]]:
        end = None if timeout is None else self._clock() + max(0.0, timeout)
        with self._condition:
            while not self._queue:
                if self._closed:
                    return []
                if timeout == 0:
                    return []
                remaining = None if end is None else end - self._clock()
                if remaining is not None and remaining <= 0:
                    return []
                self._condition.wait(remaining)

            while True:
                now = self._clock()
                available = self._batchable_item_count()
                oldest_expired = self._queue[0].deadline <= now
                if available >= self.min_items or oldest_expired or self._closed:
                    break
                waits = [max(0.0, self._queue[0].deadline - now)]
                if end is not None:
                    waits.append(max(0.0, end - now))
                wait_for = min(waits)
                if wait_for <= 0:
                    break
                self._condition.wait(wait_for)

            batch: list[BatchEnvelope[T]] = []
            count = 0
            while self._queue:
                candidate = self._queue[0]
                if count + candidate.item_count > self.max_items:
                    break
                batch.append(self._queue.popleft())
                count += candidate.item_count
                if count >= self.max_items:
                    break
            self._queued_items -= count
            self._condition.notify_all()
            return batch

    def _batchable_item_count(self) -> int:
        total = 0
        for envelope in self._queue:
            if total + envelope.item_count > self.max_items:
                break
            total += envelope.item_count
        return total
