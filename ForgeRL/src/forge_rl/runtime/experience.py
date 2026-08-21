from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ExperienceItem:
    payload: Any
    train_rows: int
    policy_version_min: int
    policy_version_max: int
    actor_id: int
    created_at: float

    @classmethod
    def create(
        cls,
        payload: Any,
        *,
        train_rows: int,
        policy_version_min: int,
        policy_version_max: int,
        actor_id: int,
    ) -> "ExperienceItem":
        if train_rows <= 0:
            raise ValueError("train_rows must be positive")
        if policy_version_min < 0 or policy_version_max < policy_version_min:
            raise ValueError("invalid policy version range")
        return cls(
            payload=payload,
            train_rows=int(train_rows),
            policy_version_min=int(policy_version_min),
            policy_version_max=int(policy_version_max),
            actor_id=int(actor_id),
            created_at=time.monotonic(),
        )


class OnPolicyExperienceQueue:
    """Row-bounded on-policy queue with explicit backpressure and stale filtering."""

    def __init__(self, *, capacity_rows: int, mode: Literal["fifo", "lifo"] = "fifo") -> None:
        if capacity_rows <= 0:
            raise ValueError("capacity_rows must be positive")
        if mode not in {"fifo", "lifo"}:
            raise ValueError("mode must be fifo or lifo")
        self.capacity_rows = int(capacity_rows)
        self.mode = mode
        self._items: deque[ExperienceItem] = deque()
        self._rows = 0
        self._closed = False
        self._condition = threading.Condition()
        self.dropped_stale = 0
        self.dropped_oversized = 0

    @property
    def rows(self) -> int:
        with self._condition:
            return self._rows

    @property
    def pieces(self) -> int:
        with self._condition:
            return len(self._items)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def put(self, item: ExperienceItem, *, timeout: float | None = None) -> None:
        if item.train_rows > self.capacity_rows:
            self.dropped_oversized += 1
            raise ValueError("experience item is larger than queue capacity")
        end = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._rows + item.train_rows > self.capacity_rows:
                if self._closed:
                    raise RuntimeError("experience queue is closed")
                if timeout == 0:
                    raise TimeoutError("experience queue is full")
                remaining = None if end is None else end - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for experience capacity")
                self._condition.wait(remaining)
            if self._closed:
                raise RuntimeError("experience queue is closed")
            self._items.append(item)
            self._rows += item.train_rows
            self._condition.notify_all()

    def sample(
        self,
        *,
        min_rows: int,
        current_policy_version: int,
        max_policy_lag: int,
        max_age_seconds: float | None = None,
        timeout: float | None = None,
    ) -> list[ExperienceItem]:
        if min_rows <= 0:
            raise ValueError("min_rows must be positive")
        end = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                self._drop_stale_locked(
                    current_policy_version=current_policy_version,
                    max_policy_lag=max_policy_lag,
                    max_age_seconds=max_age_seconds,
                )
                if self._rows >= min_rows or (self._closed and self._items):
                    break
                if self._closed:
                    return []
                if timeout == 0:
                    return []
                remaining = None if end is None else end - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return []
                self._condition.wait(remaining)

            selected: list[ExperienceItem] = []
            rows = 0
            while self._items and rows < min_rows:
                item = self._items.popleft() if self.mode == "fifo" else self._items.pop()
                selected.append(item)
                rows += item.train_rows
                self._rows -= item.train_rows
            self._condition.notify_all()
            return selected

    def _drop_stale_locked(
        self,
        *,
        current_policy_version: int,
        max_policy_lag: int,
        max_age_seconds: float | None,
    ) -> None:
        now = time.monotonic()
        kept: deque[ExperienceItem] = deque()
        while self._items:
            item = self._items.popleft()
            lag = int(current_policy_version) - item.policy_version_max
            age = now - item.created_at
            stale = lag > max_policy_lag or (
                max_age_seconds is not None and age > max_age_seconds
            )
            if stale:
                self._rows -= item.train_rows
                self.dropped_stale += 1
            else:
                kept.append(item)
        self._items = kept
