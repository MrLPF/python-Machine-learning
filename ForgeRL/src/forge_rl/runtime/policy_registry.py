from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Mapping

import torch


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    version: int
    state_dict: dict[str, torch.Tensor]
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


class PolicyRegistry:
    """Thread-safe monotonic policy registry with bounded exact-version history."""

    def __init__(self, *, history_size: int = 2) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self.history_size = int(history_size)
        self._snapshots: OrderedDict[int, PolicySnapshot] = OrderedDict()
        self._next_version = 0
        self._condition = threading.Condition()

    @property
    def latest_version(self) -> int:
        with self._condition:
            return next(reversed(self._snapshots), -1)

    def publish(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        version: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PolicySnapshot:
        with self._condition:
            selected = self._next_version if version is None else int(version)
            if selected < self._next_version:
                raise ValueError(
                    f"policy version {selected} is older than next version {self._next_version}"
                )
            if selected in self._snapshots:
                raise ValueError(f"policy version {selected} already exists")
            cloned = {
                name: value.detach().cpu().clone(memory_format=torch.preserve_format)
                for name, value in state_dict.items()
            }
            snapshot = PolicySnapshot(
                version=selected,
                state_dict=cloned,
                created_at=time.monotonic(),
                metadata=dict(metadata or {}),
            )
            self._snapshots[selected] = snapshot
            self._next_version = selected + 1
            while len(self._snapshots) > self.history_size:
                self._snapshots.popitem(last=False)
            self._condition.notify_all()
            return snapshot

    def get(self, version: int = -1) -> PolicySnapshot:
        with self._condition:
            if not self._snapshots:
                raise LookupError("no policy has been published")
            selected = self.latest_version if version < 0 else int(version)
            try:
                return self._snapshots[selected]
            except KeyError as exc:
                available = list(self._snapshots)
                raise LookupError(
                    f"policy version {selected} unavailable; retained versions={available}"
                ) from exc

    def wait_for_newer(self, version: int, *, timeout: float | None = None) -> PolicySnapshot:
        end = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self.latest_version <= version:
                remaining = None if end is None else end - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"no policy newer than {version}")
                self._condition.wait(remaining)
            return self.get(-1)

    def lag(self, actor_seen_version: int) -> int:
        latest = self.latest_version
        return 0 if latest < 0 else max(0, latest - int(actor_seen_version))
