from __future__ import annotations

import queue
import time
from typing import Any

from .in_process_inference import (
    InProcessBatchingInferenceRuntime,
    _InProcessRequest,
)


class RendezvousInProcessBatchingInferenceRuntime(
    InProcessBatchingInferenceRuntime
):
    """Arrival-aware batching for synchronous node-local Actors.

    Requests stay on the existing worker-side queue. The coalescing timer starts when the worker
    dequeues the first request, not at the producer's earlier timestamp. Every newly arrived Actor
    request renews a short quiet window, while a separate hard deadline bounds total waiting time.

    This avoids the sequence-wave fragmentation observed when ingress rendezvous timed out before
    all Python Actor threads registered. It also avoids unbounded tail latency when an Actor is
    delayed or absent. Once the distinct-Actor target is reached, already queued requests are
    drained greedily up to ``max_batch_items`` and inference begins immediately.
    """

    def __init__(
        self,
        *args: Any,
        target_request_count: int | None = None,
        hard_wait_multiplier: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        selected = self.actor_count if target_request_count is None else int(target_request_count)
        if selected <= 0 or selected > self.actor_count:
            raise ValueError("target_request_count must be in [1, actor_count]")
        if hard_wait_multiplier < 1.0:
            raise ValueError("hard_wait_multiplier must be at least 1")
        self.target_request_count = selected
        self.hard_wait_multiplier = float(hard_wait_multiplier)

    def _collect_batch(
        self,
        first: _InProcessRequest,
    ) -> list[_InProcessRequest]:
        batch = [first]
        items = first.item_count
        contributors = {first.actor_id}

        request_capacity = max(1, self.max_batch_items // max(1, first.item_count))
        target_requests = min(self.target_request_count, request_capacity)
        target_items = min(
            self.max_batch_items,
            max(self.min_batch_items, target_requests * first.item_count),
        )

        started = time.monotonic()
        quiet_deadline = started + self.max_wait_seconds
        hard_deadline = started + self.max_wait_seconds * self.hard_wait_multiplier

        while items < self.max_batch_items:
            target_reached = (
                items >= target_items and len(contributors) >= target_requests
            )

            candidate: _InProcessRequest | None
            if self._pending:
                candidate = self._pending.popleft()
            elif target_reached:
                # A complete Actor wave is ready. Include only requests that are already queued;
                # do not add more latency after the target is reached.
                try:
                    candidate = self._queue.get_nowait()
                except queue.Empty:
                    break
            else:
                remaining = min(quiet_deadline, hard_deadline) - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    candidate = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break

            if candidate is None:
                self._stop.set()
                break
            if items + candidate.item_count > self.max_batch_items:
                self._pending.appendleft(candidate)
                break

            batch.append(candidate)
            items += candidate.item_count
            contributors.add(candidate.actor_id)

            # Debounce arrival jitter while preserving a fixed oldest-request latency budget.
            quiet_deadline = min(
                hard_deadline,
                time.monotonic() + self.max_wait_seconds,
            )

        return batch
