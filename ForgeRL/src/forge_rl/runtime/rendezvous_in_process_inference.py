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

    The base runtime starts a deadline from the Actor submission timestamp and may dispatch as
    soon as ``min_batch_items`` is reached. Under Python thread scheduling this can make the first
    request's deadline expire before peer Actors enter the queue, producing one-Actor batches.

    This variant starts the coalescing window when the worker dequeues the first request and waits
    for a bounded wave of distinct Actor contributors. It still has a hard timeout, so a missing or
    stalled Actor cannot deadlock inference. Once the wave target is reached, requests already in
    the queue are drained greedily up to ``max_batch_items``.
    """

    def __init__(
        self,
        *args: Any,
        target_request_count: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        selected = self.actor_count if target_request_count is None else int(target_request_count)
        if selected <= 0 or selected > self.actor_count:
            raise ValueError("target_request_count must be in [1, actor_count]")
        self.target_request_count = selected

    def _collect_batch(
        self,
        first: _InProcessRequest,
    ) -> list[_InProcessRequest]:
        batch = [first]
        items = first.item_count
        contributors = {first.actor_id}

        # One synchronous Actor can have only one outstanding request. The useful target is
        # therefore a wave of distinct Actors, capped by the tensor batch capacity.
        request_capacity = max(1, self.max_batch_items // max(1, first.item_count))
        target_requests = min(self.target_request_count, request_capacity)
        target_items = min(
            self.max_batch_items,
            max(self.min_batch_items, target_requests * first.item_count),
        )

        # Match established dynamic-batching schedulers: the batching window begins when the
        # scheduler receives/dequeues the first item, not at the producer's earlier timestamp.
        deadline = time.monotonic() + self.max_wait_seconds

        while items < self.max_batch_items:
            target_reached = (
                items >= target_items and len(contributors) >= target_requests
            )

            candidate: _InProcessRequest | None
            if self._pending:
                candidate = self._pending.popleft()
            elif target_reached:
                # The target wave is complete. Include anything already queued, but do not add
                # another wait to the critical path.
                try:
                    candidate = self._queue.get_nowait()
                except queue.Empty:
                    break
            else:
                remaining = deadline - time.monotonic()
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

        return batch
