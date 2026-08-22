from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading
import time
from typing import Any, Mapping

import numpy as np

from .in_process_inference import (
    InProcessBatchingInferenceRuntime,
    _InProcessRequest,
)
from .inference import InferenceResponse


@dataclass(slots=True)
class _IngressWave:
    deadline: float
    requests: dict[int, _InProcessRequest] = field(default_factory=dict)
    items: int = 0


@dataclass(frozen=True, slots=True)
class _WaveEnvelope:
    requests: tuple[_InProcessRequest, ...]


class RendezvousInProcessBatchingInferenceRuntime(
    InProcessBatchingInferenceRuntime
):
    """Sequence-aware ingress rendezvous for synchronous node-local Actors.

    The legacy in-process runtime exposed each request to the worker immediately. The worker then
    competed with producer threads for scheduling time while trying to collect a batch. At short
    deadlines this frequently made a partial Actor wave visible and produced one-Actor batches.

    This runtime first groups same-sequence requests at ingress. A complete Actor wave is placed on
    the worker queue as one envelope, making every member visible to the worker at once. A bounded
    timeout releases a partial wave when an Actor is slow, absent or failed, so rendezvous cannot
    deadlock inference. Tensor payloads remain ordinary in-process NumPy references.
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
        self._wave_condition = threading.Condition()
        self._waves: dict[tuple[int, int], _IngressWave] = {}

    def _flush_wave_locked(self, key: tuple[int, int]) -> None:
        wave = self._waves.pop(key, None)
        if wave is None:
            return
        ordered = tuple(wave.requests[actor] for actor in sorted(wave.requests))
        if ordered:
            self._queue.put(_WaveEnvelope(ordered))
        self._wave_condition.notify_all()

    def _submit_wave(self, request: _InProcessRequest) -> None:
        key = (request.sequence_id, request.min_policy_version)
        with self._wave_condition:
            wave = self._waves.get(key)
            if wave is None:
                wave = _IngressWave(
                    deadline=time.monotonic() + self.max_wait_seconds,
                )
                self._waves[key] = wave
            if request.actor_id in wave.requests:
                raise RuntimeError(
                    f"actor {request.actor_id} submitted sequence {request.sequence_id} twice"
                )
            wave.requests[request.actor_id] = request
            wave.items += request.item_count

            capacity = max(1, self.max_batch_items // max(1, request.item_count))
            target_requests = min(self.target_request_count, capacity)
            if (
                len(wave.requests) >= target_requests
                or wave.items >= self.max_batch_items
            ):
                self._flush_wave_locked(key)
                return

            while key in self._waves:
                remaining = wave.deadline - time.monotonic()
                if remaining <= 0:
                    self._flush_wave_locked(key)
                    return
                self._wave_condition.wait(remaining)

    def _collect_batch(
        self,
        first: _InProcessRequest,
    ) -> list[_InProcessRequest]:
        """Consume exactly one ingress envelope.

        The base implementation may read directly from the queue after draining ``_pending``.
        That would expose the next sequence's envelope as if it were an individual request. The
        rendezvous worker instead processes one atomic Actor wave at a time.
        """

        batch = [first]
        items = first.item_count
        while self._pending:
            candidate = self._pending.popleft()
            if items + candidate.item_count > self.max_batch_items:
                self._pending.appendleft(candidate)
                break
            batch.append(candidate)
            items += candidate.item_count
        return batch

    def infer(
        self,
        actor_id: int,
        inputs: Mapping[str, np.ndarray],
        *,
        min_policy_version: int = -1,
        timeout: float = 15.0,
    ) -> InferenceResponse:
        if not self._started or self._closed:
            raise RuntimeError("in-process runtime is not active")
        selected = int(actor_id)
        if selected < 0 or selected >= self.actor_count:
            raise IndexError("actor_id outside in-process runtime")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if not inputs:
            raise ValueError("inputs must not be empty")

        arrays: dict[str, np.ndarray] = {}
        item_count = -1
        for name, value in inputs.items():
            array = np.ascontiguousarray(value)
            if array.ndim == 0:
                raise ValueError(f"input field {name!r} has no leading dimension")
            if item_count < 0:
                item_count = int(array.shape[0])
            elif int(array.shape[0]) != item_count:
                raise ValueError("input tensors have inconsistent leading dimensions")
            arrays[name] = array
        if item_count <= 0 or item_count > self.max_batch_items:
            raise ValueError("item_count outside in-process capacity")

        started = time.monotonic()
        with self._actor_locks[selected]:
            sequence_id = self._sequences[selected]
            self._sequences[selected] += 1
            request = _InProcessRequest(
                actor_id=selected,
                sequence_id=sequence_id,
                inputs=arrays,
                item_count=item_count,
                min_policy_version=int(min_policy_version),
                submitted_at=started,
            )
            self._submit_wave(request)
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0 or not request.completed.wait(remaining):
                raise TimeoutError("timed out waiting for rendezvous inference")
            if request.error is not None:
                raise RuntimeError(
                    f"{type(request.error).__name__}: {request.error}"
                ) from request.error
            if request.outputs is None:
                raise RuntimeError("rendezvous inference returned no outputs")
            return InferenceResponse(
                outputs=request.outputs,
                policy_version=request.policy_version,
                sequence_id=sequence_id,
                latency_seconds=max(0.0, time.monotonic() - started),
            )

    def _next_request(self) -> _InProcessRequest | None:
        if self._pending:
            return self._pending.popleft()
        while True:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                return None
            if item is None:
                self._stop.set()
                return None
            if isinstance(item, _WaveEnvelope):
                if not item.requests:
                    continue
                self._pending.extend(item.requests[1:])
                return item.requests[0]
            if isinstance(item, _InProcessRequest):
                return item
            raise RuntimeError(f"unsupported rendezvous queue item: {type(item)!r}")
