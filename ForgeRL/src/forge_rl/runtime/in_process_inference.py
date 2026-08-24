from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import queue
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .inference import (
    DoubleBufferedPolicyReplica,
    InferenceFn,
    InferenceMetricsSnapshot,
    InferenceResponse,
    _InferenceMetrics,
)
from .optimized_inference import _ReusableBatchAssembler
from .policy_registry import PolicySnapshot


@dataclass(slots=True)
class _InProcessRequest:
    actor_id: int
    sequence_id: int
    inputs: dict[str, np.ndarray]
    item_count: int
    min_policy_version: int
    submitted_at: float
    completed: threading.Event = field(default_factory=threading.Event)
    outputs: dict[str, np.ndarray] | None = None
    policy_version: int = -1
    error: BaseException | None = None


class InProcessBatchingInferenceRuntime:
    """Zero-serialization batching path for threaded or C++ vectorized Actors.

    Actors submit NumPy views by reference to one local inference thread. The worker forms a
    deadline-bounded batch, executes one policy forward, and returns output views through per-call
    events. This path intentionally requires Actors and the predictor to share a process; separate
    Actor processes must use a shared-memory or network transport instead.
    """

    def __init__(
        self,
        *,
        actor_count: int,
        module_type: type[nn.Module],
        module_args: Sequence[Any] = (),
        module_kwargs: Mapping[str, Any] | None = None,
        infer_fn: InferenceFn,
        state_dict: Mapping[str, torch.Tensor],
        policy_version: int = 0,
        max_batch_items: int = 128,
        min_batch_items: int = 1,
        max_wait_ms: float = 0.5,
        device: torch.device | str = "cpu",
        amp_dtype: torch.dtype | None = None,
        copy_outputs: bool = False,
    ) -> None:
        if actor_count <= 0:
            raise ValueError("actor_count must be positive")
        if not 1 <= min_batch_items <= max_batch_items:
            raise ValueError("invalid in-process batch limits")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms cannot be negative")
        selected_device = torch.device(device)
        if selected_device.type not in {"cpu", "cuda"}:
            raise ValueError("in-process inference device must be cpu or cuda")
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA in-process inference requested but CUDA is unavailable")

        factory = lambda: module_type(*tuple(module_args), **dict(module_kwargs or {}))
        self.actor_count = int(actor_count)
        self.max_batch_items = int(max_batch_items)
        self.min_batch_items = int(min_batch_items)
        self.max_wait_seconds = float(max_wait_ms) / 1000.0
        self.copy_outputs = bool(copy_outputs)
        self.infer_fn = infer_fn
        self.replica = DoubleBufferedPolicyReplica(
            factory,
            device=selected_device,
            amp_dtype=amp_dtype,
        )
        self.replica.refresh(
            PolicySnapshot(
                version=int(policy_version),
                state_dict={
                    name: value.detach().cpu().clone(
                        memory_format=torch.preserve_format
                    )
                    for name, value in state_dict.items()
                },
                created_at=time.monotonic(),
            )
        )

        self._queue: queue.Queue[_InProcessRequest | None] = queue.Queue()
        self._pending: deque[_InProcessRequest] = deque()
        self._actor_locks = [threading.Lock() for _ in range(self.actor_count)]
        self._sequences = [0] * self.actor_count
        self._assembler = _ReusableBatchAssembler(max_items=self.max_batch_items)
        self._metrics = _InferenceMetrics()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._last_error: BaseException | None = None
        self._started = False
        self._closed = False

    @property
    def policy_version(self) -> int:
        return self.replica.active_version

    @property
    def last_error(self) -> BaseException | None:
        return self._last_error

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("in-process runtime is closed")
        if self._started:
            raise RuntimeError("in-process runtime is already started")
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._run,
            name="forge-in-process-inference",
            daemon=True,
        )
        self._worker.start()
        self._started = True

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

        with self._actor_locks[selected]:
            sequence_id = self._sequences[selected]
            self._sequences[selected] += 1
            request = _InProcessRequest(
                actor_id=selected,
                sequence_id=sequence_id,
                inputs=arrays,
                item_count=item_count,
                min_policy_version=int(min_policy_version),
                submitted_at=time.monotonic(),
            )
            self._queue.put(request)
            if not request.completed.wait(timeout):
                raise TimeoutError("timed out waiting for in-process inference")
            if request.error is not None:
                raise RuntimeError(
                    f"{type(request.error).__name__}: {request.error}"
                ) from request.error
            if request.outputs is None:
                raise RuntimeError("in-process inference returned no outputs")
            return InferenceResponse(
                outputs=request.outputs,
                policy_version=request.policy_version,
                sequence_id=sequence_id,
                latency_seconds=max(0.0, time.monotonic() - request.submitted_at),
            )

    def update_policy(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        version: int,
        timeout: float = 30.0,
    ) -> int:
        del timeout
        if not self._started or self._closed:
            raise RuntimeError("in-process runtime is not active")
        selected = int(version)
        if selected <= self.replica.active_version:
            raise ValueError("policy version must increase monotonically")
        snapshot = PolicySnapshot(
            version=selected,
            state_dict={
                name: value.detach().cpu().clone(
                    memory_format=torch.preserve_format
                )
                for name, value in state_dict.items()
            },
            created_at=time.monotonic(),
        )
        return self.replica.refresh(snapshot)

    def metrics(self) -> InferenceMetricsSnapshot:
        return self._metrics.snapshot(max_batch_items=self.max_batch_items)

    def close(self, *, timeout: float = 10.0) -> None:
        if self._closed:
            return
        self._stop.set()
        self._queue.put(None)
        worker = self._worker
        if worker is not None:
            worker.join(max(0.0, timeout))
            if worker.is_alive():
                raise TimeoutError("in-process inference worker did not stop")
        self._worker = None
        self._started = False
        self._closed = True

    def _next_request(self) -> _InProcessRequest | None:
        if self._pending:
            return self._pending.popleft()
        while True:
            try:
                request = self._queue.get(timeout=0.05)
            except queue.Empty:
                return None
            if request is None:
                self._stop.set()
                return None
            return request

    def _collect_batch(
        self,
        first: _InProcessRequest,
    ) -> list[_InProcessRequest]:
        batch = [first]
        items = first.item_count
        deadline = first.submitted_at + self.max_wait_seconds

        while items < self.max_batch_items:
            candidate: _InProcessRequest | None
            if self._pending:
                candidate = self._pending.popleft()
            else:
                remaining = deadline - time.monotonic()
                if items >= self.min_batch_items:
                    try:
                        candidate = self._queue.get_nowait()
                    except queue.Empty:
                        break
                elif remaining > 0:
                    try:
                        candidate = self._queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                else:
                    break

            if candidate is None:
                self._stop.set()
                break
            if items + candidate.item_count > self.max_batch_items:
                self._pending.appendleft(candidate)
                break
            batch.append(candidate)
            items += candidate.item_count

        return batch

    def _execute(self, requests: Sequence[_InProcessRequest]) -> None:
        views = [request.inputs for request in requests]
        merged, total_items, _copied, _zero_copy, _allocated = (
            self._assembler.assemble(views)
        )
        requested_min = max(request.min_policy_version for request in requests)
        if self.replica.active_version < requested_min:
            raise RuntimeError(
                f"active policy {self.replica.active_version} is older than "
                f"requested {requested_min}"
            )
        outputs, policy_version = self.replica.infer(merged, self.infer_fn)
        completed = time.monotonic()
        offset = 0
        latencies: list[float] = []
        for request in requests:
            stop = offset + request.item_count
            request.outputs = {
                name: (
                    value[offset:stop].copy()
                    if self.copy_outputs
                    else value[offset:stop]
                )
                for name, value in outputs.items()
            }
            request.policy_version = policy_version
            latencies.append(max(0.0, completed - request.submitted_at))
            offset = stop
        if offset != total_items:
            raise RuntimeError("in-process response split did not consume the batch")
        self._metrics.record_success(
            request_messages=len(requests),
            items=total_items,
            latencies=latencies,
        )
        for request in requests:
            request.completed.set()

    def _fail(
        self,
        requests: Sequence[_InProcessRequest],
        error: BaseException,
    ) -> None:
        self._metrics.record_error()
        for request in requests:
            request.error = error
            request.completed.set()

    def _run(self) -> None:
        while not self._stop.is_set() or self._pending or not self._queue.empty():
            first = self._next_request()
            if first is None:
                continue
            batch = self._collect_batch(first)
            try:
                self._execute(batch)
            except BaseException as error:
                self._last_error = error
                self._fail(batch, error)
