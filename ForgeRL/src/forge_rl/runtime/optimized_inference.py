from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Mapping, Sequence

import numpy as np

from .inference import (
    DoubleBufferedPolicyReplica,
    InferenceFn,
    NodeLocalInferenceService as ThreadedNodeLocalInferenceService,
    SharedInferenceEndpoint,
    _PendingRequest,
)


@dataclass(frozen=True, slots=True)
class OptimizedInferenceMetricsSnapshot:
    """Counters for the M1 local inference hot-path optimizations."""

    collector_threads: int
    collector_polls: int
    empty_collector_polls: int
    assembly_allocations: int
    assembly_copy_bytes: int
    single_request_zero_copy_batches: int
    reused_buffer_batches: int


class _OptimizationMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._collector_polls = 0
        self._empty_collector_polls = 0
        self._assembly_allocations = 0
        self._assembly_copy_bytes = 0
        self._single_request_zero_copy_batches = 0
        self._reused_buffer_batches = 0

    def record_poll(self, *, progressed: bool) -> None:
        with self._lock:
            self._collector_polls += 1
            self._empty_collector_polls += int(not progressed)

    def record_assembly(
        self,
        *,
        copied_bytes: int,
        zero_copy: bool,
        allocated: bool,
    ) -> None:
        with self._lock:
            self._assembly_copy_bytes += int(copied_bytes)
            self._assembly_allocations += int(allocated)
            self._single_request_zero_copy_batches += int(zero_copy)
            self._reused_buffer_batches += int(not zero_copy and not allocated)

    def snapshot(self) -> OptimizedInferenceMetricsSnapshot:
        with self._lock:
            return OptimizedInferenceMetricsSnapshot(
                collector_threads=1,
                collector_polls=self._collector_polls,
                empty_collector_polls=self._empty_collector_polls,
                assembly_allocations=self._assembly_allocations,
                assembly_copy_bytes=self._assembly_copy_bytes,
                single_request_zero_copy_batches=self._single_request_zero_copy_batches,
                reused_buffer_batches=self._reused_buffer_batches,
            )


class _ReusableBatchAssembler:
    """Assemble shared-memory views into reusable contiguous buffers.

    A one-request batch remains a direct view into the request slot. Multi-request batches copy
    into preallocated buffers whose storage is reused until the tensor schema changes.
    """

    def __init__(self, *, max_items: int) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self.max_items = int(max_items)
        self._signature: tuple[tuple[str, str, tuple[int, ...]], ...] | None = None
        self._buffers: dict[str, np.ndarray] = {}

    def assemble(
        self,
        views: Sequence[Mapping[str, np.ndarray]],
    ) -> tuple[dict[str, np.ndarray], int, int, bool, bool]:
        if not views:
            raise ValueError("at least one request view is required")

        names = tuple(sorted(views[0]))
        if not names:
            raise ValueError("request tensor tree must not be empty")

        expected: dict[str, tuple[np.dtype, tuple[int, ...]]] = {}
        counts: list[int] = []
        for index, view in enumerate(views):
            if tuple(sorted(view)) != names:
                raise ValueError("request endpoint tensor schemas differ")
            arrays = {name: np.asarray(view[name]) for name in names}
            count = int(arrays[names[0]].shape[0])
            if count <= 0:
                raise ValueError("request batches must contain at least one item")
            for name, array in arrays.items():
                if array.ndim == 0 or int(array.shape[0]) != count:
                    raise ValueError(
                        f"request field {name!r} has an inconsistent leading dimension"
                    )
                schema = (array.dtype, tuple(int(value) for value in array.shape[1:]))
                if index == 0:
                    expected[name] = schema
                elif schema != expected[name]:
                    raise ValueError(f"request field {name!r} schema differs across endpoints")
            counts.append(count)

        total_items = sum(counts)
        if total_items > self.max_items:
            raise ValueError(
                f"assembled batch size {total_items} exceeds capacity {self.max_items}"
            )

        if len(views) == 1:
            return (
                {name: np.asarray(views[0][name]) for name in names},
                total_items,
                0,
                True,
                False,
            )

        signature = tuple(
            (name, expected[name][0].str, expected[name][1])
            for name in names
        )
        allocated = signature != self._signature
        if allocated:
            self._buffers = {
                name: np.empty(
                    (self.max_items, *expected[name][1]),
                    dtype=expected[name][0],
                )
                for name in names
            }
            self._signature = signature

        offset = 0
        copied_bytes = 0
        for view, count in zip(views, counts, strict=True):
            stop = offset + count
            for name in names:
                source = np.asarray(view[name])
                np.copyto(self._buffers[name][offset:stop], source, casting="no")
                copied_bytes += int(source.nbytes)
            offset = stop

        return (
            {name: self._buffers[name][:total_items] for name in names},
            total_items,
            copied_bytes,
            False,
            allocated,
        )


class NodeLocalInferenceService(ThreadedNodeLocalInferenceService):
    """Optimized M1 node-local inference service.

    The service replaces one receiver thread per actor with a single fair round-robin collector.
    It also reuses contiguous input assembly buffers and keeps one-request batches zero-copy.
    The inherited threaded implementation remains available as
    :class:`ThreadedNodeLocalInferenceService` for regression comparison.
    """

    def __init__(
        self,
        *,
        endpoints: Sequence[SharedInferenceEndpoint],
        replica: DoubleBufferedPolicyReplica,
        infer_fn: InferenceFn,
        max_batch_items: int,
        min_batch_items: int = 1,
        max_wait_ms: float = 2.0,
        response_timeout_seconds: float = 5.0,
        collector_poll_ms: float = 0.25,
        max_drain_per_endpoint: int = 4,
    ) -> None:
        super().__init__(
            endpoints=endpoints,
            replica=replica,
            infer_fn=infer_fn,
            max_batch_items=max_batch_items,
            min_batch_items=min_batch_items,
            max_wait_ms=max_wait_ms,
            response_timeout_seconds=response_timeout_seconds,
        )
        if collector_poll_ms < 0:
            raise ValueError("collector_poll_ms cannot be negative")
        if max_drain_per_endpoint <= 0:
            raise ValueError("max_drain_per_endpoint must be positive")
        self.collector_poll_seconds = float(collector_poll_ms) / 1000.0
        self.max_drain_per_endpoint = int(max_drain_per_endpoint)
        self._assembler = _ReusableBatchAssembler(max_items=max_batch_items)
        self._optimization_metrics = _OptimizationMetrics()

    def start(self) -> None:
        if self._threads:
            raise RuntimeError("inference service is already running")
        self._stop.clear()
        collector = threading.Thread(
            target=self._collector_loop,
            name="forge-inference-collector",
            daemon=True,
        )
        collector.start()
        self._threads.append(collector)
        worker = threading.Thread(
            target=self._worker_loop,
            name="forge-inference-worker",
            daemon=True,
        )
        worker.start()
        self._threads.append(worker)

    def optimization_metrics(self) -> OptimizedInferenceMetricsSnapshot:
        return self._optimization_metrics.snapshot()

    def _collector_loop(self) -> None:
        endpoint_count = len(self.endpoints)
        cursor = 0
        while not self._stop.is_set():
            progressed = False
            for offset in range(endpoint_count):
                endpoint = self.endpoints[(cursor + offset) % endpoint_count]
                drained = 0
                while drained < self.max_drain_per_endpoint:
                    try:
                        message = endpoint.request.receive(timeout=0.0)
                    except TimeoutError:
                        break
                    except RuntimeError as exc:
                        if self._stop.is_set():
                            return
                        self._last_error = exc
                        self._metrics.record_error()
                        return

                    pending = _PendingRequest(endpoint, message, time.monotonic())
                    try:
                        self._batcher.submit(
                            pending,
                            item_count=message.item_count,
                            timeout=0.5,
                        )
                    except BaseException as exc:
                        try:
                            endpoint.request.release(message)
                        except RuntimeError:
                            pass
                        if self._stop.is_set():
                            return
                        self._last_error = exc
                        self._metrics.record_error()
                        return
                    progressed = True
                    drained += 1

            cursor = (cursor + 1) % endpoint_count
            self._optimization_metrics.record_poll(progressed=progressed)
            if not progressed:
                self._stop.wait(self.collector_poll_seconds)

    def _execute_batch(self, pending: Sequence[_PendingRequest]) -> None:
        views = [
            request.endpoint.request.read(request.message, copy=False)
            for request in pending
        ]
        merged, total_items, copied_bytes, zero_copy, allocated = self._assembler.assemble(
            views
        )
        requested_min = max(
            int(request.message.metadata.get("min_policy_version", -1))
            for request in pending
        )
        if self.replica.active_version < requested_min:
            raise RuntimeError(
                f"active policy {self.replica.active_version} is older than requested "
                f"{requested_min}"
            )

        outputs, policy_version = self.replica.infer(merged, self.infer_fn)
        offset = 0
        completed_at = time.monotonic()
        latencies: list[float] = []
        for request in pending:
            count = request.message.item_count
            stop = offset + count
            split = {name: value[offset:stop] for name, value in outputs.items()}
            submitted_at = float(
                request.message.metadata.get("submitted_at", request.received_at)
            )
            request.endpoint.response.send(
                split,
                item_count=count,
                sequence_id=request.message.sequence_id,
                metadata={
                    "actor_id": request.endpoint.actor_id,
                    "policy_version": policy_version,
                    "submitted_at": submitted_at,
                },
                timeout=self.response_timeout_seconds,
            )
            request.endpoint.request.release(request.message)
            latencies.append(max(0.0, completed_at - submitted_at))
            offset = stop

        if offset != total_items:
            raise RuntimeError(
                f"response split consumed {offset} items but batch contained {total_items}"
            )
        self._metrics.record_success(
            request_messages=len(pending),
            items=total_items,
            latencies=latencies,
        )
        self._optimization_metrics.record_assembly(
            copied_bytes=copied_bytes,
            zero_copy=zero_copy,
            allocated=allocated,
        )
