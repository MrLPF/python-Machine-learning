from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
from multiprocessing.context import BaseContext
import struct
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

from forge_rl.transport import SharedTensorTreeArena, SharedTensorTreeDescriptor, TensorFieldSpec

from .inference import (
    DoubleBufferedPolicyReplica,
    InferenceFn,
    InferenceMetricsSnapshot,
    InferenceResponse,
    _InferenceMetrics,
)
from .optimized_inference import _ReusableBatchAssembler
from .policy_registry import PolicySnapshot

_REQUEST = struct.Struct("<qqqd")
_RESPONSE = struct.Struct("<qqdI")


@dataclass(frozen=True, slots=True)
class MailboxClientDescriptor:
    actor_id: int
    max_items: int
    request: SharedTensorTreeDescriptor
    response: SharedTensorTreeDescriptor
    connection: Connection


@dataclass(slots=True)
class MailboxInferenceEndpoint:
    """Persistent one-request mailbox for a synchronous Actor."""

    actor_id: int
    max_items: int
    request: SharedTensorTreeArena
    response: SharedTensorTreeArena
    service_connection: Connection
    client_connection: Connection

    @classmethod
    def create(
        cls,
        *,
        actor_id: int,
        max_items: int,
        request_fields: Mapping[str, TensorFieldSpec | tuple[tuple[int, ...], Any]],
        response_fields: Mapping[str, TensorFieldSpec | tuple[tuple[int, ...], Any]],
        mp_context: BaseContext | None = None,
    ) -> "MailboxInferenceEndpoint":
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        context = mp_context or mp.get_context()
        request = SharedTensorTreeArena.create(
            slot_count=1, max_items=max_items, fields=request_fields
        )
        try:
            response = SharedTensorTreeArena.create(
                slot_count=1, max_items=max_items, fields=response_fields
            )
        except BaseException:
            request.close()
            request.unlink()
            raise
        service_connection, client_connection = context.Pipe(duplex=True)
        return cls(
            int(actor_id),
            int(max_items),
            request,
            response,
            service_connection,
            client_connection,
        )

    def client_descriptor(self) -> MailboxClientDescriptor:
        return MailboxClientDescriptor(
            self.actor_id,
            self.max_items,
            self.request.descriptor,
            self.response.descriptor,
            self.client_connection,
        )

    def shutdown(self) -> None:
        """Compatibility no-op; the service owns its worker lifecycle."""

    def close(self) -> None:
        self.request.close()
        self.response.close()
        for connection in (self.service_connection, self.client_connection):
            try:
                connection.close()
            except OSError:
                pass

    def unlink(self) -> None:
        self.request.unlink()
        self.response.unlink()


class MailboxInferenceClient:
    """Actor-side synchronous client.

    With ``copy_outputs=False`` output arrays are valid until the next call on this client.
    """

    def __init__(
        self,
        endpoint: MailboxInferenceEndpoint | MailboxClientDescriptor,
        *,
        copy_outputs: bool = False,
    ) -> None:
        if isinstance(endpoint, MailboxInferenceEndpoint):
            self.actor_id = endpoint.actor_id
            self.max_items = endpoint.max_items
            self._request = endpoint.request
            self._response = endpoint.response
            self._connection = endpoint.client_connection
            self._owns_arenas = False
        else:
            self.actor_id = int(endpoint.actor_id)
            self.max_items = int(endpoint.max_items)
            self._request = SharedTensorTreeArena.attach(endpoint.request)
            self._response = SharedTensorTreeArena.attach(endpoint.response)
            self._connection = endpoint.connection
            self._owns_arenas = True
        self.copy_outputs = bool(copy_outputs)
        self._sequence = 0
        self._lock = threading.Lock()
        self._broken = False

    def infer(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        min_policy_version: int = -1,
        timeout: float = 5.0,
    ) -> InferenceResponse:
        if self._broken:
            raise RuntimeError("mailbox client is unusable after a failed exchange")
        if not inputs:
            raise ValueError("inputs must not be empty")
        sizes = {int(np.asarray(value).shape[0]) for value in inputs.values()}
        if len(sizes) != 1:
            raise ValueError("input tensors have inconsistent leading dimensions")
        item_count = sizes.pop()
        if item_count <= 0 or item_count > self.max_items:
            raise ValueError("item_count outside mailbox capacity")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        with self._lock:
            sequence_id = self._sequence
            self._sequence += 1
            submitted_at = time.monotonic()
            self._request.write(0, inputs, item_count=item_count)
            self._connection.send_bytes(
                _REQUEST.pack(
                    sequence_id,
                    item_count,
                    int(min_policy_version),
                    submitted_at,
                )
            )
            if not self._connection.poll(timeout):
                self._broken = True
                raise TimeoutError("timed out waiting for mailbox response")
            payload = self._connection.recv_bytes()
            if len(payload) < _RESPONSE.size:
                self._broken = True
                raise RuntimeError("truncated mailbox response")
            returned, policy_version, _completed, error_size = _RESPONSE.unpack_from(payload)
            error = payload[_RESPONSE.size :]
            if returned != sequence_id or len(error) != error_size:
                self._broken = True
                raise RuntimeError("mailbox response identity or length mismatch")
            if error:
                raise RuntimeError(error.decode("utf-8", errors="replace"))
            if policy_version < min_policy_version:
                self._broken = True
                raise RuntimeError("mailbox response used a stale policy")
            return InferenceResponse(
                outputs=self._response.read(
                    0, item_count=item_count, copy=self.copy_outputs
                ),
                policy_version=int(policy_version),
                sequence_id=sequence_id,
                latency_seconds=max(0.0, time.monotonic() - submitted_at),
            )

    def close(self) -> None:
        if self._owns_arenas:
            self._request.close()
            self._response.close()
        try:
            self._connection.close()
        except OSError:
            pass


@dataclass(frozen=True, slots=True)
class MailboxTransportMetricsSnapshot:
    collector_threads: int
    wait_calls: int
    empty_waits: int
    request_signals: int
    response_signals: int
    assembly_allocations: int
    assembly_copy_bytes: int
    reused_buffer_batches: int
    single_request_zero_copy_batches: int


@dataclass(frozen=True, slots=True)
class _MailboxRequest:
    endpoint: MailboxInferenceEndpoint
    sequence_id: int
    item_count: int
    min_policy_version: int
    submitted_at: float


class MailboxNodeLocalInferenceService:
    """One-thread collection, batching and inference for synchronous Actors."""

    def __init__(
        self,
        *,
        endpoints: Sequence[MailboxInferenceEndpoint],
        replica: DoubleBufferedPolicyReplica,
        infer_fn: InferenceFn,
        max_batch_items: int,
        min_batch_items: int = 1,
        max_wait_ms: float = 2.0,
        idle_wait_ms: float = 50.0,
    ) -> None:
        if not endpoints:
            raise ValueError("at least one mailbox endpoint is required")
        actor_ids = [endpoint.actor_id for endpoint in endpoints]
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError("mailbox actor IDs must be unique")
        if not 1 <= min_batch_items <= max_batch_items:
            raise ValueError("invalid mailbox batch limits")
        if max_wait_ms < 0 or idle_wait_ms <= 0:
            raise ValueError("invalid mailbox wait configuration")
        if any(endpoint.max_items > max_batch_items for endpoint in endpoints):
            raise ValueError("mailbox request capacity exceeds max_batch_items")
        self.endpoints = list(endpoints)
        self.replica = replica
        self.infer_fn = infer_fn
        self.max_batch_items = int(max_batch_items)
        self.min_batch_items = int(min_batch_items)
        self.max_wait_seconds = float(max_wait_ms) / 1000.0
        self.idle_wait_seconds = float(idle_wait_ms) / 1000.0
        self._connections = [endpoint.service_connection for endpoint in endpoints]
        self._endpoint_by_connection = {
            endpoint.service_connection: endpoint for endpoint in endpoints
        }
        self._pending: deque[_MailboxRequest] = deque()
        self._assembler = _ReusableBatchAssembler(max_items=max_batch_items)
        self._metrics = _InferenceMetrics()
        self._transport_lock = threading.Lock()
        self._wait_calls = 0
        self._empty_waits = 0
        self._request_signals = 0
        self._response_signals = 0
        self._assembly_allocations = 0
        self._assembly_copy_bytes = 0
        self._reused_buffer_batches = 0
        self._single_request_zero_copy_batches = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error: BaseException | None = None

    @property
    def last_error(self) -> BaseException | None:
        return self._last_error

    def refresh_policy(self, snapshot: PolicySnapshot) -> int:
        return self.replica.refresh(snapshot)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("mailbox service is already running")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="forge-mailbox-inference", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(max(0.0, timeout))
        if self._thread.is_alive():
            raise TimeoutError("mailbox service did not stop")
        self._thread = None

    def metrics(self) -> InferenceMetricsSnapshot:
        return self._metrics.snapshot(max_batch_items=self.max_batch_items)

    def transport_metrics(self) -> MailboxTransportMetricsSnapshot:
        with self._transport_lock:
            return MailboxTransportMetricsSnapshot(
                collector_threads=1,
                wait_calls=self._wait_calls,
                empty_waits=self._empty_waits,
                request_signals=self._request_signals,
                response_signals=self._response_signals,
                assembly_allocations=self._assembly_allocations,
                assembly_copy_bytes=self._assembly_copy_bytes,
                reused_buffer_batches=self._reused_buffer_batches,
                single_request_zero_copy_batches=self._single_request_zero_copy_batches,
            )

    def _receive_ready(self, timeout: float) -> list[_MailboxRequest]:
        ready = list(wait(self._connections, timeout=max(0.0, timeout)))
        with self._transport_lock:
            self._wait_calls += 1
            self._empty_waits += int(not ready)
        requests: list[_MailboxRequest] = []
        for connection in ready:
            payload = connection.recv_bytes()
            if len(payload) != _REQUEST.size:
                raise RuntimeError("invalid mailbox request header")
            sequence, items, min_version, submitted = _REQUEST.unpack(payload)
            endpoint = self._endpoint_by_connection[connection]
            if items <= 0 or items > endpoint.max_items:
                raise RuntimeError("mailbox item_count outside endpoint capacity")
            requests.append(
                _MailboxRequest(
                    endpoint,
                    int(sequence),
                    int(items),
                    int(min_version),
                    float(submitted),
                )
            )
        with self._transport_lock:
            self._request_signals += len(requests)
        return requests

    def _next_request(self) -> _MailboxRequest | None:
        if self._pending:
            return self._pending.popleft()
        ready = self._receive_ready(self.idle_wait_seconds)
        if not ready:
            return None
        self._pending.extend(ready[1:])
        return ready[0]

    def _collect_batch(self, first: _MailboxRequest) -> list[_MailboxRequest]:
        batch = [first]
        items = first.item_count
        deadline = first.submitted_at + self.max_wait_seconds
        while items < self.min_batch_items and items < self.max_batch_items:
            if self._pending:
                candidate = self._pending.popleft()
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                ready = self._receive_ready(remaining)
                if not ready:
                    break
                candidate = ready[0]
                self._pending.extend(ready[1:])
            if items + candidate.item_count > self.max_batch_items:
                self._pending.appendleft(candidate)
                break
            batch.append(candidate)
            items += candidate.item_count
        return batch

    def _send_error(
        self, requests: Sequence[_MailboxRequest], error: BaseException
    ) -> None:
        message = f"{type(error).__name__}: {error}".encode("utf-8", errors="replace")
        completed = time.monotonic()
        for request in requests:
            request.endpoint.service_connection.send_bytes(
                _RESPONSE.pack(
                    request.sequence_id,
                    self.replica.active_version,
                    completed,
                    len(message),
                )
                + message
            )
        with self._transport_lock:
            self._response_signals += len(requests)
        self._metrics.record_error()

    def _execute(self, requests: Sequence[_MailboxRequest]) -> None:
        views = [
            request.endpoint.request.read(
                0, item_count=request.item_count, copy=False
            )
            for request in requests
        ]
        merged, total_items, copied_bytes, zero_copy, allocated = self._assembler.assemble(
            views
        )
        requested_min = max(request.min_policy_version for request in requests)
        if self.replica.active_version < requested_min:
            raise RuntimeError("active policy is older than requested")
        outputs, policy_version = self.replica.infer(merged, self.infer_fn)
        completed = time.monotonic()
        offset = 0
        latencies: list[float] = []
        for request in requests:
            stop = offset + request.item_count
            request.endpoint.response.write(
                0,
                {name: value[offset:stop] for name, value in outputs.items()},
                item_count=request.item_count,
            )
            request.endpoint.service_connection.send_bytes(
                _RESPONSE.pack(request.sequence_id, policy_version, completed, 0)
            )
            latencies.append(max(0.0, completed - request.submitted_at))
            offset = stop
        if offset != total_items:
            raise RuntimeError("mailbox response split did not consume the batch")
        with self._transport_lock:
            self._response_signals += len(requests)
            self._assembly_allocations += int(allocated)
            self._assembly_copy_bytes += int(copied_bytes)
            self._reused_buffer_batches += int(not zero_copy and not allocated)
            self._single_request_zero_copy_batches += int(zero_copy)
        self._metrics.record_success(
            request_messages=len(requests),
            items=total_items,
            latencies=latencies,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._next_request()
                if first is None:
                    continue
                batch = self._collect_batch(first)
                try:
                    self._execute(batch)
                except BaseException as error:
                    self._send_error(batch, error)
            except (EOFError, OSError, ValueError) as error:
                if self._stop.is_set():
                    return
                self._last_error = error
                self._metrics.record_error()
                return
            except BaseException as error:
                self._last_error = error
                self._metrics.record_error()
                return
