from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from forge_rl.transport import SharedTensorChannel, SlotMessage, TensorFieldSpec

from .dynamic_batcher import DeadlineBatcher
from .policy_registry import PolicySnapshot

InferenceFn = Callable[[nn.Module, Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]]


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    outputs: dict[str, np.ndarray]
    policy_version: int
    sequence_id: int
    latency_seconds: float


@dataclass(slots=True)
class SharedInferenceEndpoint:
    actor_id: int
    request: SharedTensorChannel
    response: SharedTensorChannel

    @classmethod
    def create(
        cls,
        *,
        actor_id: int,
        slot_count: int,
        max_items: int,
        request_fields: Mapping[str, TensorFieldSpec | tuple[tuple[int, ...], Any]],
        response_fields: Mapping[str, TensorFieldSpec | tuple[tuple[int, ...], Any]],
        mp_context: Any | None = None,
    ) -> "SharedInferenceEndpoint":
        return cls(
            actor_id=int(actor_id),
            request=SharedTensorChannel.create(
                slot_count=slot_count,
                max_items=max_items,
                fields=request_fields,
                mp_context=mp_context,
            ),
            response=SharedTensorChannel.create(
                slot_count=slot_count,
                max_items=max_items,
                fields=response_fields,
                mp_context=mp_context,
            ),
        )

    def close(self) -> None:
        self.request.close()
        self.response.close()

    def shutdown(self) -> None:
        self.request.shutdown()
        self.response.shutdown()

    def unlink(self) -> None:
        self.request.unlink()
        self.response.unlink()


class DoubleBufferedPolicyReplica:
    """Atomic active/staging model pair for node-local inference.

    Weights are loaded into the inactive model, then the two model references are swapped while
    inference is excluded. This avoids exposing a partially loaded state dict.
    """

    def __init__(
        self,
        module_factory: Callable[[], nn.Module],
        *,
        device: torch.device | str | None = None,
        amp_dtype: torch.dtype | None = None,
        pin_memory: bool = True,
    ) -> None:
        selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference requested but CUDA is unavailable")
        self.device = selected
        self.amp_dtype = amp_dtype
        self.pin_memory = bool(pin_memory)
        self._active = module_factory().to(self.device).eval()
        self._staging = module_factory().to(self.device).eval()
        self._active_version = -1
        self._staged_version = -1
        self._active_lock = threading.RLock()
        self._stage_lock = threading.Lock()

    @property
    def active_version(self) -> int:
        with self._active_lock:
            return self._active_version

    def stage(self, snapshot: PolicySnapshot) -> None:
        with self._stage_lock:
            self._staging.load_state_dict(snapshot.state_dict, strict=True)
            self._staging.eval()
            self._staged_version = int(snapshot.version)

    def activate_staged(self) -> int:
        with self._active_lock:
            if self._staged_version < 0:
                raise RuntimeError("no staged policy is ready")
            self._active, self._staging = self._staging, self._active
            self._active_version, self._staged_version = self._staged_version, -1
            return self._active_version

    def refresh(self, snapshot: PolicySnapshot) -> int:
        self.stage(snapshot)
        return self.activate_staged()

    def _to_device(self, inputs: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
        prepared: dict[str, torch.Tensor] = {}
        for name, value in inputs.items():
            host = torch.from_numpy(np.ascontiguousarray(value))
            if self.device.type == "cuda":
                if self.pin_memory and not host.is_pinned():
                    host = host.pin_memory()
                prepared[name] = host.to(self.device, non_blocking=True)
            else:
                prepared[name] = host.to(self.device)
        return prepared

    def infer(
        self,
        inputs: Mapping[str, np.ndarray],
        infer_fn: InferenceFn,
    ) -> tuple[dict[str, np.ndarray], int]:
        tensors = self._to_device(inputs)
        with self._active_lock, torch.inference_mode():
            autocast_enabled = self.device.type == "cuda" and self.amp_dtype is not None
            autocast_context = (
                torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
                if autocast_enabled
                else nullcontext()
            )
            with autocast_context:
                raw_outputs = dict(infer_fn(self._active, tensors))
            version = self._active_version
        if not raw_outputs:
            raise ValueError("inference function returned no outputs")
        expected_items = next(iter(tensors.values())).shape[0]
        outputs: dict[str, np.ndarray] = {}
        for name, value in raw_outputs.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"inference output {name!r} is not a tensor")
            if value.ndim == 0 or value.shape[0] != expected_items:
                raise ValueError(
                    f"inference output {name!r} leading dimension must be {expected_items}"
                )
            cpu = value.detach().to("cpu", non_blocking=self.device.type == "cuda")
            outputs[name] = cpu.numpy()
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        return outputs, version


@dataclass(frozen=True, slots=True)
class InferenceMetricsSnapshot:
    batches: int
    request_messages: int
    items: int
    errors: int
    mean_batch_items: float
    batch_fill_ratio: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float


class _InferenceMetrics:
    def __init__(self, *, max_samples: int = 4096) -> None:
        self._lock = threading.Lock()
        self._batches = 0
        self._request_messages = 0
        self._items = 0
        self._errors = 0
        self._batch_items = 0
        self._latencies: deque[float] = deque(maxlen=max_samples)

    def record_success(
        self,
        *,
        request_messages: int,
        items: int,
        latencies: Sequence[float],
    ) -> None:
        with self._lock:
            self._batches += 1
            self._request_messages += int(request_messages)
            self._items += int(items)
            self._batch_items += int(items)
            self._latencies.extend(float(value) for value in latencies)

    def record_error(self) -> None:
        with self._lock:
            self._errors += 1

    def snapshot(self, *, max_batch_items: int) -> InferenceMetricsSnapshot:
        with self._lock:
            latencies = np.asarray(self._latencies, dtype=np.float64)
            percentiles = (
                np.percentile(latencies, [50, 95, 99]).tolist()
                if latencies.size
                else [0.0, 0.0, 0.0]
            )
            mean_batch = self._batch_items / self._batches if self._batches else 0.0
            return InferenceMetricsSnapshot(
                batches=self._batches,
                request_messages=self._request_messages,
                items=self._items,
                errors=self._errors,
                mean_batch_items=mean_batch,
                batch_fill_ratio=(mean_batch / max_batch_items if max_batch_items else 0.0),
                latency_p50_ms=float(percentiles[0] * 1000.0),
                latency_p95_ms=float(percentiles[1] * 1000.0),
                latency_p99_ms=float(percentiles[2] * 1000.0),
            )


@dataclass(frozen=True, slots=True)
class _PendingRequest:
    endpoint: SharedInferenceEndpoint
    message: SlotMessage
    received_at: float


class NodeLocalInferenceService:
    """Node-local shared-memory inference service with cross-actor dynamic batching."""

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
    ) -> None:
        if not endpoints:
            raise ValueError("at least one inference endpoint is required")
        actor_ids = [endpoint.actor_id for endpoint in endpoints]
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError("inference endpoint actor IDs must be unique")
        if any(endpoint.request.max_items > max_batch_items for endpoint in endpoints):
            raise ValueError("an endpoint request capacity exceeds the service batch capacity")
        self.endpoints = list(endpoints)
        self.replica = replica
        self.infer_fn = infer_fn
        self.max_batch_items = int(max_batch_items)
        self.response_timeout_seconds = float(response_timeout_seconds)
        self._batcher: DeadlineBatcher[_PendingRequest] = DeadlineBatcher(
            max_items=max_batch_items,
            min_items=min_batch_items,
            max_wait_ms=max_wait_ms,
        )
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._metrics = _InferenceMetrics()
        self._last_error: BaseException | None = None

    @property
    def last_error(self) -> BaseException | None:
        return self._last_error

    def refresh_policy(self, snapshot: PolicySnapshot) -> int:
        return self.replica.refresh(snapshot)

    def start(self) -> None:
        if self._threads:
            raise RuntimeError("inference service is already running")
        self._stop.clear()
        for endpoint in self.endpoints:
            thread = threading.Thread(
                target=self._receive_loop,
                args=(endpoint,),
                name=f"forge-inference-rx-{endpoint.actor_id}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        worker = threading.Thread(
            target=self._worker_loop,
            name="forge-inference-worker",
            daemon=True,
        )
        worker.start()
        self._threads.append(worker)

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        self._batcher.close()
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in self._threads:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(remaining)
        alive = [thread.name for thread in self._threads if thread.is_alive()]
        self._threads.clear()
        if alive:
            raise TimeoutError(f"inference service threads did not stop: {alive}")

    def metrics(self) -> InferenceMetricsSnapshot:
        return self._metrics.snapshot(max_batch_items=self.max_batch_items)

    def _receive_loop(self, endpoint: SharedInferenceEndpoint) -> None:
        while not self._stop.is_set():
            try:
                message = endpoint.request.receive(timeout=0.05)
            except TimeoutError:
                continue
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
                self._last_error = exc
                self._metrics.record_error()
                try:
                    endpoint.request.release(message)
                except RuntimeError:
                    pass
                if self._stop.is_set():
                    return

    def _worker_loop(self) -> None:
        while not self._stop.is_set() or self._batcher.queued_items:
            batch = self._batcher.pop(timeout=0.05)
            if not batch:
                continue
            pending = [envelope.payload for envelope in batch]
            try:
                self._execute_batch(pending)
            except BaseException as exc:
                self._last_error = exc
                self._metrics.record_error()
                for request in pending:
                    self._publish_error(request, exc)

    def _execute_batch(self, pending: Sequence[_PendingRequest]) -> None:
        views = [
            request.endpoint.request.read(request.message, copy=False)
            for request in pending
        ]
        field_names = set(views[0])
        if any(set(view) != field_names for view in views):
            raise ValueError("request endpoint tensor schemas differ")
        merged = {
            name: np.concatenate([view[name] for view in views], axis=0)
            for name in sorted(field_names)
        }
        requested_min = max(
            int(request.message.metadata.get("min_policy_version", -1))
            for request in pending
        )
        if self.replica.active_version < requested_min:
            raise RuntimeError(
                f"active policy {self.replica.active_version} is older than requested {requested_min}"
            )
        outputs, policy_version = self.replica.infer(merged, self.infer_fn)
        offset = 0
        completed_at = time.monotonic()
        latencies: list[float] = []
        total_items = 0
        for request in pending:
            count = request.message.item_count
            split = {name: value[offset : offset + count] for name, value in outputs.items()}
            submitted_at = float(
                request.message.metadata.get("submitted_at", request.received_at)
            )
            metadata = {
                "actor_id": request.endpoint.actor_id,
                "policy_version": policy_version,
                "submitted_at": submitted_at,
            }
            request.endpoint.response.send(
                split,
                item_count=count,
                sequence_id=request.message.sequence_id,
                metadata=metadata,
                timeout=self.response_timeout_seconds,
            )
            request.endpoint.request.release(request.message)
            latencies.append(max(0.0, completed_at - submitted_at))
            total_items += count
            offset += count
        self._metrics.record_success(
            request_messages=len(pending),
            items=total_items,
            latencies=latencies,
        )

    def _zero_response(self, endpoint: SharedInferenceEndpoint, item_count: int) -> dict[str, np.ndarray]:
        values: dict[str, np.ndarray] = {}
        for name, spec, _ in endpoint.response.descriptor.fields:
            values[name] = np.zeros((item_count, *spec.shape), dtype=np.dtype(spec.dtype))
        return values

    def _publish_error(self, request: _PendingRequest, error: BaseException) -> None:
        try:
            request.endpoint.response.send(
                self._zero_response(request.endpoint, request.message.item_count),
                item_count=request.message.item_count,
                sequence_id=request.message.sequence_id,
                metadata={
                    "actor_id": request.endpoint.actor_id,
                    "policy_version": self.replica.active_version,
                    "error": f"{type(error).__name__}: {error}",
                },
                timeout=self.response_timeout_seconds,
            )
        finally:
            try:
                request.endpoint.request.release(request.message)
            except RuntimeError:
                pass


class SharedInferenceClient:
    """Synchronous actor-side client for one dedicated shared-memory endpoint."""

    def __init__(self, endpoint: SharedInferenceEndpoint) -> None:
        self.endpoint = endpoint
        self._sequence = 0
        self._lock = threading.Lock()

    def infer(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        min_policy_version: int = -1,
        timeout: float = 5.0,
    ) -> InferenceResponse:
        if not inputs:
            raise ValueError("inputs must not be empty")
        sizes = {int(np.asarray(value).shape[0]) for value in inputs.values()}
        if len(sizes) != 1:
            raise ValueError("input tensors have inconsistent leading dimensions")
        item_count = sizes.pop()
        with self._lock:
            sequence_id = self._sequence
            self._sequence += 1
            started = time.monotonic()
            self.endpoint.request.send(
                inputs,
                item_count=item_count,
                sequence_id=sequence_id,
                metadata={
                    "actor_id": self.endpoint.actor_id,
                    "min_policy_version": int(min_policy_version),
                    "submitted_at": started,
                },
                timeout=timeout,
            )
            response = self.endpoint.response.receive(timeout=timeout)
            try:
                if response.sequence_id != sequence_id:
                    raise RuntimeError(
                        f"out-of-order response: expected {sequence_id}, got {response.sequence_id}"
                    )
                if "error" in response.metadata:
                    raise RuntimeError(str(response.metadata["error"]))
                outputs = self.endpoint.response.read(response, copy=True)
                policy_version = int(response.metadata.get("policy_version", -1))
            finally:
                self.endpoint.response.release(response)
            return InferenceResponse(
                outputs=outputs,
                policy_version=policy_version,
                sequence_id=sequence_id,
                latency_seconds=time.monotonic() - started,
            )
