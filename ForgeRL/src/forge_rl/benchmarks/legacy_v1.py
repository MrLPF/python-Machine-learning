from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import multiprocessing as mp
import pickle
from queue import Empty
import threading
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

LEGACY_V1_ARCHIVE_SHA256 = "ce8e37af749654302d839c098b203563918c58f729cbe711400341003e9ac50c"


class SyntheticPolicy(nn.Module):
    def __init__(self, width: int, output_size: int = 4) -> None:
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, output_size))
        self.value = nn.Linear(width, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(observation), self.value(observation)


def run_synthetic_policy(
    module: SyntheticPolicy, batch: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    action, value = module(batch["obs"])
    return {"action": action, "value": value}


@dataclass(slots=True)
class _Request:
    actor_id: int
    sequence_id: int
    submitted_at: float
    observation: np.ndarray
    min_policy_version: int

    @property
    def items(self) -> int:
        return int(self.observation.shape[0])


@dataclass(slots=True)
class _Response:
    sequence_id: int
    policy_version: int
    action: np.ndarray
    value: np.ndarray
    completed_at: float
    error: str = ""


@dataclass(frozen=True, slots=True)
class LegacyRuntimeMetrics:
    batches: int
    request_messages: int
    items: int
    errors: int
    mean_batch_items: float
    batch_fill_ratio: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    serialized_request_bytes: int
    serialized_response_bytes: int


def _worker(
    width: int,
    output_size: int,
    state_dict: dict[str, torch.Tensor],
    policy_version: int,
    request_queue: Any,
    response_queues: list[Any],
    metric_queue: Any,
    ready: Any,
    max_batch_items: int,
    min_batch_items: int,
    max_wait_seconds: float,
    device_name: str,
    amp_dtype_name: str | None,
) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA legacy baseline requested but CUDA is unavailable")
        torch.cuda.set_device(device)
    amp_dtype = None if amp_dtype_name is None else getattr(torch, amp_dtype_name)
    model = SyntheticPolicy(width, output_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    pending: _Request | None = None
    ready.set()
    stop = False
    while not stop:
        first = pending if pending is not None else request_queue.get()
        pending = None
        if first is None:
            return
        batch = [first]
        items = first.items
        deadline = time.monotonic() + max_wait_seconds
        # Matches uploaded v1: collect only while below min_batch_size.
        while items < min_batch_items and items < max_batch_items:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                candidate = request_queue.get(timeout=remaining)
            except Empty:
                break
            if candidate is None:
                stop = True
                break
            if items + candidate.items > max_batch_items:
                pending = candidate
                break
            batch.append(candidate)
            items += candidate.items
        try:
            observation = np.concatenate([request.observation for request in batch], axis=0)
            inputs = torch.from_numpy(observation).to(device)
            autocast_enabled = device.type == "cuda" and amp_dtype is not None
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if autocast_enabled
                else nullcontext()
            )
            with torch.inference_mode(), autocast_context:
                action, value = model(inputs)
            action = action.detach().cpu().numpy()
            value = value.detach().cpu().numpy()
            offset = 0
            completed = time.monotonic()
            metric_queue.put((len(batch), items))
            for request in batch:
                count = request.items
                response_queues[request.actor_id].put(
                    _Response(
                        request.sequence_id,
                        policy_version,
                        action[offset : offset + count],
                        value[offset : offset + count],
                        completed,
                    )
                )
                offset += count
        except Exception as error:  # pragma: no cover
            completed = time.monotonic()
            metric_queue.put((len(batch), 0))
            for request in batch:
                response_queues[request.actor_id].put(
                    _Response(
                        request.sequence_id,
                        policy_version,
                        np.empty((0, output_size), np.float32),
                        np.empty((0, 1), np.float32),
                        completed,
                        f"{type(error).__name__}: {error}",
                    )
                )


class LegacyV1InferenceRuntime:
    """Frozen v1 central predictor data plane: Queue/pickle in both directions."""

    def __init__(
        self,
        *,
        actor_count: int,
        width: int,
        state_dict: Mapping[str, torch.Tensor],
        output_size: int = 4,
        policy_version: int = 0,
        max_batch_items: int = 128,
        min_batch_items: int = 1,
        max_wait_ms: float = 2.0,
        device: torch.device | str = "cpu",
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        if actor_count <= 0 or width <= 0 or output_size <= 0:
            raise ValueError("actor_count, width and output_size must be positive")
        if not 1 <= min_batch_items <= max_batch_items:
            raise ValueError("invalid legacy batch limits")
        selected_device = torch.device(device)
        if selected_device.type not in {"cpu", "cuda"}:
            raise ValueError("legacy benchmark device must be cpu or cuda")
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA legacy baseline requested but CUDA is unavailable")
        if amp_dtype not in {None, torch.float16, torch.bfloat16}:
            raise ValueError("amp_dtype must be float16, bfloat16 or None")
        if selected_device.type != "cuda" and amp_dtype is not None:
            raise ValueError("AMP baseline is supported only on CUDA")
        self.actor_count = actor_count
        self.width = width
        self.max_batch_items = max_batch_items
        context = mp.get_context("spawn")
        self._requests = context.Queue()
        self._responses = [context.Queue() for _ in range(actor_count)]
        self._batch_metrics = context.Queue()
        self._ready = context.Event()
        state = {name: value.detach().cpu().clone() for name, value in state_dict.items()}
        self._process = context.Process(
            target=_worker,
            args=(
                width,
                output_size,
                state,
                policy_version,
                self._requests,
                self._responses,
                self._batch_metrics,
                self._ready,
                max_batch_items,
                min_batch_items,
                max_wait_ms / 1000.0,
                str(selected_device),
                None if amp_dtype is None else str(amp_dtype).removeprefix("torch."),
            ),
            daemon=True,
        )
        self._locks = [threading.Lock() for _ in range(actor_count)]
        self._latencies: list[float] = []
        self._request_bytes = 0
        self._response_bytes = 0
        self._errors = 0
        self._metrics_lock = threading.Lock()
        self._started = False

    def start(self, timeout: float = 15.0) -> None:
        self._process.start()
        if not self._ready.wait(timeout):
            self._process.terminate()
            raise TimeoutError("legacy predictor did not become ready")
        self._started = True

    def infer(
        self,
        actor_id: int,
        sequence_id: int,
        observation: np.ndarray,
        *,
        min_policy_version: int = -1,
        timeout: float = 10.0,
    ) -> tuple[dict[str, np.ndarray], int, int]:
        if not self._started:
            raise RuntimeError("legacy runtime is not started")
        array = np.ascontiguousarray(observation, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.width:
            raise ValueError(f"observation must have shape [items, {self.width}]")
        if actor_id < 0 or actor_id >= self.actor_count:
            raise IndexError(f"actor_id {actor_id} outside [0, {self.actor_count})")
        with self._locks[actor_id]:
            request = _Request(actor_id, sequence_id, time.monotonic(), array, min_policy_version)
            request_bytes = len(pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL))
            self._requests.put(request, timeout=timeout)
            response: _Response = self._responses[actor_id].get(timeout=timeout)
            if response.sequence_id != sequence_id or response.policy_version < min_policy_version:
                raise RuntimeError("legacy response identity or policy version mismatch")
            response_bytes = len(pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL))
            with self._metrics_lock:
                self._latencies.append(max(0.0, response.completed_at - request.submitted_at))
                self._request_bytes += request_bytes
                self._response_bytes += response_bytes
                self._errors += int(bool(response.error))
            if response.error:
                raise RuntimeError(response.error)
            return {"action": response.action, "value": response.value}, response.policy_version, response.sequence_id

    def metrics(self, expected_requests: int, timeout: float = 2.0) -> LegacyRuntimeMetrics:
        batches: list[tuple[int, int]] = []
        seen = 0
        deadline = time.monotonic() + timeout
        while seen < expected_requests and time.monotonic() < deadline:
            try:
                metric = self._batch_metrics.get(timeout=0.05)
            except Empty:
                continue
            batches.append(metric)
            seen += metric[0]
        with self._metrics_lock:
            latency = np.asarray(self._latencies, np.float64)
            percentiles = np.percentile(latency, [50, 95, 99]) if latency.size else [0.0] * 3
            items = sum(value[1] for value in batches)
            mean = items / len(batches) if batches else 0.0
            return LegacyRuntimeMetrics(
                len(batches),
                seen,
                items,
                self._errors,
                mean,
                mean / self.max_batch_items if batches else 0.0,
                *(float(value * 1000.0) for value in percentiles),
                self._request_bytes,
                self._response_bytes,
            )

    def close(self, timeout: float = 10.0) -> None:
        if not self._started:
            return
        if self._process.is_alive():
            self._requests.put(None)
            self._process.join(timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(2.0)
            raise TimeoutError("legacy predictor required forced termination")
        for queue in [self._requests, *self._responses, self._batch_metrics]:
            queue.close()
            queue.join_thread()
        self._started = False
