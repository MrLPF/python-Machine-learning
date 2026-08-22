from __future__ import annotations

from collections import deque
from contextlib import ExitStack
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .inference import InferenceFn, InferenceMetricsSnapshot, InferenceResponse


class ActorLocalInferenceRuntime:
    """CPU fast path that keeps one read-only policy replica beside each Actor.

    This path is intentionally restricted to CPU inference. It removes request/response IPC for
    small policies where central batching costs more than the model forward pass. Every Actor owns
    one replica and one lock, so independent Actors execute concurrently while policy activation is
    fenced across all replicas between rollout rounds.
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
        max_batch_items: int = 1,
        copy_outputs: bool = False,
        latency_samples: int = 4096,
    ) -> None:
        if actor_count <= 0:
            raise ValueError("actor_count must be positive")
        if max_batch_items <= 0:
            raise ValueError("max_batch_items must be positive")
        if latency_samples <= 0:
            raise ValueError("latency_samples must be positive")

        self.actor_count = int(actor_count)
        self.max_batch_items = int(max_batch_items)
        self.infer_fn = infer_fn
        self.copy_outputs = bool(copy_outputs)
        self._locks = [threading.RLock() for _ in range(self.actor_count)]
        self._update_lock = threading.Lock()
        self._models = [
            module_type(*tuple(module_args), **dict(module_kwargs or {})).cpu().eval()
            for _ in range(self.actor_count)
        ]
        state = {
            name: value.detach().cpu().clone(memory_format=torch.preserve_format)
            for name, value in state_dict.items()
        }
        for model in self._models:
            model.load_state_dict(state, strict=True)
            model.eval()

        self._policy_version = int(policy_version)
        self._batches = [0] * self.actor_count
        self._items = [0] * self.actor_count
        self._errors = [0] * self.actor_count
        self._latencies = [
            deque(maxlen=int(latency_samples)) for _ in range(self.actor_count)
        ]
        self._started = False
        self._closed = False

    @property
    def policy_version(self) -> int:
        return self._policy_version

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self._models[0].parameters())

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("actor-local runtime is closed")
        if self._started:
            raise RuntimeError("actor-local runtime is already started")
        self._started = True

    def infer(
        self,
        actor_id: int,
        inputs: Mapping[str, np.ndarray],
        *,
        min_policy_version: int = -1,
        timeout: float = 15.0,
    ) -> InferenceResponse:
        del timeout
        if not self._started or self._closed:
            raise RuntimeError("actor-local runtime is not active")
        selected = int(actor_id)
        if selected < 0 or selected >= self.actor_count:
            raise IndexError("actor_id outside actor-local runtime")
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
            raise ValueError("item_count outside actor-local capacity")

        started = time.monotonic()
        lock = self._locks[selected]
        try:
            with lock:
                version = self._policy_version
                if version < int(min_policy_version):
                    raise RuntimeError(
                        f"active policy {version} is older than requested "
                        f"{int(min_policy_version)}"
                    )
                tensors = {
                    name: torch.from_numpy(array)
                    for name, array in arrays.items()
                }
                with torch.inference_mode():
                    raw_outputs = dict(self.infer_fn(self._models[selected], tensors))
                outputs: dict[str, np.ndarray] = {}
                for name, value in raw_outputs.items():
                    if not isinstance(value, torch.Tensor):
                        raise TypeError(f"inference output {name!r} is not a tensor")
                    if value.ndim == 0 or int(value.shape[0]) != item_count:
                        raise ValueError(
                            f"inference output {name!r} leading dimension must be "
                            f"{item_count}"
                        )
                    array = value.detach().cpu().numpy()
                    outputs[name] = array.copy() if self.copy_outputs else array
            latency = max(0.0, time.monotonic() - started)
            self._batches[selected] += 1
            self._items[selected] += item_count
            self._latencies[selected].append(latency)
            return InferenceResponse(
                outputs=outputs,
                policy_version=version,
                sequence_id=self._batches[selected] - 1,
                latency_seconds=latency,
            )
        except BaseException:
            self._errors[selected] += 1
            raise

    def update_policy(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        version: int,
        timeout: float = 30.0,
    ) -> int:
        del timeout
        if not self._started or self._closed:
            raise RuntimeError("actor-local runtime is not active")
        selected = int(version)
        if selected <= self._policy_version:
            raise ValueError("policy version must increase monotonically")
        state = {
            name: value.detach().cpu().clone(memory_format=torch.preserve_format)
            for name, value in state_dict.items()
        }

        with self._update_lock, ExitStack() as stack:
            for lock in self._locks:
                stack.enter_context(lock)
            for model in self._models:
                model.load_state_dict(state, strict=True)
                model.eval()
            self._policy_version = selected
        return selected

    def metrics(self) -> InferenceMetricsSnapshot:
        batches = sum(self._batches)
        items = sum(self._items)
        errors = sum(self._errors)
        latencies = np.asarray(
            [value for rows in self._latencies for value in rows],
            dtype=np.float64,
        )
        percentiles = (
            np.percentile(latencies, [50, 95, 99]).tolist()
            if latencies.size
            else [0.0, 0.0, 0.0]
        )
        mean_batch = items / batches if batches else 0.0
        return InferenceMetricsSnapshot(
            batches=batches,
            request_messages=batches,
            items=items,
            errors=errors,
            mean_batch_items=mean_batch,
            batch_fill_ratio=mean_batch / self.max_batch_items,
            latency_p50_ms=float(percentiles[0] * 1000.0),
            latency_p95_ms=float(percentiles[1] * 1000.0),
            latency_p99_ms=float(percentiles[2] * 1000.0),
        )

    def close(self, *, timeout: float = 0.0) -> None:
        del timeout
        self._closed = True
        self._started = False
