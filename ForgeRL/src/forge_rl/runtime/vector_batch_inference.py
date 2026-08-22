from __future__ import annotations

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
from .policy_registry import PolicySnapshot


class VectorBatchInferenceRuntime:
    """Direct batched inference for vectorized node-local environments.

    The caller already owns one contiguous tensor containing observations for all participating
    Actor lanes. ForgeRL therefore does not split the tensor into small requests and attempt to
    reconstruct the same batch through timing-sensitive queues. The runtime executes exactly one
    policy forward and returns per-Actor views using the supplied item counts.

    This is the M1 batched-sampling path for C++ VectorEnv and other vectorized environments. It is
    intentionally in-process; separate Actor processes continue to use an explicit transport.
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
        device: torch.device | str = "cpu",
        amp_dtype: torch.dtype | None = None,
        copy_outputs: bool = False,
    ) -> None:
        if actor_count <= 0 or max_batch_items <= 0:
            raise ValueError("actor_count and max_batch_items must be positive")
        selected_device = torch.device(device)
        if selected_device.type not in {"cpu", "cuda"}:
            raise ValueError("vector-batch inference device must be cpu or cuda")
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA vector-batch inference requested but CUDA is unavailable")
        if amp_dtype not in {None, torch.float16, torch.bfloat16}:
            raise ValueError("amp_dtype must be float16, bfloat16 or None")
        if selected_device.type != "cuda" and amp_dtype is not None:
            raise ValueError("AMP vector-batch inference is supported only on CUDA")

        factory = lambda: module_type(*tuple(module_args), **dict(module_kwargs or {}))
        self.actor_count = int(actor_count)
        self.max_batch_items = int(max_batch_items)
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
        self._sequences = [0] * self.actor_count
        self._metrics = _InferenceMetrics()
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self._last_error: BaseException | None = None

    @property
    def policy_version(self) -> int:
        return self.replica.active_version

    @property
    def last_error(self) -> BaseException | None:
        return self._last_error

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("vector-batch runtime is closed")
        if self._started:
            raise RuntimeError("vector-batch runtime is already started")
        self._started = True

    def infer_batch(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        actor_item_counts: Mapping[int, int],
        min_policy_version: int = -1,
    ) -> dict[int, InferenceResponse]:
        if not self._started or self._closed:
            raise RuntimeError("vector-batch runtime is not active")
        if not inputs or not actor_item_counts:
            raise ValueError("inputs and actor_item_counts must not be empty")

        selected_counts: list[tuple[int, int]] = []
        for actor_id, count in actor_item_counts.items():
            actor = int(actor_id)
            items = int(count)
            if actor < 0 or actor >= self.actor_count:
                raise IndexError(f"actor_id {actor} outside [0, {self.actor_count})")
            if items <= 0:
                raise ValueError("every Actor partition must contain at least one item")
            selected_counts.append((actor, items))
        if len({actor for actor, _ in selected_counts}) != len(selected_counts):
            raise ValueError("actor_item_counts contains duplicate Actor IDs")

        total_items = sum(count for _, count in selected_counts)
        if total_items > self.max_batch_items:
            raise ValueError(
                f"vector batch has {total_items} items, capacity is {self.max_batch_items}"
            )
        arrays: dict[str, np.ndarray] = {}
        for name, value in inputs.items():
            array = np.ascontiguousarray(value)
            if array.ndim == 0 or int(array.shape[0]) != total_items:
                raise ValueError(
                    f"input field {name!r} must have leading dimension {total_items}"
                )
            arrays[name] = array

        submitted_at = time.monotonic()
        with self._lock:
            if self.replica.active_version < int(min_policy_version):
                raise RuntimeError(
                    f"active policy {self.replica.active_version} is older than "
                    f"requested {min_policy_version}"
                )
            sequences = {
                actor: self._sequences[actor]
                for actor, _ in selected_counts
            }
            for actor, _ in selected_counts:
                self._sequences[actor] += 1
            try:
                outputs, policy_version = self.replica.infer(arrays, self.infer_fn)
            except BaseException as error:
                self._last_error = error
                self._metrics.record_error()
                raise

            completed_at = time.monotonic()
            latency = max(0.0, completed_at - submitted_at)
            result: dict[int, InferenceResponse] = {}
            offset = 0
            for actor, count in selected_counts:
                stop = offset + count
                split = {
                    name: (
                        value[offset:stop].copy()
                        if self.copy_outputs
                        else value[offset:stop]
                    )
                    for name, value in outputs.items()
                }
                result[actor] = InferenceResponse(
                    outputs=split,
                    policy_version=policy_version,
                    sequence_id=sequences[actor],
                    latency_seconds=latency,
                )
                offset = stop
            if offset != total_items:
                raise RuntimeError("vector-batch response split did not consume all items")
            self._metrics.record_success(
                request_messages=len(selected_counts),
                items=total_items,
                latencies=[latency] * len(selected_counts),
            )
            return result

    def update_policy(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        version: int,
        timeout: float = 30.0,
    ) -> int:
        del timeout
        if not self._started or self._closed:
            raise RuntimeError("vector-batch runtime is not active")
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

    def close(self) -> None:
        self._started = False
        self._closed = True
