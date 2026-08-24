from __future__ import annotations

from dataclasses import asdict
import threading
from typing import Mapping

import numpy as np
import torch

from forge_rl.runtime import VectorBatchInferenceRuntime

from .legacy_v1 import ReferencePPOPolicy, run_synthetic_policy
from . import ppo_learning as _base

_PATCH_LOCK = threading.Lock()


class _VectorBatchLearningRuntime:
    """M1 learning adapter for the production vector-batch inference subject.

    The benchmark core still invokes one logical Actor call per thread. Each Actor writes directly
    into its fixed slice of one contiguous observation buffer. A barrier action performs exactly
    one policy forward for the complete vectorized environment wave and exposes per-Actor output
    views. No timing window, request queue, or post-arrival concatenation decides batch membership.
    """

    def __init__(
        self,
        *,
        actor_count: int,
        observation_size: int,
        action_size: int,
        hidden_size: int,
        state_dict: Mapping[str, torch.Tensor],
        config: _base.LearningBenchmarkConfig,
    ) -> None:
        total_items = actor_count * config.envs_per_actor
        if total_items > config.max_batch_items:
            raise ValueError(
                "vector-batch learning requires actors * envs_per_actor <= max_batch_items"
            )
        self.actor_count = int(actor_count)
        self.envs_per_actor = int(config.envs_per_actor)
        self.observation_size = int(observation_size)
        self._batch = np.empty(
            (total_items, self.observation_size),
            dtype=np.float32,
        )
        self._minimum_versions = [-1] * self.actor_count
        self._responses = [None] * self.actor_count
        self._wave_error: BaseException | None = None
        self._actor_counts = {
            actor_id: self.envs_per_actor for actor_id in range(self.actor_count)
        }
        self.runtime = VectorBatchInferenceRuntime(
            actor_count=self.actor_count,
            module_type=ReferencePPOPolicy,
            module_args=(observation_size, action_size, hidden_size),
            infer_fn=run_synthetic_policy,
            state_dict=state_dict,
            policy_version=0,
            max_batch_items=config.max_batch_items,
            device=config.device,
            copy_outputs=False,
        )
        self.runtime.start()
        self._barrier = threading.Barrier(
            self.actor_count,
            action=self._execute_wave,
        )

    def _execute_wave(self) -> None:
        try:
            responses = self.runtime.infer_batch(
                {"obs": self._batch},
                actor_item_counts=self._actor_counts,
                min_policy_version=max(self._minimum_versions),
            )
        except BaseException as error:
            self._wave_error = error
            self._responses = [None] * self.actor_count
        else:
            self._wave_error = None
            self._responses = [responses[index] for index in range(self.actor_count)]

    def infer(
        self,
        actor_id: int,
        observation: np.ndarray,
        *,
        min_policy_version: int,
    ) -> tuple[dict[str, np.ndarray], int]:
        actor = int(actor_id)
        if actor < 0 or actor >= self.actor_count:
            raise IndexError(f"actor_id {actor} outside [0, {self.actor_count})")
        selected = np.asarray(observation, dtype=np.float32)
        expected = (self.envs_per_actor, self.observation_size)
        if selected.shape != expected:
            raise ValueError(f"Actor observation shape {selected.shape} != {expected}")
        start = actor * self.envs_per_actor
        stop = start + self.envs_per_actor
        np.copyto(self._batch[start:stop], selected, casting="no")
        self._minimum_versions[actor] = int(min_policy_version)
        try:
            self._barrier.wait(timeout=30.0)
        except threading.BrokenBarrierError as error:
            if self._wave_error is not None:
                raise RuntimeError("vector-batch learning inference failed") from self._wave_error
            raise RuntimeError("vector-batch learning barrier broke") from error
        if self._wave_error is not None:
            raise RuntimeError("vector-batch learning inference failed") from self._wave_error
        response = self._responses[actor]
        if response is None:
            raise RuntimeError("vector-batch learning response is missing")
        return response.outputs, response.policy_version

    def update(self, state_dict: Mapping[str, torch.Tensor], version: int) -> int:
        return self.runtime.update_policy(state_dict, version=version)

    def metrics(self) -> dict[str, float | int | str]:
        return {
            **asdict(self.runtime.metrics()),
            "runtime": "vector-batch-learning",
            "batch_membership": "producer-defined",
        }

    def close(self) -> None:
        self._barrier.abort()
        self.runtime.close()


def run_learning_benchmark(
    config: _base.LearningBenchmarkConfig,
) -> _base.LearningBenchmarkReport:
    """Run the M1 learning gate with the same vector-batch subject as the throughput gate."""

    with _PATCH_LOCK:
        previous = _base._SharedLearningRuntime
        _base._SharedLearningRuntime = _VectorBatchLearningRuntime
        try:
            return _base.run_learning_benchmark(config)
        finally:
            _base._SharedLearningRuntime = previous
