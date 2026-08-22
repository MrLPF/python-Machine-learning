from __future__ import annotations

from dataclasses import asdict
import threading
from typing import Mapping

import numpy as np
import torch

from forge_rl.runtime import (
    InProcessBatchingInferenceRuntime,
    ProcessMailboxInferenceRuntime,
)

from .legacy_v1 import ReferencePPOPolicy, run_synthetic_policy
from . import ppo_learning as _base

_PATCH_LOCK = threading.Lock()


class _InProcessLearningRuntime:
    """Learning-harness adapter for zero-serialization local batching."""

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
        self.runtime = InProcessBatchingInferenceRuntime(
            actor_count=actor_count,
            module_type=ReferencePPOPolicy,
            module_args=(observation_size, action_size, hidden_size),
            infer_fn=run_synthetic_policy,
            state_dict=state_dict,
            policy_version=0,
            max_batch_items=config.max_batch_items,
            min_batch_items=min(
                config.v2_min_batch_items,
                actor_count * config.envs_per_actor,
                config.max_batch_items,
            ),
            max_wait_ms=config.max_wait_ms,
            device=config.device,
            copy_outputs=False,
        )
        self.runtime.start()

    def infer(
        self,
        actor_id: int,
        observation: np.ndarray,
        *,
        min_policy_version: int,
    ) -> tuple[dict[str, np.ndarray], int]:
        response = self.runtime.infer(
            actor_id,
            {"obs": np.ascontiguousarray(observation, dtype=np.float32)},
            min_policy_version=min_policy_version,
            timeout=30.0,
        )
        return response.outputs, response.policy_version

    def update(self, state_dict: Mapping[str, torch.Tensor], version: int) -> int:
        return self.runtime.update_policy(
            state_dict,
            version=version,
            timeout=30.0,
        )

    def metrics(self) -> dict[str, float | int | str]:
        return {
            **asdict(self.runtime.metrics()),
            "runtime": "in-process-zero-serialization-batching",
        }

    def close(self) -> None:
        self.runtime.close(timeout=15.0)


class _ProcessMailboxLearningRuntime:
    """Learning-harness adapter for the process-isolated mailbox predictor."""

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
        self.runtime = ProcessMailboxInferenceRuntime(
            actor_count=actor_count,
            max_items=config.envs_per_actor,
            request_fields={"obs": ((observation_size,), np.float32)},
            response_fields={
                "action": ((action_size,), np.float32),
                "value": ((1,), np.float32),
            },
            module_type=ReferencePPOPolicy,
            module_args=(observation_size, action_size, hidden_size),
            infer_fn=run_synthetic_policy,
            state_dict=state_dict,
            policy_version=0,
            max_batch_items=config.max_batch_items,
            min_batch_items=min(
                config.v2_min_batch_items,
                actor_count * config.envs_per_actor,
                config.max_batch_items,
            ),
            max_wait_ms=config.max_wait_ms,
            device=config.device,
            copy_outputs=False,
        )
        self.runtime.start()

    def infer(
        self,
        actor_id: int,
        observation: np.ndarray,
        *,
        min_policy_version: int,
    ) -> tuple[dict[str, np.ndarray], int]:
        response = self.runtime.infer(
            actor_id,
            {"obs": np.ascontiguousarray(observation, dtype=np.float32)},
            min_policy_version=min_policy_version,
            timeout=30.0,
        )
        return response.outputs, response.policy_version

    def update(self, state_dict: Mapping[str, torch.Tensor], version: int) -> int:
        return self.runtime.update_policy(
            state_dict,
            version=version,
            timeout=30.0,
        )

    def metrics(self) -> dict[str, float | int | str]:
        inference, transport = self.runtime.metrics_pair()
        return {
            **asdict(inference),
            **{f"mailbox_{key}": value for key, value in asdict(transport).items()},
            "runtime": "process-isolated-persistent-mailbox",
            "predictor_pid": int(self.runtime.process_pid or -1),
        }

    def close(self) -> None:
        self.runtime.close(timeout=15.0)


def run_learning_benchmark(
    config: _base.LearningBenchmarkConfig,
) -> _base.LearningBenchmarkReport:
    """Run identical PPO mathematics with a workload-appropriate v2 topology."""

    runtime_class = (
        _InProcessLearningRuntime
        if config.device == "cpu"
        else _ProcessMailboxLearningRuntime
    )
    with _PATCH_LOCK:
        previous = _base._SharedLearningRuntime
        _base._SharedLearningRuntime = runtime_class
        try:
            return _base.run_learning_benchmark(config)
        finally:
            _base._SharedLearningRuntime = previous
