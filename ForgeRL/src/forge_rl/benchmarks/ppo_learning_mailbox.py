from __future__ import annotations

from dataclasses import asdict
import threading
from typing import Mapping

import numpy as np
import torch

from forge_rl.runtime import (
    DoubleBufferedPolicyReplica,
    FastMailboxInferenceClient,
    FastMailboxNodeLocalInferenceService,
    MailboxInferenceEndpoint,
    PolicyRegistry,
)

from .legacy_v1 import ReferencePPOPolicy, run_synthetic_policy
from . import ppo_learning as _base

_PATCH_LOCK = threading.Lock()


class _MailboxLearningRuntime:
    """Learning-harness adapter for the fast synchronous mailbox data plane."""

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
        self.endpoints = [
            MailboxInferenceEndpoint.create(
                actor_id=actor_id,
                max_items=config.envs_per_actor,
                request_fields={"obs": ((observation_size,), np.float32)},
                response_fields={
                    "action": ((action_size,), np.float32),
                    "value": ((1,), np.float32),
                },
            )
            for actor_id in range(actor_count)
        ]

        def factory() -> ReferencePPOPolicy:
            return ReferencePPOPolicy(observation_size, action_size, hidden_size)

        self.registry = PolicyRegistry(history_size=2)
        snapshot = self.registry.publish(state_dict, version=0)
        self.service = FastMailboxNodeLocalInferenceService(
            endpoints=self.endpoints,
            replica=DoubleBufferedPolicyReplica(factory, device=config.device),
            infer_fn=run_synthetic_policy,
            max_batch_items=config.max_batch_items,
            min_batch_items=min(
                config.v2_min_batch_items,
                actor_count * config.envs_per_actor,
                config.max_batch_items,
            ),
            max_wait_ms=config.max_wait_ms,
            idle_wait_ms=50.0,
        )
        self.service.refresh_policy(snapshot)
        self.service.start()
        self.clients = [
            FastMailboxInferenceClient(endpoint, copy_outputs=False)
            for endpoint in self.endpoints
        ]

    def infer(
        self,
        actor_id: int,
        observation: np.ndarray,
        *,
        min_policy_version: int,
    ) -> tuple[dict[str, np.ndarray], int]:
        response = self.clients[actor_id].infer(
            {"obs": np.ascontiguousarray(observation, dtype=np.float32)},
            min_policy_version=min_policy_version,
            timeout=30.0,
        )
        return response.outputs, response.policy_version

    def update(self, state_dict: Mapping[str, torch.Tensor], version: int) -> int:
        snapshot = self.registry.publish(state_dict, version=version)
        return self.service.refresh_policy(snapshot)

    def metrics(self) -> dict[str, float | int | str]:
        inference = asdict(self.service.metrics())
        transport = asdict(self.service.transport_metrics())
        return {
            **inference,
            **{f"mailbox_{key}": value for key, value in transport.items()},
            "runtime": "fast-persistent-synchronous-mailbox",
        }

    def close(self) -> None:
        try:
            self.service.stop(timeout=10.0)
        finally:
            for client in self.clients:
                client.close()
            for endpoint in self.endpoints:
                endpoint.close()
                endpoint.unlink()


def run_learning_benchmark(
    config: _base.LearningBenchmarkConfig,
) -> _base.LearningBenchmarkReport:
    with _PATCH_LOCK:
        previous = _base._SharedLearningRuntime
        _base._SharedLearningRuntime = _MailboxLearningRuntime
        try:
            return _base.run_learning_benchmark(config)
        finally:
            _base._SharedLearningRuntime = previous
