from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
import torch

from forge_rl.distributed.failure_recovery import RestartableProcess
from forge_rl.runtime import (
    CoordinatorHTTPError,
    NetworkCoordinatorClient,
    NetworkCoordinatorService,
    NetworkExperienceClient,
    NetworkExperienceService,
    NetworkPolicyClient,
    NetworkPolicyService,
    NodeRole,
    OnPolicyExperienceQueue,
    PolicyRegistry,
    TransitionBatch,
)


def _one_row_batch(*, step_id: int) -> TransitionBatch:
    observation = np.asarray([[float(step_id)]], dtype=np.float32)
    return TransitionBatch(
        obs={"obs": observation},
        next_obs={"obs": observation + 1.0},
        action={"action": np.asarray([0], dtype=np.int64)},
        reward=np.asarray([1.0], dtype=np.float32),
        terminated=np.asarray([False]),
        truncated=np.asarray([False]),
        valid_mask=np.asarray([1.0], dtype=np.float32),
        behavior_log_prob=np.asarray([0.0], dtype=np.float32),
        behavior_value=np.asarray([0.0], dtype=np.float32),
        policy_version=np.asarray([0], dtype=np.int64),
        episode_id=np.asarray([0], dtype=np.int64),
        step_id=np.asarray([step_id], dtype=np.int64),
        actor_id=np.asarray([0], dtype=np.int64),
        env_id=np.asarray([0], dtype=np.int64),
    )


def _actor_component(
    generation: int,
    ready_event: Any,
    stop_event: Any,
    status_queue: Any,
    coordinator_endpoint: str,
    experience_endpoint: str,
    token: str,
) -> None:
    coordinator = NetworkCoordinatorClient(
        coordinator_endpoint,
        bearer_token=token,
        timeout=5.0,
    )
    lease = coordinator.register(
        node_id="actor-0",
        role=NodeRole.ENV_RUNNER,
        endpoint=f"process://{os.getpid()}",
        metadata={"supervisor_generation": generation},
    )
    with NetworkExperienceClient(
        experience_endpoint,
        bearer_token=token,
        timeout=5.0,
    ) as experience:
        ack = experience.put(
            _one_row_batch(step_id=generation),
            request_id=f"actor-0-transition-{generation}",
        )
    status_queue.put(
        {
            "type": "actor_ready",
            "generation": generation,
            "coordinator_generation": lease.generation,
            "accepted_rows": ack.accepted_rows,
        }
    )
    ready_event.set()
    while not stop_event.wait(0.05):
        coordinator.heartbeat("actor-0", generation=lease.generation)


def _inference_component(
    generation: int,
    ready_event: Any,
    stop_event: Any,
    status_queue: Any,
    coordinator_endpoint: str,
    token: str,
) -> None:
    registry = PolicyRegistry(history_size=2)
    service = NetworkPolicyService(
        registry,
        bearer_token=token,
        socket_timeout=5.0,
    )
    service.start()
    coordinator = NetworkCoordinatorClient(
        coordinator_endpoint,
        bearer_token=token,
        timeout=5.0,
    )
    lease = coordinator.register(
        node_id="inference-0",
        role=NodeRole.INFERENCE,
        endpoint=service.endpoint,
        metadata={"supervisor_generation": generation},
    )
    status_queue.put(
        {
            "type": "inference_ready",
            "generation": generation,
            "coordinator_generation": lease.generation,
            "endpoint": service.endpoint,
        }
    )
    ready_event.set()
    last_version = -1
    try:
        while not stop_event.wait(0.02):
            coordinator.heartbeat("inference-0", generation=lease.generation)
            latest = registry.latest_version
            if latest <= last_version:
                continue
            snapshot = registry.get(latest)
            checksum = sum(
                float(value.to(torch.float64).sum())
                for value in snapshot.state_dict.values()
            )
            status_queue.put(
                {
                    "type": "policy_loaded",
                    "generation": generation,
                    "version": snapshot.version,
                    "checksum": checksum,
                }
            )
            last_version = latest
    finally:
        service.close()


def test_actor_restart_is_generation_fenced_and_does_not_duplicate_rows() -> None:
    token = "m2-failure-token"
    experience_queue = OnPolicyExperienceQueue(capacity_rows=8)
    coordinator_service = NetworkCoordinatorService(
        bearer_token=token,
        lease_seconds=2.0,
    )
    experience_service = NetworkExperienceService(
        experience_queue,
        bearer_token=token,
        put_timeout=0.0,
        socket_timeout=5.0,
    )
    coordinator_service.start()
    experience_service.start()
    supervisor = RestartableProcess(
        _actor_component,
        args=(coordinator_service.endpoint, experience_service.endpoint, token),
        name="actor-0",
    )
    coordinator = NetworkCoordinatorClient(
        coordinator_service.endpoint,
        bearer_token=token,
        timeout=5.0,
    )
    try:
        supervisor.start()
        supervisor.wait_ready(timeout=10.0)
        first = supervisor.receive_status(timeout=5.0)
        assert first["accepted_rows"] == 1
        assert first["coordinator_generation"] == 0

        supervisor.restart(timeout=10.0, reason="actor_crash")
        second = supervisor.receive_status(timeout=5.0)
        assert second["accepted_rows"] == 1
        assert second["coordinator_generation"] == 1

        with pytest.raises(CoordinatorHTTPError) as stale:
            coordinator.heartbeat("actor-0", generation=0)
        assert stale.value.code == "stale_generation"

        sampled = experience_queue.sample(
            min_rows=2,
            current_policy_version=0,
            max_policy_lag=0,
            timeout=0.0,
        )
        batches = [item.payload for item in sampled]
        assert sum(batch.size for batch in batches) == 2
        identities = [
            tuple(int(value) for value in row)
            for batch in batches
            for row in np.stack(
                [batch.actor_id, batch.env_id, batch.episode_id, batch.step_id],
                axis=1,
            )
        ]
        assert sorted(identities) == [(0, 0, 0, 0), (0, 0, 0, 1)]
        assert len(set(identities)) == 2
        assert supervisor.restart_count == 1
    finally:
        supervisor.close()
        experience_service.close()
        coordinator_service.close()
        experience_queue.close()


def test_inference_replica_restart_reloads_latest_policy_on_new_generation() -> None:
    token = "m2-policy-token"
    coordinator_service = NetworkCoordinatorService(
        bearer_token=token,
        lease_seconds=2.0,
    )
    coordinator_service.start()
    supervisor = RestartableProcess(
        _inference_component,
        args=(coordinator_service.endpoint, token),
        name="inference-0",
    )
    coordinator = NetworkCoordinatorClient(
        coordinator_service.endpoint,
        bearer_token=token,
        timeout=5.0,
    )
    state = {
        "linear.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "linear.bias": torch.asarray([0.25, -0.5], dtype=torch.float32),
        "step": torch.tensor(7, dtype=torch.int64),
    }
    expected_checksum = sum(
        float(value.to(torch.float64).sum()) for value in state.values()
    )
    try:
        supervisor.start()
        supervisor.wait_ready(timeout=10.0)
        first = supervisor.receive_status(timeout=5.0)
        assert first["type"] == "inference_ready"
        assert first["coordinator_generation"] == 0
        with NetworkPolicyClient(
            first["endpoint"],
            bearer_token=token,
            timeout=5.0,
        ) as policy:
            assert policy.publish(state, version=7, request_id="policy-v7-g0").version == 7
        loaded_first = supervisor.receive_status(timeout=5.0)
        assert loaded_first["version"] == 7
        assert loaded_first["checksum"] == pytest.approx(expected_checksum)

        supervisor.restart(timeout=10.0, reason="inference_replica_crash")
        second = supervisor.receive_status(timeout=5.0)
        assert second["type"] == "inference_ready"
        assert second["coordinator_generation"] == 1
        assert second["endpoint"] != first["endpoint"]
        with NetworkPolicyClient(
            second["endpoint"],
            bearer_token=token,
            timeout=5.0,
        ) as policy:
            assert policy.publish(state, version=7, request_id="policy-v7-g1").version == 7
        loaded_second = supervisor.receive_status(timeout=5.0)
        assert loaded_second["version"] == 7
        assert loaded_second["checksum"] == pytest.approx(expected_checksum)

        nodes = coordinator.active_nodes(NodeRole.INFERENCE)
        assert len(nodes) == 1
        assert nodes[0].generation == 1
        assert nodes[0].endpoint == second["endpoint"]
    finally:
        supervisor.close()
        coordinator_service.close()
