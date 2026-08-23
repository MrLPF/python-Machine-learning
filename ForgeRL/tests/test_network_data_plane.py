from __future__ import annotations

import numpy as np
import pytest
import torch

from forge_rl.runtime import (
    DataChannelError,
    NetworkExperienceClient,
    NetworkExperienceService,
    NetworkPolicyClient,
    NetworkPolicyService,
    OnPolicyExperienceQueue,
    PolicyRegistry,
    TransitionBatch,
    decode_policy_state_dict,
    decode_transition_batch,
    encode_policy_state_dict,
    encode_transition_batch,
)


def _transition_batch(*, version: int = 0) -> TransitionBatch:
    rows = 4
    return TransitionBatch(
        obs={
            "obs": np.arange(rows * 3, dtype=np.float32).reshape(rows, 3),
            "mask": np.ones((rows, 2), dtype=np.bool_),
        },
        next_obs={
            "obs": np.arange(1, rows * 3 + 1, dtype=np.float32).reshape(rows, 3),
            "mask": np.ones((rows, 2), dtype=np.bool_),
        },
        action={"action": np.asarray([0, 1, 0, 1], dtype=np.int64)},
        reward=np.asarray([1.0, 0.5, -0.25, 3.0], dtype=np.float32),
        terminated=np.asarray([False, False, True, False]),
        truncated=np.asarray([False, False, False, False]),
        valid_mask=np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        behavior_log_prob=np.asarray([-0.1, -0.2, -0.3, -0.4], dtype=np.float32),
        behavior_value=np.asarray([0.5, 0.4, 0.3, 0.2], dtype=np.float32),
        policy_version=np.full(rows, version, dtype=np.int64),
        episode_id=np.zeros(rows, dtype=np.int64),
        step_id=np.zeros(rows, dtype=np.int64),
        actor_id=np.asarray([0, 0, 1, 1], dtype=np.int64),
        env_id=np.asarray([0, 1, 0, 1], dtype=np.int64),
        hidden_in={"state": np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)},
        extras={"sim_time": np.arange(rows, dtype=np.float64)},
    )


def _assert_batches_equal(actual: TransitionBatch, expected: TransitionBatch) -> None:
    assert actual.size == expected.size
    for name in ("obs", "next_obs", "action", "hidden_in", "extras"):
        actual_tree = getattr(actual, name)
        expected_tree = getattr(expected, name)
        assert actual_tree is not None
        assert expected_tree is not None
        assert set(actual_tree) == set(expected_tree)
        for key in expected_tree:
            np.testing.assert_array_equal(actual_tree[key], expected_tree[key])
    for name in (
        "reward",
        "terminated",
        "truncated",
        "valid_mask",
        "behavior_log_prob",
        "behavior_value",
        "policy_version",
        "episode_id",
        "step_id",
        "actor_id",
        "env_id",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_transition_and_policy_codecs_are_pickle_free_round_trips() -> None:
    batch = _transition_batch()
    metadata, payload = encode_transition_batch(batch)
    restored = decode_transition_batch(metadata, payload)
    _assert_batches_equal(restored, batch)

    state = {
        "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "counter": torch.tensor(7, dtype=torch.int64),
        "bf16": torch.arange(4, dtype=torch.bfloat16),
    }
    policy_metadata, policy_payload = encode_policy_state_dict(state)
    restored_state = decode_policy_state_dict(policy_metadata, policy_payload)
    assert set(restored_state) == set(state)
    for name, value in state.items():
        assert restored_state[name].dtype == value.dtype
        assert torch.equal(restored_state[name], value)


def test_network_experience_channel_is_idempotent_and_preserves_backpressure() -> None:
    batch = _transition_batch()
    queue = OnPolicyExperienceQueue(capacity_rows=3)
    service = NetworkExperienceService(
        queue,
        bearer_token="experience-secret",
        put_timeout=0.0,
    )
    service.start()
    client = NetworkExperienceClient(
        service.endpoint,
        bearer_token="experience-secret",
        timeout=5.0,
    )
    try:
        first = client.put(batch, request_id="trajectory-0")
        assert first.accepted_rows == 3
        assert first.policy_version_min == 0
        assert first.policy_version_max == 0
        assert first.queue_rows == 3

        replayed = client.put(batch, request_id="trajectory-0")
        assert replayed == first
        assert queue.rows == 3
        assert service.metrics().accepted_requests == 1

        with pytest.raises(DataChannelError) as backpressure:
            client.put(batch, request_id="trajectory-1")
        assert backpressure.value.code == "backpressure"
        assert backpressure.value.retryable

        sampled = queue.sample(
            min_rows=3,
            current_policy_version=0,
            max_policy_lag=0,
            timeout=0.0,
        )
        assert len(sampled) == 1
        assert sampled[0].train_rows == 3
        assert isinstance(sampled[0].payload, TransitionBatch)
        _assert_batches_equal(sampled[0].payload, batch)

        with NetworkExperienceClient(
            service.endpoint,
            bearer_token="wrong-secret",
            timeout=5.0,
        ) as unauthorized:
            with pytest.raises(DataChannelError) as denied:
                unauthorized.put(batch, request_id="unauthorized-trajectory")
        assert denied.value.code == "unauthorized"

        metrics = service.metrics()
        assert metrics.accepted_requests == 1
        assert metrics.accepted_rows == 3
        assert metrics.rejected_requests == 2
    finally:
        client.close()
        service.close()
        queue.close()


def test_network_policy_channel_is_idempotent_monotonic_and_authenticated() -> None:
    registry = PolicyRegistry(history_size=3)
    service = NetworkPolicyService(
        registry,
        bearer_token="policy-secret",
    )
    service.start()
    client = NetworkPolicyClient(
        service.endpoint,
        bearer_token="policy-secret",
        timeout=5.0,
    )
    state_v0 = {
        "linear.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "linear.bias": torch.asarray([0.25, -0.5], dtype=torch.float32),
        "step": torch.tensor(0, dtype=torch.int64),
    }
    state_v1 = {
        name: value.clone()
        for name, value in state_v0.items()
    }
    state_v1["linear.weight"].add_(1.0)
    state_v1["step"].fill_(1)
    try:
        first = client.publish(
            state_v0,
            version=0,
            metadata={"learner_rank": 0},
            request_id="policy-0",
        )
        assert first.version == 0
        assert first.tensor_count == len(state_v0)

        replayed = client.publish(
            state_v0,
            version=0,
            metadata={"learner_rank": 0},
            request_id="policy-0",
        )
        assert replayed == first
        assert service.metrics().accepted_requests == 1

        snapshot = registry.get(0)
        assert snapshot.metadata == {"learner_rank": 0}
        for name, value in state_v0.items():
            assert torch.equal(snapshot.state_dict[name], value)

        with pytest.raises(DataChannelError) as conflict:
            client.publish(
                state_v0,
                version=0,
                request_id="policy-duplicate-version",
            )
        assert conflict.value.code == "policy_version_conflict"
        assert not conflict.value.retryable

        second = client.publish(
            state_v1,
            version=1,
            metadata={"learner_rank": 0, "optimizer_step": 1},
            request_id="policy-1",
        )
        assert second.version == 1
        assert registry.latest_version == 1
        assert torch.equal(registry.get(1).state_dict["linear.weight"], state_v1["linear.weight"])

        with NetworkPolicyClient(
            service.endpoint,
            bearer_token="wrong-secret",
            timeout=5.0,
        ) as unauthorized:
            with pytest.raises(DataChannelError) as denied:
                unauthorized.publish(
                    state_v1,
                    version=2,
                    request_id="unauthorized-policy",
                )
        assert denied.value.code == "unauthorized"

        metrics = service.metrics()
        assert metrics.accepted_requests == 2
        assert metrics.published_versions == 2
        assert metrics.rejected_requests == 2
    finally:
        client.close()
        service.close()
