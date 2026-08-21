from __future__ import annotations

import pytest

from forge_rl.runtime.coordinator import InMemoryCoordinator, NodeRole


def test_node_generation_rejects_stale_heartbeat() -> None:
    coordinator = InMemoryCoordinator(lease_seconds=5.0)
    first = coordinator.register(
        node_id="node-1", role=NodeRole.LEARNER, endpoint="127.0.0.1:9000"
    )
    second = coordinator.register(
        node_id="node-1", role=NodeRole.LEARNER, endpoint="127.0.0.1:9001"
    )
    assert second.generation == first.generation + 1
    with pytest.raises(RuntimeError):
        coordinator.heartbeat("node-1", generation=first.generation)


def test_policy_version_is_monotonic() -> None:
    coordinator = InMemoryCoordinator()
    assert coordinator.commit_policy_version(0) == 0
    with pytest.raises(ValueError):
        coordinator.commit_policy_version(0)
