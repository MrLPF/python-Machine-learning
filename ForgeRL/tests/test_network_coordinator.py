from __future__ import annotations

import pytest

from forge_rl.runtime import (
    CoordinatorHTTPError,
    InMemoryCoordinator,
    NetworkCoordinatorClient,
    NetworkCoordinatorService,
    NodeRole,
)


def test_network_coordinator_registration_discovery_and_policy_version() -> None:
    with NetworkCoordinatorService(lease_seconds=5.0) as service:
        client = NetworkCoordinatorClient(service.endpoint)
        assert client.health()

        lease = client.register(
            node_id="learner-0",
            role=NodeRole.LEARNER,
            endpoint="tcp://127.0.0.1:9100",
            resources={"cpu": 8, "gpu": 1},
            metadata={"hostname": "node-a"},
        )
        assert lease.node_id == "learner-0"
        assert lease.role is NodeRole.LEARNER
        assert lease.generation == 0
        assert lease.expires_in_seconds > 0

        refreshed = client.heartbeat("learner-0", generation=lease.generation)
        assert refreshed.generation == lease.generation
        assert refreshed.expires_in_seconds > 0

        learners = client.active_nodes(NodeRole.LEARNER)
        assert [node.node_id for node in learners] == ["learner-0"]
        assert client.active_nodes(NodeRole.ENV_RUNNER) == []

        assert client.commit_policy_version(0) == 0
        state = client.state()
        assert state["policy_version"] == 0
        assert state["lease_seconds"] == 5.0
        assert state["nodes"][0]["resources"] == {"cpu": 8.0, "gpu": 1.0}


def test_generation_fencing_survives_expiry_and_reregistration() -> None:
    coordinator = InMemoryCoordinator(lease_seconds=5.0)
    with NetworkCoordinatorService(coordinator=coordinator) as service:
        client = NetworkCoordinatorClient(service.endpoint)
        first = client.register(
            node_id="actor-7",
            role=NodeRole.ENV_RUNNER,
            endpoint="tcp://127.0.0.1:9200",
        )
        local = coordinator.active_nodes()[0]
        assert coordinator.expire(now=local.expires_at + 1.0) == ["actor-7"]

        second = client.register(
            node_id="actor-7",
            role=NodeRole.ENV_RUNNER,
            endpoint="tcp://127.0.0.1:9201",
        )
        assert second.generation == first.generation + 1

        with pytest.raises(CoordinatorHTTPError) as stale:
            client.heartbeat("actor-7", generation=first.generation)
        assert stale.value.status == 409
        assert stale.value.code == "stale_generation"

        with pytest.raises(CoordinatorHTTPError) as missing:
            client.heartbeat("missing-node", generation=0)
        assert missing.value.status == 404
        assert missing.value.code == "node_not_found"


def test_network_coordinator_bearer_auth_and_conflict_errors() -> None:
    with NetworkCoordinatorService(bearer_token="test-secret") as service:
        unauthorized = NetworkCoordinatorClient(service.endpoint)
        with pytest.raises(CoordinatorHTTPError) as denied:
            unauthorized.health()
        assert denied.value.status == 401
        assert denied.value.code == "unauthorized"

        client = NetworkCoordinatorClient(service.endpoint, bearer_token="test-secret")
        assert client.health()
        assert client.commit_policy_version(3) == 3
        with pytest.raises(CoordinatorHTTPError) as conflict:
            client.commit_policy_version(3)
        assert conflict.value.status == 409
        assert conflict.value.code == "policy_version_conflict"

        with pytest.raises(CoordinatorHTTPError) as invalid:
            client.register(
                node_id="bad-role",
                role="not-a-role",
                endpoint="tcp://127.0.0.1:9300",
            )
        assert invalid.value.status == 400
        assert invalid.value.code == "invalid_request"
