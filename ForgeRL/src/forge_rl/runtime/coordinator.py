from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import threading
import time
from typing import Any, Mapping


class NodeRole(StrEnum):
    ENV_RUNNER = "env_runner"
    INFERENCE = "inference"
    EXPERIENCE = "experience"
    LEARNER = "learner"
    EVALUATOR = "evaluator"


@dataclass(slots=True)
class NodeLease:
    node_id: str
    role: NodeRole
    endpoint: str
    resources: dict[str, float]
    metadata: dict[str, Any]
    generation: int
    heartbeat_at: float
    expires_at: float

    @property
    def alive(self) -> bool:
        return time.monotonic() < self.expires_at


class InMemoryCoordinator:
    """Reference control-plane implementation used by tests and local mode."""

    def __init__(self, *, lease_seconds: float = 10.0) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.lease_seconds = float(lease_seconds)
        self._nodes: dict[str, NodeLease] = {}
        self._policy_version = -1
        self._lock = threading.RLock()

    @property
    def policy_version(self) -> int:
        with self._lock:
            return self._policy_version

    def commit_policy_version(self, version: int) -> int:
        with self._lock:
            selected = int(version)
            if selected <= self._policy_version:
                raise ValueError(
                    f"policy version must increase: current={self._policy_version}, new={selected}"
                )
            self._policy_version = selected
            return selected

    def register(
        self,
        *,
        node_id: str,
        role: NodeRole | str,
        endpoint: str,
        resources: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> NodeLease:
        if not node_id or not endpoint:
            raise ValueError("node_id and endpoint are required")
        normalized_role = role if isinstance(role, NodeRole) else NodeRole(role)
        with self._lock:
            previous = self._nodes.get(node_id)
            generation = 0 if previous is None else previous.generation + 1
            now = time.monotonic()
            lease = NodeLease(
                node_id=node_id,
                role=normalized_role,
                endpoint=endpoint,
                resources={key: float(value) for key, value in (resources or {}).items()},
                metadata=dict(metadata or {}),
                generation=generation,
                heartbeat_at=now,
                expires_at=now + self.lease_seconds,
            )
            self._nodes[node_id] = lease
            return lease

    def heartbeat(self, node_id: str, *, generation: int) -> NodeLease:
        with self._lock:
            lease = self._nodes[node_id]
            if lease.generation != int(generation):
                raise RuntimeError(
                    f"stale heartbeat generation for {node_id}: "
                    f"expected {lease.generation}, got {generation}"
                )
            now = time.monotonic()
            lease.heartbeat_at = now
            lease.expires_at = now + self.lease_seconds
            return lease

    def active_nodes(self, role: NodeRole | str | None = None) -> list[NodeLease]:
        normalized = None if role is None else (role if isinstance(role, NodeRole) else NodeRole(role))
        with self._lock:
            self.expire()
            return [
                lease
                for lease in self._nodes.values()
                if normalized is None or lease.role == normalized
            ]

    def expire(self, *, now: float | None = None) -> list[str]:
        selected_now = time.monotonic() if now is None else float(now)
        with self._lock:
            expired = [
                node_id for node_id, lease in self._nodes.items() if lease.expires_at <= selected_now
            ]
            for node_id in expired:
                del self._nodes[node_id]
            return expired
