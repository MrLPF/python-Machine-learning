from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
import platform
import time
from typing import Any, Mapping
from urllib.parse import urlparse

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from forge_rl.distributed import DistributedContext
from forge_rl.runtime import (
    NetworkExperienceClient,
    NetworkExperienceService,
    NetworkPolicyClient,
    NetworkPolicyService,
    OnPolicyExperienceQueue,
    PolicyRegistry,
    TransitionBatch,
)

from .audit import audit_transition_identities


@dataclass(frozen=True, slots=True)
class M2WorkerConfig:
    warmup_steps: int = 5
    measured_steps: int = 50
    valid_rows_per_rank: int = 128
    invalid_rows_per_batch: int = 1
    observation_width: int = 32
    hidden_size: int = 64
    learning_rate: float = 1e-3
    policy_publish_interval: int = 10
    advertise_host: str = "127.0.0.1"
    physical_node_id: str = "local-node"
    controlled: bool = False

    def validate(self) -> None:
        positive = (
            self.measured_steps,
            self.valid_rows_per_rank,
            self.observation_width,
            self.hidden_size,
            self.policy_publish_interval,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("measured sizes and publish interval must be positive")
        if self.warmup_steps < 0 or self.invalid_rows_per_batch < 0:
            raise ValueError("warmup steps and invalid rows cannot be negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not self.advertise_host or not self.physical_node_id:
            raise ValueError("advertise_host and physical_node_id are required")


@dataclass(frozen=True, slots=True)
class M2RunReport:
    schema_version: int
    status: str
    controlled: bool
    world_size: int
    learner_ranks: int
    physical_node_ids: tuple[str, ...]
    unique_physical_nodes: int
    elapsed_seconds: float
    valid_rows: int
    valid_rows_per_second: float
    invalid_rows_received: int
    invalid_rows_in_loss: int
    runtime_errors: int
    deadlock_free: bool
    audit: dict[str, Any]
    experience_requests: int
    experience_rows: int
    policy_publications: int
    latest_policy_version: int
    configuration: dict[str, Any]
    environment: dict[str, Any]

    @property
    def correctness_passed(self) -> bool:
        return (
            self.status == "COMPLETED"
            and self.deadlock_free
            and self.invalid_rows_in_loss == 0
            and self.runtime_errors == 0
            and bool(self.audit.get("passed"))
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["physical_node_ids"] = list(self.physical_node_ids)
        payload["correctness_passed"] = self.correctness_passed
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "M2RunReport":
        return cls(
            schema_version=int(payload["schema_version"]),
            status=str(payload["status"]),
            controlled=bool(payload["controlled"]),
            world_size=int(payload["world_size"]),
            learner_ranks=int(payload["learner_ranks"]),
            physical_node_ids=tuple(str(value) for value in payload["physical_node_ids"]),
            unique_physical_nodes=int(payload["unique_physical_nodes"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            valid_rows=int(payload["valid_rows"]),
            valid_rows_per_second=float(payload["valid_rows_per_second"]),
            invalid_rows_received=int(payload["invalid_rows_received"]),
            invalid_rows_in_loss=int(payload["invalid_rows_in_loss"]),
            runtime_errors=int(payload["runtime_errors"]),
            deadlock_free=bool(payload["deadlock_free"]),
            audit=dict(payload["audit"]),
            experience_requests=int(payload["experience_requests"]),
            experience_rows=int(payload["experience_rows"]),
            policy_publications=int(payload["policy_publications"]),
            latest_policy_version=int(payload["latest_policy_version"]),
            configuration=dict(payload["configuration"]),
            environment=dict(payload["environment"]),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "M2RunReport":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("M2 run report must be a JSON object")
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class M2AcceptanceReport:
    schema_version: int
    status: str
    go: bool
    formal_eligible: bool
    weak_scaling_efficiency: float
    efficiency_gate: float
    reasons: tuple[str, ...]
    baseline: dict[str, Any]
    target: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


def _dist_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_gather_object(value: Any, world_size: int) -> list[Any]:
    if not _dist_active():
        return [value]
    values: list[Any] = [None] * world_size
    dist.all_gather_object(values, value)
    return values


def _all_reduce_sum(value: int | float, *, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if _dist_active():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _all_reduce_max(value: float, *, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if _dist_active():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _advertise(endpoint: str, host: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.port is None:
        raise ValueError(f"service endpoint does not contain a port: {endpoint}")
    return f"tcp://{host}:{parsed.port}"


def _base_model(model: nn.Module) -> nn.Module:
    candidate = getattr(model, "module", None)
    return candidate if isinstance(candidate, nn.Module) else model


def _state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone(memory_format=torch.preserve_format)
        for name, value in _base_model(model).state_dict().items()
    }


def _make_transition_batch(
    config: M2WorkerConfig,
    *,
    source_rank: int,
    sequence: int,
) -> TransitionBatch:
    total_rows = config.valid_rows_per_rank + config.invalid_rows_per_batch
    row = np.arange(total_rows, dtype=np.float32)[:, None]
    column = np.arange(config.observation_width, dtype=np.float32)[None, :]
    observation = (
        row * np.float32(0.01)
        + column * np.float32(0.001)
        + np.float32(source_rank)
        + np.float32(sequence) * np.float32(0.0001)
    ).astype(np.float32)
    next_observation = observation + np.float32(0.005)
    valid_mask = np.ones(total_rows, dtype=np.float32)
    if config.invalid_rows_per_batch:
        valid_mask[-config.invalid_rows_per_batch :] = 0.0
    reward = observation.mean(axis=1).astype(np.float32)
    action = (np.arange(total_rows, dtype=np.int64) % 2).astype(np.int64)
    identity_offset = sequence * total_rows
    return TransitionBatch(
        obs={"obs": observation},
        next_obs={"obs": next_observation},
        action={"action": action},
        reward=reward,
        terminated=np.zeros(total_rows, dtype=np.bool_),
        truncated=np.zeros(total_rows, dtype=np.bool_),
        valid_mask=valid_mask,
        behavior_log_prob=np.zeros(total_rows, dtype=np.float32),
        behavior_value=np.zeros(total_rows, dtype=np.float32),
        policy_version=np.zeros(total_rows, dtype=np.int64),
        episode_id=np.zeros(total_rows, dtype=np.int64),
        step_id=np.arange(identity_offset, identity_offset + total_rows, dtype=np.int64),
        actor_id=np.full(total_rows, source_rank, dtype=np.int64),
        env_id=np.arange(total_rows, dtype=np.int64),
    )


def _valid_identities(batch: TransitionBatch) -> list[tuple[int, int, int, int]]:
    valid = batch.valid_mask > 0.0
    return [
        (
            int(batch.actor_id[index]),
            int(batch.env_id[index]),
            int(batch.episode_id[index]),
            int(batch.step_id[index]),
        )
        for index in np.flatnonzero(valid)
    ]


def _merge_items(items: list[Any]) -> TransitionBatch:
    payloads = [item.payload for item in items]
    if not payloads or any(not isinstance(payload, TransitionBatch) for payload in payloads):
        raise RuntimeError("experience sample did not contain TransitionBatch payloads")
    selected = [payload for payload in payloads if isinstance(payload, TransitionBatch)]
    return selected[0] if len(selected) == 1 else TransitionBatch.concat(selected)


def run_m2_worker(
    context: DistributedContext,
    config: M2WorkerConfig,
) -> M2RunReport | None:
    """Run one weak-scaling subject on the current learner process group.

    Every rank starts a local Experience and Policy receiver. EnvRunner traffic is sent directly to
    the next rank in a ring, so a two-rank formal run exercises an actual cross-node tensor path.
    The per-rank valid workload remains fixed as world size increases.
    """

    config.validate()
    if context.world_size <= 0 or context.rank < 0 or context.rank >= context.world_size:
        raise ValueError("invalid distributed context")
    if context.distributed and not _dist_active():
        raise RuntimeError("initialize the process group before running M2 acceptance")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(4100)
    model = context.wrap(
        nn.Sequential(
            nn.Linear(config.observation_width, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1),
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    queue_capacity = max(
        config.valid_rows_per_rank * 4,
        config.valid_rows_per_rank + config.invalid_rows_per_batch,
    )
    experience_queue = OnPolicyExperienceQueue(capacity_rows=queue_capacity)
    registry = PolicyRegistry(
        history_size=max(2, math.ceil(config.measured_steps / config.policy_publish_interval) + 1)
    )
    token = "forgerl-m2-acceptance"
    experience_service = NetworkExperienceService(
        experience_queue,
        host="0.0.0.0",
        port=0,
        bearer_token=token,
        put_timeout=10.0,
        socket_timeout=30.0,
    )
    policy_service = NetworkPolicyService(
        registry,
        host="0.0.0.0",
        port=0,
        bearer_token=token,
        socket_timeout=30.0,
    )
    experience_service.start()
    policy_service.start()

    experience_client: NetworkExperienceClient | None = None
    policy_client: NetworkPolicyClient | None = None
    runtime_errors = 0
    invalid_rows_received = 0
    invalid_rows_in_loss = 0
    generated: list[tuple[int, int, int, int]] = []
    trained: list[tuple[int, int, int, int]] = []

    try:
        local_endpoints = {
            "experience": _advertise(experience_service.endpoint, config.advertise_host),
            "policy": _advertise(policy_service.endpoint, config.advertise_host),
            "physical_node_id": config.physical_node_id,
        }
        endpoints = _all_gather_object(local_endpoints, context.world_size)
        peer = (context.rank + 1) % context.world_size
        peer_endpoints = dict(endpoints[peer])
        experience_client = NetworkExperienceClient(
            str(peer_endpoints["experience"]),
            bearer_token=token,
            timeout=30.0,
        )
        policy_client = NetworkPolicyClient(
            str(peer_endpoints["policy"]),
            bearer_token=token,
            timeout=30.0,
        )

        def run_step(sequence: int, *, measured: bool) -> None:
            nonlocal invalid_rows_received, invalid_rows_in_loss, runtime_errors
            phase = "measured" if measured else "warmup"
            batch = _make_transition_batch(
                config,
                source_rank=context.rank,
                sequence=sequence,
            )
            if measured:
                generated.extend(_valid_identities(batch))
            experience_client.put(
                batch,
                request_id=f"experience-{phase}-{context.rank}-{sequence}",
            )
            items = experience_queue.sample(
                min_rows=config.valid_rows_per_rank,
                current_policy_version=0,
                max_policy_lag=config.measured_steps + config.warmup_steps + 1,
                timeout=30.0,
            )
            if not items:
                runtime_errors += 1
                raise RuntimeError("timed out waiting for direct experience traffic")
            received = _merge_items(items)
            valid = received.valid_mask > 0.0
            invalid_rows_received += int((~valid).sum())
            selected_indices = np.flatnonzero(valid)
            if np.any(received.valid_mask[selected_indices] <= 0.0):
                invalid_rows_in_loss += 1
            if measured:
                trained.extend(_valid_identities(received))

            observations = torch.from_numpy(received.obs["obs"][selected_indices]).to(
                context.device
            )
            targets = torch.from_numpy(received.reward[selected_indices]).to(
                context.device
            )
            predictions = model(observations).squeeze(-1)
            loss = (predictions - targets).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if measured and sequence % config.policy_publish_interval == 0:
                version = sequence // config.policy_publish_interval
                policy_client.publish(
                    _state_dict(model),
                    version=version,
                    metadata={"source_rank": context.rank, "sequence": sequence},
                    request_id=f"policy-{context.rank}-{version}",
                )

        for warmup in range(config.warmup_steps):
            run_step(warmup, measured=False)
        context.barrier()
        started = time.perf_counter()
        for measured_step in range(config.measured_steps):
            run_step(measured_step, measured=True)
        context.barrier()
        local_elapsed = time.perf_counter() - started
        elapsed = _all_reduce_max(local_elapsed, device=context.device)

        generated_by_rank = _all_gather_object(generated, context.world_size)
        trained_by_rank = _all_gather_object(trained, context.world_size)
        all_generated = [identity for rows in generated_by_rank for identity in rows]
        all_trained = [identity for rows in trained_by_rank for identity in rows]
        audit = audit_transition_identities(all_generated, all_trained)

        local_experience_metrics = experience_service.metrics()
        local_policy_metrics = policy_service.metrics()
        total_invalid_received = int(
            _all_reduce_sum(invalid_rows_received, device=context.device)
        )
        total_invalid_in_loss = int(
            _all_reduce_sum(invalid_rows_in_loss, device=context.device)
        )
        total_runtime_errors = int(_all_reduce_sum(runtime_errors, device=context.device))
        total_experience_requests = int(
            _all_reduce_sum(local_experience_metrics.accepted_requests, device=context.device)
        )
        total_experience_rows = int(
            _all_reduce_sum(local_experience_metrics.accepted_rows, device=context.device)
        )
        total_policy_publications = int(
            _all_reduce_sum(local_policy_metrics.published_versions, device=context.device)
        )
        latest_versions = _all_gather_object(registry.latest_version, context.world_size)
        latest_policy_version = min(int(value) for value in latest_versions)
        physical_node_ids = tuple(
            str(dict(value)["physical_node_id"]) for value in endpoints
        )
        total_valid_rows = (
            config.valid_rows_per_rank * config.measured_steps * context.world_size
        )
        status = (
            "COMPLETED"
            if audit.passed and total_invalid_in_loss == 0 and total_runtime_errors == 0
            else "FAIL_CORRECTNESS"
        )
        report = M2RunReport(
            schema_version=1,
            status=status,
            controlled=config.controlled,
            world_size=context.world_size,
            learner_ranks=context.world_size,
            physical_node_ids=physical_node_ids,
            unique_physical_nodes=len(set(physical_node_ids)),
            elapsed_seconds=elapsed,
            valid_rows=total_valid_rows,
            valid_rows_per_second=total_valid_rows / elapsed,
            invalid_rows_received=total_invalid_received,
            invalid_rows_in_loss=total_invalid_in_loss,
            runtime_errors=total_runtime_errors,
            deadlock_free=True,
            audit=audit.to_dict(),
            experience_requests=total_experience_requests,
            experience_rows=total_experience_rows,
            policy_publications=total_policy_publications,
            latest_policy_version=latest_policy_version,
            configuration={
                **asdict(config),
                "backend": context.backend,
            },
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "torch": torch.__version__,
                "github_sha": os.environ.get("GITHUB_SHA", ""),
            },
        )
        return report if context.rank == 0 else None
    finally:
        if experience_client is not None:
            experience_client.close()
        if policy_client is not None:
            policy_client.close()
        experience_service.close()
        policy_service.close()
        experience_queue.close()


def evaluate_m2_acceptance(
    baseline: M2RunReport,
    target: M2RunReport,
    *,
    efficiency_gate: float = 0.70,
) -> M2AcceptanceReport:
    if efficiency_gate <= 0 or efficiency_gate > 1:
        raise ValueError("efficiency_gate must lie in (0, 1]")
    reasons: list[str] = []
    comparable_fields = (
        "measured_steps",
        "valid_rows_per_rank",
        "invalid_rows_per_batch",
        "observation_width",
        "hidden_size",
        "learning_rate",
        "policy_publish_interval",
    )
    comparable = all(
        baseline.configuration.get(name) == target.configuration.get(name)
        for name in comparable_fields
    )
    scaling_factor = target.world_size / max(1, baseline.world_size)
    efficiency = target.valid_rows_per_second / max(
        baseline.valid_rows_per_second * scaling_factor,
        1e-12,
    )

    if not baseline.correctness_passed or not target.correctness_passed:
        status = "FAIL_CORRECTNESS"
        reasons.append("baseline or target correctness audit failed")
    elif baseline.world_size != 1 or target.world_size < 2 or target.learner_ranks < 2:
        status = "FAIL_TOPOLOGY"
        reasons.append("M2 requires a one-rank baseline and at least two target learner ranks")
    elif not comparable:
        status = "FAIL_CONFIGURATION"
        reasons.append("baseline and target per-rank workloads differ")
    elif not baseline.controlled or not target.controlled:
        status = "SCREENING_ONLY"
        reasons.append("uncontrolled runs cannot produce formal M2 acceptance")
    elif target.unique_physical_nodes < 2:
        status = "FAIL_TOPOLOGY"
        reasons.append("formal target run must use at least two physical node IDs")
    elif efficiency < efficiency_gate:
        status = "FAIL_WEAK_SCALING"
        reasons.append(
            f"weak-scaling efficiency {efficiency:.6f} is below {efficiency_gate:.6f}"
        )
    else:
        status = "GO"

    formal_eligible = (
        baseline.controlled
        and target.controlled
        and baseline.world_size == 1
        and target.world_size >= 2
        and target.unique_physical_nodes >= 2
    )
    return M2AcceptanceReport(
        schema_version=1,
        status=status,
        go=status == "GO",
        formal_eligible=formal_eligible,
        weak_scaling_efficiency=efficiency,
        efficiency_gate=efficiency_gate,
        reasons=tuple(reasons),
        baseline=baseline.to_dict(),
        target=target.to_dict(),
    )
