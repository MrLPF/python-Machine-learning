from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
import os
import platform
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch

from forge_rl.runtime import (
    DoubleBufferedPolicyReplica,
    NodeLocalInferenceService,
    PolicyRegistry,
    SharedInferenceClient,
    SharedInferenceEndpoint,
)

from .audit import TransitionAudit, audit_transition_identities
from .legacy_v1 import (
    LEGACY_V1_ARCHIVE_SHA256,
    LegacyV1InferenceRuntime,
    SyntheticPolicy,
    run_synthetic_policy,
)


@dataclass(frozen=True, slots=True)
class M1AcceptanceConfig:
    actors: int = 4
    requests_per_actor: int = 250
    items_per_request: int = 8
    width: int = 64
    output_size: int = 4
    max_batch_items: int = 128
    v2_min_batch_items: int = 32
    legacy_min_batch_items: int = 1
    max_wait_ms: float = 2.0
    request_slots: int = 8
    model_seed: int = 17
    throughput_gate: float = 2.0
    checksum_tolerance: float = 1e-6
    device: str = "cpu"
    amp_dtype: str | None = None

    def validate(self) -> None:
        values = [
            self.actors,
            self.requests_per_actor,
            self.items_per_request,
            self.width,
            self.output_size,
            self.max_batch_items,
            self.request_slots,
        ]
        if any(value <= 0 for value in values):
            raise ValueError("benchmark sizes must be positive")
        if self.items_per_request > self.max_batch_items:
            raise ValueError("request exceeds max batch")
        if not 1 <= self.v2_min_batch_items <= self.max_batch_items:
            raise ValueError("invalid v2 minimum batch")
        if not 1 <= self.legacy_min_batch_items <= self.max_batch_items:
            raise ValueError("invalid legacy minimum batch")
        if self.throughput_gate < 0:
            raise ValueError("throughput gate cannot be negative")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA acceptance requested but CUDA is unavailable")
        if self.amp_dtype not in {None, "float16", "bfloat16"}:
            raise ValueError("amp_dtype must be float16, bfloat16 or None")
        if self.device != "cuda" and self.amp_dtype is not None:
            raise ValueError("AMP acceptance is supported only on CUDA")


@dataclass(frozen=True, slots=True)
class RuntimeReport:
    name: str
    elapsed_seconds: float
    requests: int
    items: int
    valid_rows_per_second: float
    batches: int
    mean_batch_items: float
    batch_fill_ratio: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    errors: int
    serialized_queue_bytes: int
    shared_memory_tensor_bytes: int
    output_checksum: float
    audit: TransitionAudit

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["audit"] = self.audit.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class LearningResult:
    environment: str
    seeds: int
    v1_time_to_target_median_seconds: float
    v2_time_to_target_median_seconds: float
    v1_normalized_auc: float
    v2_normalized_auc: float

    @property
    def time_ratio(self) -> float:
        return self.v2_time_to_target_median_seconds / self.v1_time_to_target_median_seconds

    @property
    def auc_ratio(self) -> float:
        if self.v1_normalized_auc == 0:
            return 1.0 if self.v2_normalized_auc >= 0 else 0.0
        return self.v2_normalized_auc / self.v1_normalized_auc

    @property
    def passed(self) -> bool:
        return self.seeds >= 5 and self.time_ratio <= 1.05 and self.auc_ratio >= 0.95


@dataclass(frozen=True, slots=True)
class LearningGate:
    environments: tuple[LearningResult, ...]

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "LearningGate":
        with open(path, encoding="utf-8") as handle:
            rows = json.load(handle).get("environments")
        if not isinstance(rows, list) or not rows:
            raise ValueError("learning report requires environments")
        result = cls(tuple(LearningResult(**row) for row in rows))
        names = {row.environment for row in result.environments}
        if not {"CartPole-v1", "Pendulum-v1"}.issubset(names):
            raise ValueError("learning report requires CartPole-v1 and Pendulum-v1")
        if any(
            row.v1_time_to_target_median_seconds <= 0
            or row.v2_time_to_target_median_seconds <= 0
            for row in result.environments
        ):
            raise ValueError("time-to-target values must be positive")
        return result

    @property
    def passed(self) -> bool:
        return all(row.passed for row in self.environments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "environments": [
                {
                    **asdict(row),
                    "time_ratio": row.time_ratio,
                    "auc_ratio": row.auc_ratio,
                    "passed": row.passed,
                }
                for row in self.environments
            ],
        }


@dataclass(frozen=True, slots=True)
class M1AcceptanceReport:
    config: M1AcceptanceConfig
    environment: dict[str, Any]
    legacy: RuntimeReport
    v2: RuntimeReport
    throughput_speedup: float
    checksum_relative_error: float
    learning_gate: LearningGate | None
    status: str
    reasons: tuple[str, ...]

    @property
    def go(self) -> bool:
        return self.status == "GO"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "config": asdict(self.config),
            "environment": self.environment,
            "legacy": self.legacy.to_dict(),
            "v2": self.v2.to_dict(),
            "throughput_speedup": self.throughput_speedup,
            "checksum_relative_error": self.checksum_relative_error,
            "learning_gate": None if self.learning_gate is None else self.learning_gate.to_dict(),
            "status": self.status,
            "go": self.go,
            "reasons": list(self.reasons),
        }


def _observation(config: M1AcceptanceConfig, actor: int, sequence: int) -> np.ndarray:
    columns = np.arange(config.width, dtype=np.float32)[None]
    rows = np.arange(config.items_per_request, dtype=np.float32)[:, None]
    return np.sin(
        columns * 0.013 + rows * 0.071 + actor * 0.19 + sequence * 0.003
    ).astype(np.float32)


def _identities(
    config: M1AcceptanceConfig,
    actor: int,
    sequence: int,
) -> list[tuple[int, int, int, int]]:
    start = sequence * config.items_per_request
    return [(actor, 0, 0, start + offset) for offset in range(config.items_per_request)]


def _checksum(outputs: Mapping[str, np.ndarray]) -> float:
    return float(sum(np.asarray(value, np.float64).sum() for value in outputs.values()))


def _fingerprint() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "github_sha": os.environ.get("GITHUB_SHA", ""),
        "legacy_v1_archive_sha256": LEGACY_V1_ARCHIVE_SHA256,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def _run_legacy(config: M1AcceptanceConfig, state: Mapping[str, torch.Tensor]) -> RuntimeReport:
    runtime = LegacyV1InferenceRuntime(
        actor_count=config.actors,
        width=config.width,
        output_size=config.output_size,
        state_dict=state,
        max_batch_items=config.max_batch_items,
        min_batch_items=config.legacy_min_batch_items,
        max_wait_ms=config.max_wait_ms,
        device=config.device,
        amp_dtype=None if config.amp_dtype is None else getattr(torch, config.amp_dtype),
    )
    runtime.start()
    collected: list[tuple[int, int, int, int]] = []
    trained: list[tuple[int, int, int, int]] = []
    checksums = [0.0] * config.actors

    def actor_loop(actor: int) -> None:
        total = 0.0
        for sequence in range(config.requests_per_actor):
            collected.extend(_identities(config, actor, sequence))
            outputs, _, returned_sequence = runtime.infer(
                actor,
                sequence,
                _observation(config, actor, sequence),
                min_policy_version=0,
            )
            trained.extend(_identities(config, actor, returned_sequence))
            total += _checksum(outputs)
        checksums[actor] = total

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=config.actors) as pool:
            list(pool.map(actor_loop, range(config.actors)))
        elapsed = time.perf_counter() - started
        metrics = runtime.metrics(config.actors * config.requests_per_actor)
    finally:
        runtime.close()
    audit = audit_transition_identities(collected, trained)
    return RuntimeReport(
        "forge_rl_v1_frozen_queue",
        elapsed,
        config.actors * config.requests_per_actor,
        config.actors * config.requests_per_actor * config.items_per_request,
        audit.unique_trained_rows / elapsed,
        metrics.batches,
        metrics.mean_batch_items,
        metrics.batch_fill_ratio,
        metrics.latency_p50_ms,
        metrics.latency_p95_ms,
        metrics.latency_p99_ms,
        metrics.errors,
        metrics.serialized_request_bytes + metrics.serialized_response_bytes,
        0,
        float(math.fsum(checksums)),
        audit,
    )


def _run_v2(config: M1AcceptanceConfig, state: Mapping[str, torch.Tensor]) -> RuntimeReport:
    endpoints = [
        SharedInferenceEndpoint.create(
            actor_id=actor,
            slot_count=config.request_slots,
            max_items=config.items_per_request,
            request_fields={"obs": ((config.width,), np.float32)},
            response_fields={
                "action": ((config.output_size,), np.float32),
                "value": ((1,), np.float32),
            },
        )
        for actor in range(config.actors)
    ]

    def factory() -> SyntheticPolicy:
        return SyntheticPolicy(config.width, config.output_size)

    registry = PolicyRegistry(history_size=2)
    snapshot = registry.publish(state, version=0)
    service = NodeLocalInferenceService(
        endpoints=endpoints,
        replica=DoubleBufferedPolicyReplica(
            factory,
            device=config.device,
            amp_dtype=None if config.amp_dtype is None else getattr(torch, config.amp_dtype),
        ),
        infer_fn=run_synthetic_policy,
        max_batch_items=config.max_batch_items,
        min_batch_items=config.v2_min_batch_items,
        max_wait_ms=config.max_wait_ms,
    )
    service.refresh_policy(snapshot)
    service.start()
    clients = [SharedInferenceClient(endpoint) for endpoint in endpoints]
    collected: list[tuple[int, int, int, int]] = []
    trained: list[tuple[int, int, int, int]] = []
    checksums = [0.0] * config.actors
    tensor_bytes = [0] * config.actors

    def actor_loop(actor: int) -> None:
        total = 0.0
        byte_count = 0
        for sequence in range(config.requests_per_actor):
            observation = _observation(config, actor, sequence)
            collected.extend(_identities(config, actor, sequence))
            response = clients[actor].infer(
                {"obs": observation},
                min_policy_version=0,
                timeout=15,
            )
            trained.extend(_identities(config, actor, response.sequence_id))
            total += _checksum(response.outputs)
            byte_count += observation.nbytes + sum(
                value.nbytes for value in response.outputs.values()
            )
        checksums[actor] = total
        tensor_bytes[actor] = byte_count

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=config.actors) as pool:
            list(pool.map(actor_loop, range(config.actors)))
        elapsed = time.perf_counter() - started
        metrics = service.metrics()
        if service.last_error is not None:
            raise RuntimeError(service.last_error)
    finally:
        try:
            service.stop()
        finally:
            for endpoint in endpoints:
                endpoint.shutdown()
                endpoint.close()
                endpoint.unlink()
    audit = audit_transition_identities(collected, trained)
    return RuntimeReport(
        "forge_rl_v2_shared_memory",
        elapsed,
        config.actors * config.requests_per_actor,
        config.actors * config.requests_per_actor * config.items_per_request,
        audit.unique_trained_rows / elapsed,
        metrics.batches,
        metrics.mean_batch_items,
        metrics.batch_fill_ratio,
        metrics.latency_p50_ms,
        metrics.latency_p95_ms,
        metrics.latency_p99_ms,
        metrics.errors,
        0,
        sum(tensor_bytes),
        float(math.fsum(checksums)),
        audit,
    )


def run_m1_acceptance(
    config: M1AcceptanceConfig,
    learning_gate: LearningGate | None = None,
) -> M1AcceptanceReport:
    config.validate()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(config.model_seed)
    model = SyntheticPolicy(config.width, config.output_size)
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    legacy = _run_legacy(config, state)
    v2 = _run_v2(config, state)
    speedup = v2.valid_rows_per_second / legacy.valid_rows_per_second
    scale = max(1.0, abs(legacy.output_checksum), abs(v2.output_checksum))
    checksum_error = abs(v2.output_checksum - legacy.output_checksum) / scale
    if not legacy.audit.passed or not v2.audit.passed or legacy.errors or v2.errors:
        status, reasons = "FAIL_CORRECTNESS", (
            "transition audit or runtime error gate failed",
        )
    elif checksum_error > config.checksum_tolerance:
        status, reasons = "FAIL_NUMERICAL_EQUIVALENCE", (
            "v1/v2 output checksum mismatch",
        )
    elif speedup < config.throughput_gate:
        status, reasons = "FAIL_SYNTHETIC_THROUGHPUT", (
            "valid-row speedup is below the configured gate",
        )
    elif learning_gate is None:
        status, reasons = "PENDING_LEARNING_GATE", (
            "five-seed CartPole/Pendulum report is absent",
        )
    elif not learning_gate.passed:
        status, reasons = "FAIL_LEARNING_GATE", (
            "time-to-target or normalized AUC gate failed",
        )
    else:
        status, reasons = "GO", ("all M1 gates passed",)
    return M1AcceptanceReport(
        config,
        _fingerprint(),
        legacy,
        v2,
        speedup,
        checksum_error,
        learning_gate,
        status,
        reasons,
    )
