from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import time
from typing import Callable, Mapping

import numpy as np
import torch

from forge_rl.runtime import (
    ActorLocalInferenceRuntime,
    InProcessBatchingInferenceRuntime,
    ProcessMailboxInferenceRuntime,
    VectorBatchInferenceRuntime,
)

from .audit import audit_transition_identities
from .legacy_v1 import SyntheticPolicy, run_synthetic_policy
from .m1_acceptance import (
    LearningGate,
    M1AcceptanceConfig,
    M1AcceptanceReport,
    RuntimeReport,
    _checksum,
    _fingerprint,
    _identities,
    _observation,
    _run_legacy,
)


def _run_requests(
    config: M1AcceptanceConfig,
    infer: Callable[[int, np.ndarray], tuple[Mapping[str, np.ndarray], int, int]],
) -> tuple[
    float,
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
    list[float],
    list[int],
]:
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
            outputs, _version, returned = infer(actor, observation)
            trained.extend(_identities(config, actor, returned))
            total += _checksum(outputs)
            byte_count += observation.nbytes + sum(
                value.nbytes for value in outputs.values()
            )
        checksums[actor] = total
        tensor_bytes[actor] = byte_count

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.actors) as pool:
        list(pool.map(actor_loop, range(config.actors)))
    return (
        time.perf_counter() - started,
        collected,
        trained,
        checksums,
        tensor_bytes,
    )


def _report(
    *,
    name: str,
    config: M1AcceptanceConfig,
    elapsed: float,
    collected: list[tuple[int, int, int, int]],
    trained: list[tuple[int, int, int, int]],
    checksums: list[float],
    tensor_bytes: list[int],
    metrics,
) -> RuntimeReport:
    audit = audit_transition_identities(collected, trained)
    return RuntimeReport(
        name,
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


def _run_actor_local_v2(
    config: M1AcceptanceConfig,
    state: Mapping[str, torch.Tensor],
) -> RuntimeReport:
    runtime = ActorLocalInferenceRuntime(
        actor_count=config.actors,
        module_type=SyntheticPolicy,
        module_args=(config.width, config.output_size),
        infer_fn=run_synthetic_policy,
        state_dict=state,
        policy_version=0,
        max_batch_items=config.items_per_request,
        copy_outputs=False,
    )
    runtime.start()

    def infer(
        actor: int, observation: np.ndarray
    ) -> tuple[Mapping[str, np.ndarray], int, int]:
        response = runtime.infer(
            actor,
            {"obs": observation},
            min_policy_version=0,
        )
        return response.outputs, response.policy_version, response.sequence_id

    try:
        elapsed, collected, trained, checksums, tensor_bytes = _run_requests(
            config, infer
        )
        metrics = runtime.metrics()
    finally:
        runtime.close()
    return _report(
        name="forge_rl_v2_actor_local_cpu",
        config=config,
        elapsed=elapsed,
        collected=collected,
        trained=trained,
        checksums=checksums,
        tensor_bytes=tensor_bytes,
        metrics=metrics,
    )


def _run_in_process_v2(
    config: M1AcceptanceConfig,
    state: Mapping[str, torch.Tensor],
) -> RuntimeReport:
    runtime = InProcessBatchingInferenceRuntime(
        actor_count=config.actors,
        module_type=SyntheticPolicy,
        module_args=(config.width, config.output_size),
        infer_fn=run_synthetic_policy,
        state_dict=state,
        policy_version=0,
        max_batch_items=config.max_batch_items,
        min_batch_items=min(
            config.v2_min_batch_items,
            config.actors * config.items_per_request,
            config.max_batch_items,
        ),
        max_wait_ms=config.max_wait_ms,
        device=config.device,
        amp_dtype=(
            None if config.amp_dtype is None else getattr(torch, config.amp_dtype)
        ),
        copy_outputs=False,
    )
    runtime.start()

    def infer(
        actor: int, observation: np.ndarray
    ) -> tuple[Mapping[str, np.ndarray], int, int]:
        response = runtime.infer(
            actor,
            {"obs": observation},
            min_policy_version=0,
            timeout=15.0,
        )
        return response.outputs, response.policy_version, response.sequence_id

    try:
        elapsed, collected, trained, checksums, tensor_bytes = _run_requests(
            config, infer
        )
        metrics = runtime.metrics()
        if runtime.last_error is not None:
            raise RuntimeError(runtime.last_error)
    finally:
        runtime.close()
    return _report(
        name="forge_rl_v2_in_process_batch",
        config=config,
        elapsed=elapsed,
        collected=collected,
        trained=trained,
        checksums=checksums,
        tensor_bytes=tensor_bytes,
        metrics=metrics,
    )


def _run_vector_batch_v2(
    config: M1AcceptanceConfig,
    state: Mapping[str, torch.Tensor],
) -> RuntimeReport:
    total_items = config.actors * config.items_per_request
    if total_items > config.max_batch_items:
        raise ValueError(
            "vector-batch acceptance requires actors * items_per_request <= max_batch_items"
        )
    runtime = VectorBatchInferenceRuntime(
        actor_count=config.actors,
        module_type=SyntheticPolicy,
        module_args=(config.width, config.output_size),
        infer_fn=run_synthetic_policy,
        state_dict=state,
        policy_version=0,
        max_batch_items=config.max_batch_items,
        device=config.device,
        amp_dtype=(
            None if config.amp_dtype is None else getattr(torch, config.amp_dtype)
        ),
        copy_outputs=False,
    )
    runtime.start()
    collected: list[tuple[int, int, int, int]] = []
    trained: list[tuple[int, int, int, int]] = []
    checksums = [0.0] * config.actors
    tensor_bytes = [0] * config.actors
    actor_counts = {
        actor: config.items_per_request
        for actor in range(config.actors)
    }

    started = time.perf_counter()
    try:
        for sequence in range(config.requests_per_actor):
            batch = np.empty((total_items, config.width), dtype=np.float32)
            observations: list[np.ndarray] = []
            offset = 0
            for actor in range(config.actors):
                observation = _observation(config, actor, sequence)
                observations.append(observation)
                stop = offset + config.items_per_request
                np.copyto(batch[offset:stop], observation, casting="no")
                collected.extend(_identities(config, actor, sequence))
                offset = stop
            responses = runtime.infer_batch(
                {"obs": batch},
                actor_item_counts=actor_counts,
                min_policy_version=0,
            )
            for actor, observation in enumerate(observations):
                response = responses[actor]
                if response.sequence_id != sequence:
                    raise RuntimeError(
                        f"vector-batch response sequence {response.sequence_id} != {sequence}"
                    )
                trained.extend(_identities(config, actor, response.sequence_id))
                checksums[actor] += _checksum(response.outputs)
                tensor_bytes[actor] += observation.nbytes + sum(
                    value.nbytes for value in response.outputs.values()
                )
        elapsed = time.perf_counter() - started
        metrics = runtime.metrics()
        if runtime.last_error is not None:
            raise RuntimeError(runtime.last_error)
    finally:
        runtime.close()

    return _report(
        name="forge_rl_v2_vector_batch",
        config=config,
        elapsed=elapsed,
        collected=collected,
        trained=trained,
        checksums=checksums,
        tensor_bytes=tensor_bytes,
        metrics=metrics,
    )


def _run_process_mailbox_v2(
    config: M1AcceptanceConfig,
    state: Mapping[str, torch.Tensor],
) -> RuntimeReport:
    runtime = ProcessMailboxInferenceRuntime(
        actor_count=config.actors,
        max_items=config.items_per_request,
        request_fields={"obs": ((config.width,), np.float32)},
        response_fields={
            "action": ((config.output_size,), np.float32),
            "value": ((1,), np.float32),
        },
        module_type=SyntheticPolicy,
        module_args=(config.width, config.output_size),
        infer_fn=run_synthetic_policy,
        state_dict=state,
        policy_version=0,
        max_batch_items=config.max_batch_items,
        min_batch_items=config.v2_min_batch_items,
        max_wait_ms=config.max_wait_ms,
        device=config.device,
        amp_dtype=(
            None if config.amp_dtype is None else getattr(torch, config.amp_dtype)
        ),
        copy_outputs=False,
    )
    runtime.start()

    def infer(
        actor: int, observation: np.ndarray
    ) -> tuple[Mapping[str, np.ndarray], int, int]:
        response = runtime.infer(
            actor,
            {"obs": observation},
            min_policy_version=0,
            timeout=15.0,
        )
        return response.outputs, response.policy_version, response.sequence_id

    try:
        elapsed, collected, trained, checksums, tensor_bytes = _run_requests(
            config, infer
        )
        metrics, _transport = runtime.metrics_pair()
    finally:
        runtime.close()
    return _report(
        name="forge_rl_v2_process_mailbox",
        config=config,
        elapsed=elapsed,
        collected=collected,
        trained=trained,
        checksums=checksums,
        tensor_bytes=tensor_bytes,
        metrics=metrics,
    )


def _evaluate(
    config: M1AcceptanceConfig,
    learning_gate: LearningGate | None,
    *,
    runtime_name: str,
    runner: Callable[
        [M1AcceptanceConfig, Mapping[str, torch.Tensor]], RuntimeReport
    ],
) -> M1AcceptanceReport:
    config.validate()
    if runtime_name == "actor-local" and config.device != "cpu":
        raise ValueError("actor-local M1 runtime supports CPU only")
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
    v2 = runner(config, state)
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
            "time-to-target, target-reach or normalized-AUC gate failed",
        )
    else:
        status, reasons = "GO", ()

    return M1AcceptanceReport(
        config=config,
        environment={
            **_fingerprint(),
            "v2_runtime": runtime_name,
            "topology_selection": (
                "vectorized EnvRunners should submit one contiguous batch; threaded Actors may "
                "use arrival-aware in-process batching; separate processes use mailbox transport"
            ),
        },
        legacy=legacy,
        v2=v2,
        throughput_speedup=speedup,
        checksum_relative_error=checksum_error,
        learning_gate=learning_gate,
        status=status,
        reasons=reasons,
    )


def run_actor_local_m1_acceptance(
    config: M1AcceptanceConfig,
    learning_gate: LearningGate | None = None,
) -> M1AcceptanceReport:
    return _evaluate(
        config,
        learning_gate,
        runtime_name="actor-local",
        runner=_run_actor_local_v2,
    )


def run_in_process_m1_acceptance(
    config: M1AcceptanceConfig,
    learning_gate: LearningGate | None = None,
) -> M1AcceptanceReport:
    return _evaluate(
        config,
        learning_gate,
        runtime_name="in-process-batch",
        runner=_run_in_process_v2,
    )


def run_vector_batch_m1_acceptance(
    config: M1AcceptanceConfig,
    learning_gate: LearningGate | None = None,
) -> M1AcceptanceReport:
    return _evaluate(
        config,
        learning_gate,
        runtime_name="vector-batch",
        runner=_run_vector_batch_v2,
    )


def run_process_mailbox_m1_acceptance(
    config: M1AcceptanceConfig,
    learning_gate: LearningGate | None = None,
) -> M1AcceptanceReport:
    return _evaluate(
        config,
        learning_gate,
        runtime_name="process-mailbox",
        runner=_run_process_mailbox_v2,
    )


def run_m1_acceptance(
    config: M1AcceptanceConfig,
    learning_gate: LearningGate | None = None,
    *,
    runtime_mode: str = "process-mailbox",
) -> M1AcceptanceReport:
    selected = runtime_mode
    if selected == "auto":
        selected = "in-process-batch" if config.device == "cpu" else "process-mailbox"
    if selected == "actor-local":
        return run_actor_local_m1_acceptance(config, learning_gate)
    if selected == "in-process-batch":
        return run_in_process_m1_acceptance(config, learning_gate)
    if selected == "vector-batch":
        return run_vector_batch_m1_acceptance(config, learning_gate)
    if selected == "process-mailbox":
        return run_process_mailbox_m1_acceptance(config, learning_gate)
    raise ValueError(
        "runtime_mode must be auto, actor-local, in-process-batch, vector-batch or "
        "process-mailbox"
    )
