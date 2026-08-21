from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import time
from typing import Mapping

import numpy as np
import torch

from forge_rl.runtime import (
    DoubleBufferedPolicyReplica,
    MailboxInferenceClient,
    MailboxInferenceEndpoint,
    MailboxNodeLocalInferenceService,
    PolicyRegistry,
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


def _run_mailbox_v2(
    config: M1AcceptanceConfig,
    state: Mapping[str, torch.Tensor],
) -> RuntimeReport:
    endpoints = [
        MailboxInferenceEndpoint.create(
            actor_id=actor,
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
    service = MailboxNodeLocalInferenceService(
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
    clients = [
        MailboxInferenceClient(endpoint, copy_outputs=False)
        for endpoint in endpoints
    ]
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
                timeout=15.0,
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
            for client in clients:
                client.close()
            for endpoint in endpoints:
                endpoint.close()
                endpoint.unlink()

    audit = audit_transition_identities(collected, trained)
    return RuntimeReport(
        "forge_rl_v2_persistent_mailbox",
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
    """Run the frozen v1 baseline against the synchronous mailbox v2 fast path."""

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
    v2 = _run_mailbox_v2(config, state)
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
            "v2_runtime": "persistent-synchronous-mailbox",
        },
        legacy=legacy,
        v2=v2,
        throughput_speedup=speedup,
        checksum_relative_error=checksum_error,
        learning_gate=learning_gate,
        status=status,
        reasons=reasons,
    )
