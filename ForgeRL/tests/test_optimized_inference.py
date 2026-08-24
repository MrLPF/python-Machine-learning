from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn

from forge_rl.runtime import (
    DoubleBufferedPolicyReplica,
    NodeLocalInferenceService,
    PolicyRegistry,
    SharedInferenceClient,
    SharedInferenceEndpoint,
    ThreadedNodeLocalInferenceService,
)


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)


def _infer(module: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    action = module.linear(batch["obs"])
    return {"action": action, "value": action + 1.0}


def _endpoint(actor_id: int) -> SharedInferenceEndpoint:
    return SharedInferenceEndpoint.create(
        actor_id=actor_id,
        slot_count=4,
        max_items=1,
        request_fields={"obs": ((2,), np.float32)},
        response_fields={
            "action": ((1,), np.float32),
            "value": ((1,), np.float32),
        },
    )


def test_optimized_inference_uses_one_collector_and_reuses_assembly_buffers() -> None:
    assert NodeLocalInferenceService is not ThreadedNodeLocalInferenceService
    endpoints = [_endpoint(actor_id) for actor_id in range(3)]
    source = TinyPolicy()
    with torch.no_grad():
        source.linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
    registry = PolicyRegistry(history_size=2)
    snapshot = registry.publish(source.state_dict(), version=0)
    service = NodeLocalInferenceService(
        endpoints=endpoints,
        replica=DoubleBufferedPolicyReplica(TinyPolicy, device="cpu"),
        infer_fn=_infer,
        max_batch_items=3,
        min_batch_items=3,
        max_wait_ms=10.0,
        collector_poll_ms=0.05,
        max_drain_per_endpoint=2,
    )
    clients = [SharedInferenceClient(endpoint) for endpoint in endpoints]
    service.refresh_policy(snapshot)
    service.start()

    def run_round(base: float) -> list[np.ndarray]:
        values = [
            np.asarray([[base + actor_id, base + actor_id + 1.0]], dtype=np.float32)
            for actor_id in range(3)
        ]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(client.infer, {"obs": value}, timeout=5.0)
                for client, value in zip(clients, values, strict=True)
            ]
            responses = [future.result(timeout=8.0) for future in futures]
        expected = [value @ np.asarray([[2.0], [-1.0]], dtype=np.float32) for value in values]
        for response, target in zip(responses, expected, strict=True):
            assert response.policy_version == 0
            np.testing.assert_allclose(response.outputs["action"], target)
            np.testing.assert_allclose(response.outputs["value"], target + 1.0)
        return values

    try:
        run_round(1.0)
        run_round(10.0)

        single = np.asarray([[4.0, 3.0]], dtype=np.float32)
        response = clients[0].infer({"obs": single}, timeout=5.0)
        np.testing.assert_allclose(response.outputs["action"], [[5.0]])

        metrics = service.metrics()
        optimized = service.optimization_metrics()
        assert metrics.request_messages == 7
        assert metrics.items == 7
        assert metrics.errors == 0
        assert optimized.collector_threads == 1
        assert optimized.assembly_allocations == 1
        assert optimized.reused_buffer_batches >= 1
        assert optimized.single_request_zero_copy_batches >= 1
        assert optimized.assembly_copy_bytes == 6 * 2 * np.dtype(np.float32).itemsize
    finally:
        service.stop()
        for endpoint in endpoints:
            endpoint.shutdown()
            endpoint.close()
            endpoint.unlink()
