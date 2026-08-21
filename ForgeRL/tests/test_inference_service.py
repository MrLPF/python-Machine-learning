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
)


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)


def infer_fn(module: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    action = module.linear(batch["obs"])
    return {
        "action": action,
        "value": action + 1.0,
        "log_prob": torch.zeros_like(action),
    }


def _make_endpoint(actor_id: int) -> SharedInferenceEndpoint:
    return SharedInferenceEndpoint.create(
        actor_id=actor_id,
        slot_count=4,
        max_items=4,
        request_fields={"obs": ((2,), np.float32)},
        response_fields={
            "action": ((1,), np.float32),
            "value": ((1,), np.float32),
            "log_prob": ((1,), np.float32),
        },
    )


def test_node_local_inference_batches_across_actors_and_refreshes_atomically() -> None:
    endpoints = [_make_endpoint(1), _make_endpoint(2)]
    registry = PolicyRegistry(history_size=3)
    source = TinyPolicy()
    with torch.no_grad():
        source.linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
    snapshot0 = registry.publish(source.state_dict(), version=0)

    replica = DoubleBufferedPolicyReplica(TinyPolicy, device="cpu")
    service = NodeLocalInferenceService(
        endpoints=endpoints,
        replica=replica,
        infer_fn=infer_fn,
        max_batch_items=8,
        min_batch_items=4,
        max_wait_ms=50,
    )
    service.refresh_policy(snapshot0)
    service.start()
    clients = [SharedInferenceClient(endpoint) for endpoint in endpoints]
    inputs = [
        np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
        np.asarray([[5.0, 2.0], [3.0, 1.0]], dtype=np.float32),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(client.infer, {"obs": value}, min_policy_version=0, timeout=5)
            for client, value in zip(clients, inputs)
        ]
        responses = [future.result(timeout=8) for future in futures]

    expected = [value @ np.asarray([[2.0], [-1.0]], dtype=np.float32) for value in inputs]
    for response, target in zip(responses, expected):
        assert response.policy_version == 0
        np.testing.assert_allclose(response.outputs["action"], target)
        np.testing.assert_allclose(response.outputs["value"], target + 1.0)

    metrics = service.metrics()
    assert metrics.batches == 1
    assert metrics.request_messages == 2
    assert metrics.items == 4
    assert metrics.mean_batch_items == 4
    assert metrics.errors == 0

    with torch.no_grad():
        source.linear.weight.copy_(torch.tensor([[1.0, 1.0]]))
    snapshot1 = registry.publish(source.state_dict(), version=1)
    assert service.refresh_policy(snapshot1) == 1
    response = clients[0].infer({"obs": inputs[0]}, min_policy_version=1, timeout=5)
    assert response.policy_version == 1
    np.testing.assert_allclose(response.outputs["action"], inputs[0].sum(axis=1, keepdims=True))

    service.stop()
    for endpoint in endpoints:
        endpoint.shutdown()
        endpoint.close()
        endpoint.unlink()
