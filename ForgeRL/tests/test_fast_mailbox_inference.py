from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn

from forge_rl.runtime import (
    DoubleBufferedPolicyReplica,
    FastMailboxInferenceClient,
    FastMailboxNodeLocalInferenceService,
    MailboxInferenceEndpoint,
    PolicyRegistry,
)


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)


def _infer(module: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    action = module.linear(batch["obs"])
    return {"action": action, "value": action + 1.0}


def _endpoint(actor_id: int) -> MailboxInferenceEndpoint:
    return MailboxInferenceEndpoint.create(
        actor_id=actor_id,
        max_items=2,
        request_fields={"obs": ((2,), np.float32)},
        response_fields={
            "action": ((1,), np.float32),
            "value": ((1,), np.float32),
        },
    )


def test_fast_mailbox_reuses_views_and_keeps_actors_batch_aligned() -> None:
    endpoints = [_endpoint(actor_id) for actor_id in range(3)]
    source = TinyPolicy()
    with torch.no_grad():
        source.linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
    registry = PolicyRegistry(history_size=2)
    snapshot = registry.publish(source.state_dict(), version=0)
    service = FastMailboxNodeLocalInferenceService(
        endpoints=endpoints,
        replica=DoubleBufferedPolicyReplica(TinyPolicy, device="cpu"),
        infer_fn=_infer,
        max_batch_items=6,
        min_batch_items=6,
        max_wait_ms=50.0,
    )
    service.refresh_policy(snapshot)
    clients = [
        FastMailboxInferenceClient(endpoint, copy_outputs=False)
        for endpoint in endpoints
    ]
    inputs = [
        np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
        np.asarray([[5.0, 2.0], [3.0, 1.0]], dtype=np.float32),
        np.asarray([[0.5, 1.5], [4.0, 2.0]], dtype=np.float32),
    ]
    service.start()
    try:
        def run_round() -> list:
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [
                    pool.submit(
                        client.infer,
                        {"obs": value},
                        min_policy_version=0,
                        timeout=5.0,
                    )
                    for client, value in zip(clients, inputs, strict=True)
                ]
                return [future.result(timeout=8.0) for future in futures]

        first = run_round()
        first_views = [response.outputs["action"] for response in first]
        second = run_round()
        expected = [value @ np.asarray([[2.0], [-1.0]], np.float32) for value in inputs]
        for response, target in zip(second, expected, strict=True):
            np.testing.assert_allclose(response.outputs["action"], target)
            np.testing.assert_allclose(response.outputs["value"], target + 1.0)
        for old, new in zip(first_views, second, strict=True):
            assert np.shares_memory(old, new.outputs["action"])

        metrics = service.metrics()
        transport = service.transport_metrics()
        assert metrics.batches == 2
        assert metrics.request_messages == 6
        assert metrics.items == 12
        assert metrics.errors == 0
        assert transport.request_signals == 6
        assert transport.response_signals == 6
        assert transport.assembly_allocations == 1
        assert transport.reused_buffer_batches == 1
    finally:
        service.stop()
        for client in clients:
            client.close()
        for endpoint in endpoints:
            endpoint.close()
            endpoint.unlink()
