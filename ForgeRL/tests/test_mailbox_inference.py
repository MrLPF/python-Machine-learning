from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import numpy as np
import torch
from torch import nn

from forge_rl.runtime import (
    DoubleBufferedPolicyReplica,
    MailboxClientDescriptor,
    MailboxInferenceClient,
    MailboxInferenceEndpoint,
    MailboxNodeLocalInferenceService,
    PolicyRegistry,
)


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)


def _infer(module: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    action = module.linear(batch["obs"])
    return {"action": action, "value": action + 1.0}


def _endpoint(actor_id: int, *, context: mp.context.BaseContext | None = None) -> MailboxInferenceEndpoint:
    return MailboxInferenceEndpoint.create(
        actor_id=actor_id,
        max_items=2,
        request_fields={"obs": ((2,), np.float32)},
        response_fields={
            "action": ((1,), np.float32),
            "value": ((1,), np.float32),
        },
        mp_context=context,
    )


def _service(endpoints: list[MailboxInferenceEndpoint]) -> tuple[MailboxNodeLocalInferenceService, TinyPolicy, PolicyRegistry]:
    source = TinyPolicy()
    with torch.no_grad():
        source.linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
    registry = PolicyRegistry(history_size=2)
    snapshot = registry.publish(source.state_dict(), version=0)
    service = MailboxNodeLocalInferenceService(
        endpoints=endpoints,
        replica=DoubleBufferedPolicyReplica(TinyPolicy, device="cpu"),
        infer_fn=_infer,
        max_batch_items=6,
        min_batch_items=len(endpoints) * 2,
        max_wait_ms=50.0,
    )
    service.refresh_policy(snapshot)
    return service, source, registry


def _cleanup(
    service: MailboxNodeLocalInferenceService,
    clients: list[MailboxInferenceClient],
    endpoints: list[MailboxInferenceEndpoint],
) -> None:
    try:
        service.stop()
    finally:
        for client in clients:
            client.close()
        for endpoint in endpoints:
            endpoint.close()
            endpoint.unlink()


def test_mailbox_batches_without_generic_slot_queues_and_refreshes_policy() -> None:
    endpoints = [_endpoint(actor_id) for actor_id in range(3)]
    service, source, registry = _service(endpoints)
    clients = [MailboxInferenceClient(endpoint, copy_outputs=False) for endpoint in endpoints]
    inputs = [
        np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
        np.asarray([[5.0, 2.0], [3.0, 1.0]], dtype=np.float32),
        np.asarray([[0.5, 1.5], [4.0, 2.0]], dtype=np.float32),
    ]
    service.start()
    try:
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
            responses = [future.result(timeout=8.0) for future in futures]

        expected = [value @ np.asarray([[2.0], [-1.0]], np.float32) for value in inputs]
        for response, target in zip(responses, expected, strict=True):
            assert response.policy_version == 0
            np.testing.assert_allclose(response.outputs["action"], target)
            np.testing.assert_allclose(response.outputs["value"], target + 1.0)

        metrics = service.metrics()
        transport = service.transport_metrics()
        assert metrics.batches == 1
        assert metrics.request_messages == 3
        assert metrics.items == 6
        assert metrics.errors == 0
        assert transport.collector_threads == 1
        assert transport.request_signals == 3
        assert transport.response_signals == 3
        assert transport.assembly_allocations == 1

        with torch.no_grad():
            source.linear.weight.copy_(torch.tensor([[1.0, 1.0]]))
        snapshot = registry.publish(source.state_dict(), version=1)
        assert service.refresh_policy(snapshot) == 1
        response = clients[0].infer(
            {"obs": inputs[0]},
            min_policy_version=1,
            timeout=5.0,
        )
        assert response.policy_version == 1
        np.testing.assert_allclose(
            response.outputs["action"],
            inputs[0].sum(axis=1, keepdims=True),
        )
        assert service.transport_metrics().single_request_zero_copy_batches == 1
    finally:
        _cleanup(service, clients, endpoints)


def _spawn_actor(
    descriptor: MailboxClientDescriptor,
    result_queue: mp.Queue,
) -> None:
    client = MailboxInferenceClient(descriptor, copy_outputs=True)
    try:
        observation = np.asarray([[3.0, 1.0]], dtype=np.float32)
        response = client.infer(
            {"obs": observation},
            min_policy_version=0,
            timeout=10.0,
        )
        result_queue.put(
            (
                response.policy_version,
                response.sequence_id,
                response.outputs["action"].tolist(),
                response.outputs["value"].tolist(),
            )
        )
    finally:
        client.close()


def test_mailbox_client_descriptor_supports_spawned_actor_process() -> None:
    context = mp.get_context("spawn")
    endpoint = _endpoint(7, context=context)
    service, _source, _registry = _service([endpoint])
    service.min_batch_items = 1
    result_queue = context.Queue()
    process = context.Process(
        target=_spawn_actor,
        args=(endpoint.client_descriptor(), result_queue),
    )
    service.start()
    try:
        process.start()
        process.join(20.0)
        assert process.exitcode == 0
        version, sequence, action, value = result_queue.get(timeout=5.0)
        assert version == 0
        assert sequence == 0
        np.testing.assert_allclose(action, [[5.0]])
        np.testing.assert_allclose(value, [[6.0]])
        assert service.metrics().errors == 0
        assert service.transport_metrics().request_signals == 1
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5.0)
        service.stop()
        endpoint.close()
        endpoint.unlink()
