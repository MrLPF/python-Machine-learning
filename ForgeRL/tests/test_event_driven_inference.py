from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn

from forge_rl.runtime import (
    DoubleBufferedPolicyReplica,
    NodeLocalInferenceService,
    PolicyRegistry,
    PollingNodeLocalInferenceService,
    SharedInferenceClient,
    SharedInferenceEndpoint,
)


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)


def _infer(module: TinyPolicy, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    action = module.linear(batch["obs"])
    return {"action": action, "value": action + 1.0}


def _endpoint(actor_id: int) -> SharedInferenceEndpoint:
    return SharedInferenceEndpoint.create(
        actor_id=actor_id,
        slot_count=4,
        max_items=2,
        request_fields={"obs": ((2,), np.float32)},
        response_fields={
            "action": ((1,), np.float32),
            "value": ((1,), np.float32),
        },
    )


def _close(endpoints: list[SharedInferenceEndpoint]) -> None:
    for endpoint in endpoints:
        endpoint.shutdown()
        endpoint.close()
        endpoint.unlink()


def test_event_driven_service_routes_descriptors_without_endpoint_polling() -> None:
    endpoints = [_endpoint(actor_id) for actor_id in range(3)]
    original_queues = [endpoint.request._ready_queue for endpoint in endpoints]
    source = TinyPolicy()
    with torch.no_grad():
        source.linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
    registry = PolicyRegistry(history_size=2)
    snapshot = registry.publish(source.state_dict(), version=0)
    service = NodeLocalInferenceService(
        endpoints=endpoints,
        replica=DoubleBufferedPolicyReplica(TinyPolicy, device="cpu"),
        infer_fn=_infer,
        max_batch_items=6,
        min_batch_items=6,
        max_wait_ms=50.0,
    )
    service.refresh_policy(snapshot)
    service.start()
    clients = [SharedInferenceClient(endpoint) for endpoint in endpoints]
    observations = [
        np.asarray([[actor + 1.0, 1.0], [actor + 2.0, 2.0]], dtype=np.float32)
        for actor in range(3)
    ]
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            responses = list(
                executor.map(
                    lambda pair: pair[0].infer(
                        {"obs": pair[1]}, min_policy_version=0, timeout=5.0
                    ),
                    zip(clients, observations, strict=True),
                )
            )
        for response, observation in zip(responses, observations, strict=True):
            expected = observation @ np.asarray([[2.0], [-1.0]], dtype=np.float32)
            assert response.policy_version == 0
            np.testing.assert_allclose(response.outputs["action"], expected)
            np.testing.assert_allclose(response.outputs["value"], expected + 1.0)

        metrics = service.metrics()
        event = service.event_metrics()
        assert metrics.errors == 0
        assert metrics.request_messages == 3
        assert metrics.items == 6
        assert event.collector_mode == "shared-ready-descriptor-queue"
        assert event.notification_tokens == 3
        assert event.stale_notifications == 0
        assert event.invalid_actor_notifications == 0
        assert service.optimization_metrics().collector_threads == 1
    finally:
        service.stop()
        assert all(
            endpoint.request._ready_queue is original
            for endpoint, original in zip(endpoints, original_queues, strict=True)
        )
        _close(endpoints)


def test_polling_implementation_remains_available_for_regression() -> None:
    assert PollingNodeLocalInferenceService is not NodeLocalInferenceService


def _spawn_actor(
    endpoint: SharedInferenceEndpoint,
    observation: np.ndarray,
    result_queue: object,
) -> None:
    try:
        client = SharedInferenceClient(endpoint)
        response = client.infer(
            {"obs": observation},
            min_policy_version=0,
            timeout=10.0,
        )
        result_queue.put(
            (
                response.policy_version,
                response.outputs["action"],
                response.outputs["value"],
            )
        )
    finally:
        endpoint.close()


def test_event_notification_queue_survives_spawned_actor_process() -> None:
    import multiprocessing as mp

    context = mp.get_context("spawn")
    endpoint = SharedInferenceEndpoint.create(
        actor_id=7,
        slot_count=2,
        max_items=2,
        request_fields={"obs": ((2,), np.float32)},
        response_fields={
            "action": ((1,), np.float32),
            "value": ((1,), np.float32),
        },
        mp_context=context,
    )
    source = TinyPolicy()
    with torch.no_grad():
        source.linear.weight.copy_(torch.tensor([[1.5, -0.5]]))
    registry = PolicyRegistry(history_size=2)
    snapshot = registry.publish(source.state_dict(), version=0)
    service = NodeLocalInferenceService(
        endpoints=[endpoint],
        replica=DoubleBufferedPolicyReplica(TinyPolicy, device="cpu"),
        infer_fn=_infer,
        max_batch_items=2,
        min_batch_items=2,
        max_wait_ms=10.0,
        mp_context=context,
    )
    service.refresh_policy(snapshot)
    service.start()
    observation = np.asarray([[2.0, 1.0], [4.0, 3.0]], dtype=np.float32)
    result_queue = context.Queue()
    process = context.Process(
        target=_spawn_actor,
        args=(endpoint, observation, result_queue),
    )
    try:
        process.start()
        process.join(15.0)
        assert not process.is_alive()
        assert process.exitcode == 0
        version, action, value = result_queue.get(timeout=2.0)
        expected = observation @ np.asarray([[1.5], [-0.5]], dtype=np.float32)
        assert version == 0
        np.testing.assert_allclose(action, expected)
        np.testing.assert_allclose(value, expected + 1.0)
        assert service.event_metrics().notification_tokens == 1
        assert service.metrics().errors == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(2.0)
        service.stop()
        result_queue.close()
        result_queue.join_thread()
        _close([endpoint])
