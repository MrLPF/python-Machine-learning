from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import torch

from forge_rl.benchmarks.legacy_v1 import SyntheticPolicy, run_synthetic_policy
from forge_rl.runtime import ProcessMailboxInferenceRuntime


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _expected(
    module: SyntheticPolicy,
    observation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        action, value = module(torch.from_numpy(observation))
    return action.numpy(), value.numpy()


def test_process_mailbox_isolates_predictor_batches_and_updates_policy() -> None:
    torch.manual_seed(123)
    source = SyntheticPolicy(2, 1)
    runtime = ProcessMailboxInferenceRuntime(
        actor_count=3,
        max_items=2,
        request_fields={"obs": ((2,), np.float32)},
        response_fields={
            "action": ((1,), np.float32),
            "value": ((1,), np.float32),
        },
        module_type=SyntheticPolicy,
        module_args=(2, 1),
        infer_fn=run_synthetic_policy,
        state_dict=_state(source),
        max_batch_items=6,
        min_batch_items=6,
        max_wait_ms=50.0,
        copy_outputs=False,
    )
    observations = [
        np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
        np.asarray([[5.0, 2.0], [3.0, 1.0]], dtype=np.float32),
        np.asarray([[0.5, 1.5], [4.0, 2.0]], dtype=np.float32),
    ]
    runtime.start()
    try:
        assert runtime.process_pid is not None
        assert runtime.process_pid != os.getpid()

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(
                    runtime.infer,
                    actor_id,
                    {"obs": observation},
                    min_policy_version=0,
                    timeout=10.0,
                )
                for actor_id, observation in enumerate(observations)
            ]
            responses = [future.result(timeout=15.0) for future in futures]

        for response, observation in zip(responses, observations, strict=True):
            expected_action, expected_value = _expected(source, observation)
            assert response.policy_version == 0
            np.testing.assert_allclose(response.outputs["action"], expected_action)
            np.testing.assert_allclose(response.outputs["value"], expected_value)

        metrics, transport = runtime.metrics_pair()
        assert metrics.batches == 1
        assert metrics.request_messages == 3
        assert metrics.items == 6
        assert metrics.errors == 0
        assert transport.request_signals == 3
        assert transport.response_signals == 3

        with torch.no_grad():
            for parameter in source.parameters():
                parameter.add_(0.25)
        assert runtime.update_policy(_state(source), version=1) == 1
        response = runtime.infer(
            0,
            {"obs": observations[0]},
            min_policy_version=1,
            timeout=10.0,
        )
        expected_action, expected_value = _expected(source, observations[0])
        assert response.policy_version == 1
        np.testing.assert_allclose(response.outputs["action"], expected_action)
        np.testing.assert_allclose(response.outputs["value"], expected_value)
    finally:
        runtime.close()
