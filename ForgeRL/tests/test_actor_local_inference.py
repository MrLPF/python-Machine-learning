from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from forge_rl.benchmarks.legacy_v1 import SyntheticPolicy, run_synthetic_policy
from forge_rl.runtime import ActorLocalInferenceRuntime


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def test_actor_local_runtime_runs_replicas_concurrently_and_updates_atomically() -> None:
    torch.manual_seed(321)
    source = SyntheticPolicy(3, 2)
    runtime = ActorLocalInferenceRuntime(
        actor_count=4,
        module_type=SyntheticPolicy,
        module_args=(3, 2),
        infer_fn=run_synthetic_policy,
        state_dict=_state(source),
        max_batch_items=2,
    )
    observations = [
        np.asarray(
            [[actor + 0.5, actor + 1.5, actor + 2.5], [1.0, 2.0, 3.0]],
            dtype=np.float32,
        )
        for actor in range(4)
    ]
    runtime.start()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(
                pool.map(
                    lambda row: runtime.infer(
                        row[0],
                        {"obs": row[1]},
                        min_policy_version=0,
                    ),
                    enumerate(observations),
                )
            )

        for response, observation in zip(responses, observations, strict=True):
            with torch.inference_mode():
                expected_action, expected_value = source(torch.from_numpy(observation))
            assert response.policy_version == 0
            assert response.sequence_id == 0
            np.testing.assert_allclose(
                response.outputs["action"], expected_action.numpy()
            )
            np.testing.assert_allclose(
                response.outputs["value"], expected_value.numpy()
            )

        metrics = runtime.metrics()
        assert metrics.batches == 4
        assert metrics.request_messages == 4
        assert metrics.items == 8
        assert metrics.errors == 0

        with torch.no_grad():
            for parameter in source.parameters():
                parameter.add_(0.125)
        assert runtime.update_policy(_state(source), version=1) == 1

        response = runtime.infer(
            0,
            {"obs": observations[0]},
            min_policy_version=1,
        )
        with torch.inference_mode():
            expected_action, expected_value = source(
                torch.from_numpy(observations[0])
            )
        assert response.policy_version == 1
        assert response.sequence_id == 1
        np.testing.assert_allclose(
            response.outputs["action"], expected_action.numpy()
        )
        np.testing.assert_allclose(
            response.outputs["value"], expected_value.numpy()
        )
    finally:
        runtime.close()
