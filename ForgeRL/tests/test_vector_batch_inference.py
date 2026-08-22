from __future__ import annotations

import numpy as np
import pytest
import torch

from forge_rl.benchmarks.legacy_v1 import SyntheticPolicy, run_synthetic_policy
from forge_rl.runtime import VectorBatchInferenceRuntime


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def test_vector_batch_executes_one_forward_and_splits_actor_views() -> None:
    torch.manual_seed(2028)
    source = SyntheticPolicy(3, 2)
    runtime = VectorBatchInferenceRuntime(
        actor_count=3,
        module_type=SyntheticPolicy,
        module_args=(3, 2),
        infer_fn=run_synthetic_policy,
        state_dict=_state(source),
        max_batch_items=8,
        copy_outputs=False,
    )
    observations = {
        0: np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], np.float32),
        1: np.asarray([[5.0, 6.0, 7.0]], np.float32),
        2: np.asarray([[8.0, 9.0, 10.0], [11.0, 12.0, 13.0]], np.float32),
    }
    merged = np.concatenate([observations[index] for index in range(3)], axis=0)

    runtime.start()
    try:
        responses = runtime.infer_batch(
            {"obs": merged},
            actor_item_counts={0: 2, 1: 1, 2: 2},
            min_policy_version=0,
        )
        with torch.inference_mode():
            expected_action, expected_value = source(torch.from_numpy(merged))

        offset = 0
        for actor, count in ((0, 2), (1, 1), (2, 2)):
            stop = offset + count
            response = responses[actor]
            assert response.policy_version == 0
            assert response.sequence_id == 0
            np.testing.assert_allclose(
                response.outputs["action"], expected_action[offset:stop].numpy()
            )
            np.testing.assert_allclose(
                response.outputs["value"], expected_value[offset:stop].numpy()
            )
            offset = stop

        second = runtime.infer_batch(
            {"obs": merged},
            actor_item_counts={0: 2, 1: 1, 2: 2},
            min_policy_version=0,
        )
        assert all(response.sequence_id == 1 for response in second.values())
        metrics = runtime.metrics()
        assert metrics.batches == 2
        assert metrics.request_messages == 6
        assert metrics.items == 10
        assert metrics.mean_batch_items == 5
        assert metrics.errors == 0
    finally:
        runtime.close()


def test_vector_batch_policy_update_and_contract_validation() -> None:
    torch.manual_seed(2029)
    source = SyntheticPolicy(2, 1)
    runtime = VectorBatchInferenceRuntime(
        actor_count=2,
        module_type=SyntheticPolicy,
        module_args=(2, 1),
        infer_fn=run_synthetic_policy,
        state_dict=_state(source),
        max_batch_items=4,
    )
    observation = np.asarray(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        dtype=np.float32,
    )

    runtime.start()
    try:
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.add_(0.125)
        assert runtime.update_policy(_state(source), version=1) == 1
        responses = runtime.infer_batch(
            {"obs": observation},
            actor_item_counts={0: 2, 1: 2},
            min_policy_version=1,
        )
        assert all(response.policy_version == 1 for response in responses.values())

        with pytest.raises(ValueError, match="leading dimension"):
            runtime.infer_batch(
                {"obs": observation[:3]},
                actor_item_counts={0: 2, 1: 2},
            )
        with pytest.raises(ValueError, match="capacity"):
            runtime.infer_batch(
                {"obs": np.concatenate([observation, observation[:1]], axis=0)},
                actor_item_counts={0: 3, 1: 2},
            )
    finally:
        runtime.close()
