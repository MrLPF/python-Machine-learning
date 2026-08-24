from __future__ import annotations

import numpy as np
import pytest
import torch

from forge_rl.benchmarks import LegacyV1InferenceRuntime, ReferencePPOPolicy


def test_legacy_runtime_atomically_updates_policy_version() -> None:
    model = ReferencePPOPolicy(4, 2, 8)
    runtime = LegacyV1InferenceRuntime(
        actor_count=1,
        width=4,
        output_size=2,
        hidden_size=8,
        model_kind="reference_ppo",
        state_dict=model.state_dict(),
        max_batch_items=4,
        min_batch_items=1,
    )
    runtime.start()
    try:
        observation = np.ones((2, 4), np.float32)
        before, version, _ = runtime.infer(0, 0, observation, min_policy_version=0)
        assert version == 0
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.25)
        assert runtime.update_policy(model.state_dict(), version=1) == 1
        after, version, _ = runtime.infer(0, 1, observation, min_policy_version=1)
        assert version == 1
        assert not np.allclose(before["action"], after["action"])
        with pytest.raises(ValueError):
            runtime.update_policy(model.state_dict(), version=1)
    finally:
        runtime.close()
