from __future__ import annotations

import numpy as np
import torch

from forge_rl.benchmarks.legacy_v1 import LegacyV1InferenceRuntime, SyntheticPolicy


def test_frozen_v1_queue_runtime_preserves_request_identity() -> None:
    torch.manual_seed(3)
    policy = SyntheticPolicy(8, 3)
    runtime = LegacyV1InferenceRuntime(
        actor_count=2,
        width=8,
        output_size=3,
        state_dict=policy.state_dict(),
        max_batch_items=8,
        min_batch_items=1,
        max_wait_ms=1.0,
    )
    runtime.start()
    try:
        for actor_id in range(2):
            for sequence_id in range(3):
                observation = np.full((4, 8), actor_id + sequence_id, dtype=np.float32)
                outputs, policy_version, returned_sequence = runtime.infer(
                    actor_id,
                    sequence_id,
                    observation,
                    timeout=10,
                )
                assert returned_sequence == sequence_id
                assert policy_version == 0
                assert outputs["action"].shape == (4, 3)
                assert outputs["value"].shape == (4, 1)
        metrics = runtime.metrics(expected_requests=6)
        assert metrics.errors == 0
        assert metrics.request_messages == 6
        assert metrics.items == 24
        assert metrics.serialized_request_bytes > 0
        assert metrics.serialized_response_bytes > 0
    finally:
        runtime.close()
