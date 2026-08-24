from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import torch

from forge_rl.benchmarks.legacy_v1 import SyntheticPolicy, run_synthetic_policy
from forge_rl.runtime import InProcessBatchingInferenceRuntime


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def test_rendezvous_waits_for_a_staggered_actor_wave() -> None:
    torch.manual_seed(2026)
    source = SyntheticPolicy(2, 1)
    runtime = InProcessBatchingInferenceRuntime(
        actor_count=4,
        module_type=SyntheticPolicy,
        module_args=(2, 1),
        infer_fn=run_synthetic_policy,
        state_dict=_state(source),
        max_batch_items=4,
        min_batch_items=2,
        max_wait_ms=100.0,
        copy_outputs=False,
    )
    observations = [
        np.asarray([[float(actor), float(actor + 1)]], dtype=np.float32)
        for actor in range(4)
    ]

    def infer(actor: int):
        # Deliberately stagger arrivals beyond the old minimum-size dispatch point. The runtime
        # should still rendezvous the complete synchronous Actor wave before the hard deadline.
        time.sleep(actor * 0.005)
        return runtime.infer(actor, {"obs": observations[actor]}, timeout=5.0)

    runtime.start()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(infer, range(4)))

        metrics = runtime.metrics()
        assert metrics.batches == 1
        assert metrics.request_messages == 4
        assert metrics.items == 4
        assert metrics.mean_batch_items == 4
        assert metrics.errors == 0

        for actor, response in enumerate(responses):
            with torch.inference_mode():
                expected_action, expected_value = source(
                    torch.from_numpy(observations[actor])
                )
            assert response.sequence_id == 0
            np.testing.assert_allclose(
                response.outputs["action"], expected_action.numpy()
            )
            np.testing.assert_allclose(
                response.outputs["value"], expected_value.numpy()
            )
    finally:
        runtime.close()


def test_rendezvous_timeout_does_not_deadlock_when_an_actor_is_absent() -> None:
    torch.manual_seed(2027)
    source = SyntheticPolicy(2, 1)
    runtime = InProcessBatchingInferenceRuntime(
        actor_count=4,
        module_type=SyntheticPolicy,
        module_args=(2, 1),
        infer_fn=run_synthetic_policy,
        state_dict=_state(source),
        max_batch_items=4,
        min_batch_items=2,
        max_wait_ms=20.0,
        copy_outputs=False,
    )
    observation = np.asarray([[1.0, 2.0]], dtype=np.float32)

    runtime.start()
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(runtime.infer, actor, {"obs": observation}, timeout=5.0)
                for actor in range(3)
            ]
            responses = [future.result(timeout=5.0) for future in futures]

        metrics = runtime.metrics()
        assert len(responses) == 3
        assert metrics.batches == 1
        assert metrics.request_messages == 3
        assert metrics.items == 3
        assert metrics.errors == 0
    finally:
        runtime.close()
