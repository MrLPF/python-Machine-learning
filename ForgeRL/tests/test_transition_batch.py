from __future__ import annotations

import numpy as np
import pytest

from forge_rl.runtime.transition import TransitionBatch, TransitionIdentityError


def _batch(*, duplicate: bool = False) -> TransitionBatch:
    step_id = np.asarray([0, 0 if duplicate else 1, 2], dtype=np.int64)
    return TransitionBatch(
        obs={"features": np.zeros((3, 4), dtype=np.float32)},
        next_obs={"features": np.ones((3, 4), dtype=np.float32)},
        action={"action": np.asarray([0, 1, 0], dtype=np.int64)},
        reward=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        terminated=np.asarray([False, True, False]),
        truncated=np.asarray([False, False, True]),
        valid_mask=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        behavior_log_prob=np.zeros(3, dtype=np.float32),
        behavior_value=np.zeros(3, dtype=np.float32),
        policy_version=np.asarray([4, 4, 4], dtype=np.int64),
        episode_id=np.asarray([7, 7, 7], dtype=np.int64),
        step_id=step_id,
        actor_id=np.zeros(3, dtype=np.int64),
        env_id=np.zeros(3, dtype=np.int64),
    )


def test_bootstrap_and_trace_masks_are_distinct() -> None:
    batch = _batch()
    np.testing.assert_array_equal(batch.bootstrap_mask, np.asarray([1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(batch.trace_mask, np.asarray([1.0, 0.0, 0.0]))


def test_duplicate_transition_identity_is_rejected() -> None:
    with pytest.raises(TransitionIdentityError):
        _batch(duplicate=True)


def test_select_preserves_contract() -> None:
    selected = _batch().select([2, 0])
    assert selected.size == 2
    np.testing.assert_array_equal(selected.step_id, np.asarray([2, 0]))
