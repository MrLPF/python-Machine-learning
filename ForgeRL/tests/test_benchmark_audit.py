from __future__ import annotations

import numpy as np

from forge_rl.benchmarks.audit import audit_transition_identities, identities_from_batch
from forge_rl.runtime import TransitionBatch


def _batch() -> TransitionBatch:
    size = 3
    return TransitionBatch(
        obs={"obs": np.zeros((size, 2), dtype=np.float32)},
        next_obs={"obs": np.ones((size, 2), dtype=np.float32)},
        action={"action": np.zeros((size,), dtype=np.int64)},
        reward=np.zeros(size, dtype=np.float32),
        terminated=np.asarray([False, False, True]),
        truncated=np.asarray([False, False, False]),
        valid_mask=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        behavior_log_prob=np.zeros(size, dtype=np.float32),
        behavior_value=np.zeros(size, dtype=np.float32),
        policy_version=np.zeros(size, dtype=np.int64),
        episode_id=np.asarray([4, 4, 4], dtype=np.int64),
        step_id=np.asarray([0, 1, 2], dtype=np.int64),
        actor_id=np.asarray([2, 2, 2], dtype=np.int64),
        env_id=np.asarray([7, 7, 7], dtype=np.int64),
    )


def test_identity_audit_passes_for_exact_valid_rows() -> None:
    identities = identities_from_batch(_batch())
    assert identities == [(2, 7, 4, 0), (2, 7, 4, 2)]
    audit = audit_transition_identities(identities, identities)
    assert audit.passed
    assert audit.useful_sample_ratio == 1.0


def test_identity_audit_reports_missing_duplicate_and_invalid_rows() -> None:
    collected = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2)]
    trained = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 9)]
    audit = audit_transition_identities(collected, trained, invalid_rows_in_loss=1)
    assert not audit.passed
    assert audit.duplicate_trained_rows == 1
    assert audit.missing_rows == 2
    assert audit.unexpected_rows == 1
    assert audit.invalid_rows_in_loss == 1
