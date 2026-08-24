from __future__ import annotations

import numpy as np

from forge_rl.runtime import TrajectoryBuilder, TransitionBatch


def _append(builder: TrajectoryBuilder, step_id: int, *, terminated: bool = False, truncated: bool = False, valid_mask: float = 1.0) -> None:
    builder.append(
        obs={"obs": np.asarray([step_id], dtype=np.float32)},
        next_obs={"obs": np.asarray([step_id + 1], dtype=np.float32)},
        action={"action": np.asarray(step_id % 2, dtype=np.int64)},
        reward=float(step_id),
        terminated=terminated,
        truncated=truncated,
        valid_mask=valid_mask,
        behavior_log_prob=-0.1,
        behavior_value=float(step_id) / 10.0,
        policy_version=3,
        episode_id=9,
        step_id=step_id,
    )


def test_fragmenting_never_drops_the_last_real_transition() -> None:
    builder = TrajectoryBuilder(actor_id=4, env_id=2, fragment_length=4)
    fragments = []
    for step in range(5):
        _append(builder, step)
        if builder.needs_flush:
            fragments.append(
                builder.flush(
                    bootstrap_obs={"obs": np.asarray([step + 1], dtype=np.float32)},
                    bootstrap_value=0.5,
                )
            )
    fragments.append(
        builder.flush(
            bootstrap_obs={"obs": np.asarray([5], dtype=np.float32)},
            bootstrap_value=0.75,
        )
    )
    combined = TransitionBatch.concat([fragment.transitions for fragment in fragments])
    np.testing.assert_array_equal(combined.step_id, np.arange(5, dtype=np.int64))
    assert combined.size == 5
    assert sum(fragment.train_rows for fragment in fragments) == 5


def test_terminated_zeroes_bootstrap_but_truncation_preserves_it() -> None:
    terminated_builder = TrajectoryBuilder(actor_id=1, env_id=1, fragment_length=8)
    _append(terminated_builder, 0, terminated=True)
    terminal = terminated_builder.flush(
        bootstrap_obs={"obs": np.asarray([1], dtype=np.float32)},
        bootstrap_value=99.0,
    )
    np.testing.assert_array_equal(terminal.bootstrap_value, np.zeros(1, dtype=np.float32))
    assert terminal.transitions.bootstrap_mask.tolist() == [0.0]

    truncated_builder = TrajectoryBuilder(actor_id=1, env_id=1, fragment_length=8)
    _append(truncated_builder, 0, truncated=True)
    truncated = truncated_builder.flush(
        bootstrap_obs={"obs": np.asarray([1], dtype=np.float32)},
        bootstrap_value=7.0,
    )
    np.testing.assert_array_equal(truncated.bootstrap_value, np.asarray([7.0], dtype=np.float32))
    assert truncated.transitions.bootstrap_mask.tolist() == [1.0]
    assert truncated.transitions.trace_mask.tolist() == [0.0]


def test_fault_rows_are_retained_for_diagnostics_but_not_counted_for_training() -> None:
    builder = TrajectoryBuilder(actor_id=2, env_id=3, fragment_length=4)
    _append(builder, 0, valid_mask=1.0)
    _append(builder, 1, valid_mask=0.0)
    fragment = builder.flush(
        bootstrap_obs={"obs": np.asarray([2], dtype=np.float32)},
        bootstrap_value=0.0,
    )
    assert fragment.transitions.size == 2
    assert fragment.train_rows == 1
    item = fragment.to_experience_item()
    assert item.train_rows == 1
