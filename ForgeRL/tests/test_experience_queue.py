from __future__ import annotations

import pytest

from forge_rl.runtime.experience import ExperienceItem, OnPolicyExperienceQueue


def _item(name: str, *, rows: int, version: int) -> ExperienceItem:
    return ExperienceItem.create(
        name,
        train_rows=rows,
        policy_version_min=version,
        policy_version_max=version,
        actor_id=0,
    )


def test_queue_applies_backpressure_by_rows() -> None:
    queue = OnPolicyExperienceQueue(capacity_rows=4)
    queue.put(_item("a", rows=3, version=1), timeout=0)
    with pytest.raises(TimeoutError):
        queue.put(_item("b", rows=2, version=1), timeout=0)


def test_queue_filters_policy_stale_items() -> None:
    queue = OnPolicyExperienceQueue(capacity_rows=10)
    queue.put(_item("old", rows=2, version=1), timeout=0)
    queue.put(_item("new", rows=2, version=5), timeout=0)

    sampled = queue.sample(
        min_rows=2,
        current_policy_version=5,
        max_policy_lag=1,
        timeout=0,
    )
    assert [item.payload for item in sampled] == ["new"]
    assert queue.dropped_stale == 1
