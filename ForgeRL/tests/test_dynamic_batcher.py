from __future__ import annotations

from forge_rl.runtime.dynamic_batcher import DeadlineBatcher


def test_batcher_respects_item_limit_and_defers_overflow() -> None:
    batcher: DeadlineBatcher[str] = DeadlineBatcher(
        max_items=5, min_items=1, max_wait_ms=0.0
    )
    batcher.submit("first", item_count=4)
    batcher.submit("second", item_count=3)

    first = batcher.pop(timeout=0)
    second = batcher.pop(timeout=0)

    assert [item.payload for item in first] == ["first"]
    assert [item.payload for item in second] == ["second"]


def test_batcher_combines_requests_to_minimum() -> None:
    batcher: DeadlineBatcher[str] = DeadlineBatcher(
        max_items=8, min_items=4, max_wait_ms=10.0
    )
    batcher.submit("a", item_count=2)
    batcher.submit("b", item_count=2)
    batch = batcher.pop(timeout=0.1)
    assert [item.payload for item in batch] == ["a", "b"]
