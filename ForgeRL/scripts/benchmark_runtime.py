#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from forge_rl.runtime.dynamic_batcher import DeadlineBatcher
from forge_rl.transport.shared_memory import SharedMemoryArena


def benchmark_shared_memory(*, items: int, width: int, slots: int) -> dict[str, float]:
    payload = np.arange(width, dtype=np.float32)
    arena = SharedMemoryArena.create(slot_count=slots, slot_shape=(width,), dtype=np.float32)
    try:
        started = time.perf_counter()
        checksum = 0.0
        for index in range(items):
            slot = index % slots
            arena.write(slot, payload)
            checksum += float(arena.view(slot)[0])
        elapsed = time.perf_counter() - started
    finally:
        arena.close()
        arena.unlink()
    return {
        "shared_memory_ops": float(items),
        "shared_memory_seconds": elapsed,
        "shared_memory_ops_per_second": items / max(elapsed, 1e-12),
        "shared_memory_checksum": checksum,
    }


def benchmark_batcher(*, items: int, max_batch_size: int) -> dict[str, float]:
    batcher: DeadlineBatcher[int] = DeadlineBatcher(
        max_items=max_batch_size,
        min_items=max_batch_size,
        max_wait_ms=0.0,
        max_queued_items=max(items, max_batch_size),
    )
    started = time.perf_counter()
    for index in range(items):
        batcher.submit(index, item_count=1, timeout=0)
    batch_count = 0
    emitted = 0
    while emitted < items:
        batch = batcher.pop(timeout=0)
        if not batch:
            raise RuntimeError("batcher stopped before emitting every item")
        emitted += sum(item.item_count for item in batch)
        batch_count += 1
    elapsed = time.perf_counter() - started
    return {
        "batcher_items": float(items),
        "batcher_batches": float(batch_count),
        "batcher_mean_batch_size": items / max(batch_count, 1),
        "batcher_seconds": elapsed,
        "batcher_items_per_second": items / max(elapsed, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=100_000)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.items <= 0:
        raise SystemExit("--items must be positive")
    result = {
        **benchmark_shared_memory(items=args.items, width=args.width, slots=args.slots),
        **benchmark_batcher(items=args.items, max_batch_size=args.batch_size),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
