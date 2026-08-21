from __future__ import annotations

import multiprocessing as mp

import numpy as np

from forge_rl.transport.shared_memory import SharedMemoryArena, SharedMemoryDescriptor


def _writer(descriptor: SharedMemoryDescriptor) -> None:
    arena = SharedMemoryArena.attach(descriptor)
    try:
        arena.write(1, np.asarray([3.0, 4.0, 5.0], dtype=np.float32))
    finally:
        arena.close()


def test_shared_memory_arena_is_visible_across_processes() -> None:
    arena = SharedMemoryArena.create(slot_count=2, slot_shape=(3,), dtype=np.float32)
    try:
        process = mp.get_context("spawn").Process(target=_writer, args=(arena.descriptor,))
        process.start()
        process.join(timeout=10)
        assert process.exitcode == 0
        np.testing.assert_array_equal(
            arena.view(1), np.asarray([3.0, 4.0, 5.0], dtype=np.float32)
        )
    finally:
        arena.close()
        arena.unlink()
