from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
import pytest

from forge_rl.transport import SharedTensorChannel, SlotState


def _child_send(channel: SharedTensorChannel) -> None:
    values = {
        "obs": np.arange(12, dtype=np.float32).reshape(3, 4),
        "mask": np.ones((3, 2), dtype=np.int8),
    }
    channel.send(values, item_count=3, sequence_id=17, metadata={"actor_id": 3}, timeout=2)
    channel.close()


def test_shared_tensor_channel_cross_process_payload() -> None:
    context = mp.get_context("spawn")
    channel = SharedTensorChannel.create(
        slot_count=2,
        max_items=4,
        fields={"obs": ((4,), np.float32), "mask": ((2,), np.int8)},
        mp_context=context,
    )
    process = context.Process(target=_child_send, args=(channel,))
    process.start()
    message = channel.receive(timeout=5)
    payload = channel.read(message, copy=True)
    assert message.sequence_id == 17
    assert message.metadata == {"actor_id": 3}
    np.testing.assert_array_equal(payload["obs"], np.arange(12, dtype=np.float32).reshape(3, 4))
    np.testing.assert_array_equal(payload["mask"], np.ones((3, 2), dtype=np.int8))
    channel.release(message)
    process.join(timeout=5)
    assert process.exitcode == 0
    channel.shutdown()
    channel.close()
    channel.unlink()


def test_generation_rejects_stale_descriptor_after_reclaim() -> None:
    channel = SharedTensorChannel.create(
        slot_count=1,
        max_items=2,
        fields={"obs": ((1,), np.float32)},
    )
    lease = channel.acquire(timeout=0.5)
    channel.write(lease, {"obs": np.ones((1, 1), dtype=np.float32)}, item_count=1)
    # Make the lease old without sleeping for a long time.
    channel._timestamps[lease.slot_id] = time.monotonic() - 1.0  # noqa: SLF001
    assert channel.reclaim_stale(max_age_seconds=0.01) == [0]
    with pytest.raises(RuntimeError, match="stale slot lease"):
        channel.commit(lease, item_count=1, sequence_id=1)
    replacement = channel.acquire(timeout=0.5)
    assert replacement.generation > lease.generation
    assert SlotState(int(channel._states[0])) is SlotState.WRITING  # noqa: SLF001
    channel.abort(replacement)
    channel.shutdown()
    channel.close()
    channel.unlink()
