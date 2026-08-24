from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SharedMemoryDescriptor:
    name: str
    slot_count: int
    slot_shape: tuple[int, ...]
    dtype: str


class SharedMemoryArena:
    """Fixed-shape shared-memory slots for local zero-copy payload exchange.

    Slot ownership/backpressure is intentionally handled by the runtime queue. The arena only
    defines allocation and attachment, keeping data-plane bytes separate from control messages.
    """

    def __init__(
        self,
        *,
        memory: shared_memory.SharedMemory,
        slot_count: int,
        slot_shape: tuple[int, ...],
        dtype: np.dtype[Any],
        owner: bool,
    ) -> None:
        self._memory = memory
        self.slot_count = int(slot_count)
        self.slot_shape = tuple(int(value) for value in slot_shape)
        self.dtype = np.dtype(dtype)
        self.owner = bool(owner)
        self._closed = False
        self._array = np.ndarray(
            (self.slot_count, *self.slot_shape), dtype=self.dtype, buffer=self._memory.buf
        )

    @classmethod
    def create(
        cls,
        *,
        slot_count: int,
        slot_shape: tuple[int, ...],
        dtype: np.dtype[Any] | str = np.float32,
    ) -> "SharedMemoryArena":
        if slot_count <= 0 or not slot_shape or any(int(value) <= 0 for value in slot_shape):
            raise ValueError("slot_count and every slot_shape dimension must be positive")
        normalized = np.dtype(dtype)
        nbytes = int(slot_count) * int(np.prod(slot_shape)) * normalized.itemsize
        memory = shared_memory.SharedMemory(create=True, size=nbytes)
        arena = cls(
            memory=memory,
            slot_count=slot_count,
            slot_shape=slot_shape,
            dtype=normalized,
            owner=True,
        )
        arena._array.fill(0)
        return arena

    @classmethod
    def attach(cls, descriptor: SharedMemoryDescriptor) -> "SharedMemoryArena":
        memory = shared_memory.SharedMemory(name=descriptor.name, create=False)
        return cls(
            memory=memory,
            slot_count=descriptor.slot_count,
            slot_shape=descriptor.slot_shape,
            dtype=np.dtype(descriptor.dtype),
            owner=False,
        )

    @property
    def descriptor(self) -> SharedMemoryDescriptor:
        return SharedMemoryDescriptor(
            name=self._memory.name,
            slot_count=self.slot_count,
            slot_shape=self.slot_shape,
            dtype=self.dtype.str,
        )

    def view(self, slot_id: int) -> np.ndarray:
        if self._closed:
            raise RuntimeError("shared-memory arena is closed")
        selected = int(slot_id)
        if selected < 0 or selected >= self.slot_count:
            raise IndexError(f"slot_id {selected} outside [0, {self.slot_count})")
        return self._array[selected]

    def write(self, slot_id: int, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=self.dtype)
        if array.shape != self.slot_shape:
            raise ValueError(f"expected shape {self.slot_shape}, got {array.shape}")
        np.copyto(self.view(slot_id), array, casting="no")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._memory.close()

    def unlink(self) -> None:
        if not self.owner:
            raise RuntimeError("only the creating process may unlink the arena")
        try:
            self._memory.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "SharedMemoryArena":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
        if self.owner:
            self.unlink()
