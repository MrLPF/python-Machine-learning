from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import multiprocessing as mp
from multiprocessing.context import BaseContext
import queue
import time
from typing import Any, Mapping

import numpy as np

from .shared_memory import SharedMemoryArena, SharedMemoryDescriptor


class SlotState(IntEnum):
    FREE = 0
    WRITING = 1
    READY = 2
    READING = 3


@dataclass(frozen=True, slots=True)
class TensorFieldSpec:
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        normalized_shape = tuple(int(value) for value in self.shape)
        if any(value <= 0 for value in normalized_shape):
            raise ValueError("every field dimension must be positive")
        object.__setattr__(self, "shape", normalized_shape)
        object.__setattr__(self, "dtype", np.dtype(self.dtype).str)


@dataclass(frozen=True, slots=True)
class SharedTensorTreeDescriptor:
    slot_count: int
    max_items: int
    fields: tuple[tuple[str, TensorFieldSpec, SharedMemoryDescriptor], ...]


@dataclass(frozen=True, slots=True)
class SlotLease:
    slot_id: int
    generation: int
    acquired_at: float


@dataclass(frozen=True, slots=True)
class SlotMessage:
    slot_id: int
    generation: int
    item_count: int
    sequence_id: int
    metadata: dict[str, Any] = field(default_factory=dict)


class SharedTensorTreeArena:
    """A fixed-schema tensor tree stored in one shared-memory arena per field."""

    def __init__(
        self,
        *,
        slot_count: int,
        max_items: int,
        fields: Mapping[str, TensorFieldSpec],
        arenas: Mapping[str, SharedMemoryArena],
        owner: bool,
    ) -> None:
        self.slot_count = int(slot_count)
        self.max_items = int(max_items)
        self.fields = dict(fields)
        self._arenas = dict(arenas)
        self.owner = bool(owner)
        self._closed = False

    @classmethod
    def create(
        cls,
        *,
        slot_count: int,
        max_items: int,
        fields: Mapping[str, TensorFieldSpec | tuple[tuple[int, ...], Any]],
    ) -> "SharedTensorTreeArena":
        if slot_count <= 0 or max_items <= 0:
            raise ValueError("slot_count and max_items must be positive")
        normalized: dict[str, TensorFieldSpec] = {}
        for name, value in fields.items():
            if not name:
                raise ValueError("field names must be non-empty")
            if isinstance(value, TensorFieldSpec):
                spec = value
            else:
                shape, dtype = value
                spec = TensorFieldSpec(tuple(shape), np.dtype(dtype).str)
            normalized[name] = spec
        if not normalized:
            raise ValueError("fields must not be empty")
        arenas: dict[str, SharedMemoryArena] = {}
        try:
            for name, spec in normalized.items():
                arenas[name] = SharedMemoryArena.create(
                    slot_count=slot_count,
                    slot_shape=(max_items, *spec.shape),
                    dtype=np.dtype(spec.dtype),
                )
        except BaseException:
            for arena in arenas.values():
                arena.close()
                arena.unlink()
            raise
        return cls(
            slot_count=slot_count,
            max_items=max_items,
            fields=normalized,
            arenas=arenas,
            owner=True,
        )

    @classmethod
    def attach(cls, descriptor: SharedTensorTreeDescriptor) -> "SharedTensorTreeArena":
        fields: dict[str, TensorFieldSpec] = {}
        arenas: dict[str, SharedMemoryArena] = {}
        try:
            for name, spec, memory_descriptor in descriptor.fields:
                fields[name] = spec
                arenas[name] = SharedMemoryArena.attach(memory_descriptor)
        except BaseException:
            for arena in arenas.values():
                arena.close()
            raise
        return cls(
            slot_count=descriptor.slot_count,
            max_items=descriptor.max_items,
            fields=fields,
            arenas=arenas,
            owner=False,
        )

    @property
    def descriptor(self) -> SharedTensorTreeDescriptor:
        return SharedTensorTreeDescriptor(
            slot_count=self.slot_count,
            max_items=self.max_items,
            fields=tuple(
                (name, self.fields[name], self._arenas[name].descriptor)
                for name in sorted(self.fields)
            ),
        )

    def write(self, slot_id: int, values: Mapping[str, np.ndarray], *, item_count: int) -> None:
        if self._closed:
            raise RuntimeError("shared tensor arena is closed")
        if item_count <= 0 or item_count > self.max_items:
            raise ValueError("item_count outside channel capacity")
        if set(values) != set(self.fields):
            raise ValueError(
                f"tensor tree fields differ: expected {sorted(self.fields)}, got {sorted(values)}"
            )
        for name, spec in self.fields.items():
            source = np.asarray(values[name], dtype=np.dtype(spec.dtype))
            expected = (item_count, *spec.shape)
            if source.shape != expected:
                raise ValueError(f"field {name!r}: expected {expected}, got {source.shape}")
            target = self._arenas[name].view(slot_id)
            np.copyto(target[:item_count], source, casting="no")

    def read(
        self,
        slot_id: int,
        *,
        item_count: int,
        copy: bool = False,
    ) -> dict[str, np.ndarray]:
        if item_count <= 0 or item_count > self.max_items:
            raise ValueError("item_count outside channel capacity")
        result = {
            name: self._arenas[name].view(slot_id)[:item_count]
            for name in sorted(self.fields)
        }
        if copy:
            return {name: value.copy() for name, value in result.items()}
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for arena in self._arenas.values():
            arena.close()

    def unlink(self) -> None:
        if not self.owner:
            raise RuntimeError("only the creating process may unlink the tensor arena")
        for arena in self._arenas.values():
            arena.unlink()


class SharedTensorChannel:
    """Bounded shared-memory tensor channel with generation-safe slot ownership.

    Tensor payloads stay in shared memory. Only compact ``SlotMessage`` descriptors cross the
    multiprocessing queues. The state machine catches use-after-release and permits the owner to
    reclaim slots held by a crashed producer/consumer.
    """

    def __init__(
        self,
        *,
        descriptor: SharedTensorTreeDescriptor,
        free_queue: Any,
        ready_queue: Any,
        states: Any,
        generations: Any,
        timestamps: Any,
        lock: Any,
        closed: Any,
        owner: bool,
        arena: SharedTensorTreeArena | None = None,
    ) -> None:
        self._descriptor = descriptor
        self._free_queue = free_queue
        self._ready_queue = ready_queue
        self._states = states
        self._generations = generations
        self._timestamps = timestamps
        self._lock = lock
        self._closed_event = closed
        self.owner = bool(owner)
        self._arena = arena or SharedTensorTreeArena.attach(descriptor)
        self._local_closed = False

    @classmethod
    def create(
        cls,
        *,
        slot_count: int,
        max_items: int,
        fields: Mapping[str, TensorFieldSpec | tuple[tuple[int, ...], Any]],
        mp_context: BaseContext | None = None,
    ) -> "SharedTensorChannel":
        context = mp_context or mp.get_context()
        arena = SharedTensorTreeArena.create(
            slot_count=slot_count,
            max_items=max_items,
            fields=fields,
        )
        free_queue = context.Queue(maxsize=slot_count)
        ready_queue = context.Queue(maxsize=slot_count)
        for slot_id in range(slot_count):
            free_queue.put(slot_id)
        states = context.Array("b", [int(SlotState.FREE)] * slot_count, lock=False)
        generations = context.Array("Q", [0] * slot_count, lock=False)
        timestamps = context.Array("d", [0.0] * slot_count, lock=False)
        lock = context.RLock()
        closed = context.Event()
        return cls(
            descriptor=arena.descriptor,
            free_queue=free_queue,
            ready_queue=ready_queue,
            states=states,
            generations=generations,
            timestamps=timestamps,
            lock=lock,
            closed=closed,
            owner=True,
            arena=arena,
        )

    @property
    def descriptor(self) -> SharedTensorTreeDescriptor:
        return self._descriptor

    @property
    def slot_count(self) -> int:
        return self._descriptor.slot_count

    @property
    def max_items(self) -> int:
        return self._descriptor.max_items

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arena"] = None
        state["owner"] = False
        state["_local_closed"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._arena = SharedTensorTreeArena.attach(self._descriptor)

    def _remaining(self, deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def acquire(self, *, timeout: float | None = None) -> SlotLease:
        if self._closed_event.is_set():
            raise RuntimeError("shared tensor channel is closed")
        try:
            slot_id = self._free_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("timed out waiting for a free shared-memory slot") from exc
        now = time.monotonic()
        with self._lock:
            state = SlotState(int(self._states[slot_id]))
            if state is not SlotState.FREE:
                raise RuntimeError(f"free queue returned slot {slot_id} in state {state.name}")
            generation = int(self._generations[slot_id]) + 1
            self._generations[slot_id] = generation
            self._states[slot_id] = int(SlotState.WRITING)
            self._timestamps[slot_id] = now
        return SlotLease(int(slot_id), generation, now)

    def _validate_lease(self, lease: SlotLease, expected: SlotState) -> None:
        state = SlotState(int(self._states[lease.slot_id]))
        generation = int(self._generations[lease.slot_id])
        if generation != lease.generation or state is not expected:
            raise RuntimeError(
                f"stale slot lease: slot={lease.slot_id}, generation={lease.generation}, "
                f"actual_generation={generation}, state={state.name}, expected={expected.name}"
            )

    def write(
        self,
        lease: SlotLease,
        values: Mapping[str, np.ndarray],
        *,
        item_count: int,
    ) -> None:
        with self._lock:
            self._validate_lease(lease, SlotState.WRITING)
        self._arena.write(lease.slot_id, values, item_count=item_count)

    def commit(
        self,
        lease: SlotLease,
        *,
        item_count: int,
        sequence_id: int,
        metadata: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> SlotMessage:
        if item_count <= 0 or item_count > self.max_items:
            raise ValueError("item_count outside channel capacity")
        message = SlotMessage(
            slot_id=lease.slot_id,
            generation=lease.generation,
            item_count=int(item_count),
            sequence_id=int(sequence_id),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._validate_lease(lease, SlotState.WRITING)
            self._states[lease.slot_id] = int(SlotState.READY)
            self._timestamps[lease.slot_id] = time.monotonic()
        try:
            self._ready_queue.put(message, timeout=timeout)
        except queue.Full as exc:
            with self._lock:
                if (
                    int(self._generations[lease.slot_id]) == lease.generation
                    and SlotState(int(self._states[lease.slot_id])) is SlotState.READY
                ):
                    self._states[lease.slot_id] = int(SlotState.FREE)
                    self._timestamps[lease.slot_id] = 0.0
                    self._free_queue.put(lease.slot_id)
            raise TimeoutError("timed out publishing a ready shared-memory slot") from exc
        return message

    def send(
        self,
        values: Mapping[str, np.ndarray],
        *,
        item_count: int,
        sequence_id: int,
        metadata: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> SlotMessage:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        lease = self.acquire(timeout=self._remaining(deadline))
        try:
            self.write(lease, values, item_count=item_count)
            return self.commit(
                lease,
                item_count=item_count,
                sequence_id=sequence_id,
                metadata=metadata,
                timeout=self._remaining(deadline),
            )
        except BaseException:
            try:
                self.abort(lease)
            except RuntimeError:
                pass
            raise

    def abort(self, lease: SlotLease) -> None:
        with self._lock:
            self._validate_lease(lease, SlotState.WRITING)
            self._states[lease.slot_id] = int(SlotState.FREE)
            self._timestamps[lease.slot_id] = 0.0
        self._free_queue.put(lease.slot_id)

    def receive(self, *, timeout: float | None = None) -> SlotMessage:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            if self._closed_event.is_set() and deadline is None:
                raise RuntimeError("shared tensor channel is closed")
            try:
                message: SlotMessage = self._ready_queue.get(timeout=self._remaining(deadline))
            except queue.Empty as exc:
                raise TimeoutError("timed out waiting for a ready shared-memory slot") from exc
            with self._lock:
                current_generation = int(self._generations[message.slot_id])
                current_state = SlotState(int(self._states[message.slot_id]))
                if (
                    current_generation != message.generation
                    or current_state is not SlotState.READY
                ):
                    if deadline is not None and self._remaining(deadline) == 0:
                        raise TimeoutError("only stale slot descriptors were available")
                    continue
                self._states[message.slot_id] = int(SlotState.READING)
                self._timestamps[message.slot_id] = time.monotonic()
                return message

    def read(self, message: SlotMessage, *, copy: bool = False) -> dict[str, np.ndarray]:
        with self._lock:
            lease = SlotLease(message.slot_id, message.generation, 0.0)
            self._validate_lease(lease, SlotState.READING)
        return self._arena.read(message.slot_id, item_count=message.item_count, copy=copy)

    def release(self, message: SlotMessage) -> None:
        with self._lock:
            lease = SlotLease(message.slot_id, message.generation, 0.0)
            self._validate_lease(lease, SlotState.READING)
            self._states[message.slot_id] = int(SlotState.FREE)
            self._timestamps[message.slot_id] = 0.0
        self._free_queue.put(message.slot_id)

    def reclaim_stale(self, *, max_age_seconds: float) -> list[int]:
        if not self.owner:
            raise RuntimeError("only the channel owner may reclaim stale slots")
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        now = time.monotonic()
        reclaimed: list[int] = []
        with self._lock:
            for slot_id in range(self.slot_count):
                state = SlotState(int(self._states[slot_id]))
                timestamp = float(self._timestamps[slot_id])
                if state is SlotState.FREE or timestamp <= 0:
                    continue
                if now - timestamp <= max_age_seconds:
                    continue
                self._generations[slot_id] = int(self._generations[slot_id]) + 1
                self._states[slot_id] = int(SlotState.FREE)
                self._timestamps[slot_id] = 0.0
                reclaimed.append(slot_id)
        for slot_id in reclaimed:
            self._free_queue.put(slot_id)
        return reclaimed

    def close(self) -> None:
        if self._local_closed:
            return
        self._local_closed = True
        self._arena.close()

    def shutdown(self) -> None:
        if not self.owner:
            raise RuntimeError("only the owner may shut down the channel")
        self._closed_event.set()

    def unlink(self) -> None:
        if not self.owner:
            raise RuntimeError("only the owner may unlink shared memory")
        self._arena.unlink()

    def __enter__(self) -> "SharedTensorChannel":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.owner:
            self.shutdown()
        self.close()
        if self.owner:
            self.unlink()
