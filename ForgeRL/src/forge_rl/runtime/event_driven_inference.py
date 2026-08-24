from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.context import BaseContext
import queue
import threading
import time
from typing import Any, Sequence

from forge_rl.transport.shared_channel import SharedTensorChannel, SlotMessage, SlotState

from .inference import (
    DoubleBufferedPolicyReplica,
    InferenceFn,
    SharedInferenceEndpoint,
    _PendingRequest,
)
from .optimized_inference import (
    NodeLocalInferenceService as PollingNodeLocalInferenceService,
)


@dataclass(frozen=True, slots=True)
class EventDrivenInferenceMetricsSnapshot:
    """Counters for the shared ready-descriptor notification path."""

    collector_mode: str
    notification_tokens: int
    stale_notifications: int
    invalid_actor_notifications: int
    notification_batches: int
    max_notification_batch: int


class _EventMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._notification_tokens = 0
        self._stale_notifications = 0
        self._invalid_actor_notifications = 0
        self._notification_batches = 0
        self._max_notification_batch = 0

    def record_batch(self, *, token_count: int) -> None:
        with self._lock:
            self._notification_tokens += int(token_count)
            self._notification_batches += 1
            self._max_notification_batch = max(self._max_notification_batch, int(token_count))

    def record_stale(self) -> None:
        with self._lock:
            self._stale_notifications += 1

    def record_invalid_actor(self) -> None:
        with self._lock:
            self._invalid_actor_notifications += 1

    def snapshot(self) -> EventDrivenInferenceMetricsSnapshot:
        with self._lock:
            return EventDrivenInferenceMetricsSnapshot(
                collector_mode="shared-ready-descriptor-queue",
                notification_tokens=self._notification_tokens,
                stale_notifications=self._stale_notifications,
                invalid_actor_notifications=self._invalid_actor_notifications,
                notification_batches=self._notification_batches,
                max_notification_batch=self._max_notification_batch,
            )


@dataclass(frozen=True, slots=True)
class _ReadyToken:
    actor_id: int
    message: SlotMessage


class _NotifyingReadyQueue:
    """Route request descriptors to one node-level queue instead of per-endpoint polling.

    The original endpoint queue is retained only so it can be restored when the service stops.
    Tensor payloads remain in each endpoint's shared-memory slots; only ``_ReadyToken`` objects
    traverse the node-level queue.
    """

    def __init__(self, original: Any, notification_queue: Any, actor_id: int) -> None:
        self.original = original
        self.notification_queue = notification_queue
        self.actor_id = int(actor_id)

    def put(
        self,
        message: SlotMessage,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        token = _ReadyToken(self.actor_id, message)
        if timeout is None:
            self.notification_queue.put(token, block=block)
        else:
            self.notification_queue.put(token, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        if timeout is None:
            return self.original.get(block=block)
        return self.original.get(block=block, timeout=timeout)

    def put_nowait(self, message: SlotMessage) -> None:
        self.put(message, block=False)

    def get_nowait(self) -> Any:
        return self.original.get(block=False)


class NodeLocalInferenceService(PollingNodeLocalInferenceService):
    """Event-driven M1 node-local inference service.

    A single node-level descriptor queue replaces the round-robin empty polling used by the
    previous optimized service. Endpoint request queues are wrapped before actors are started, so
    existing :class:`SharedInferenceClient` instances need no API change. The old polling service
    remains available as ``PollingNodeLocalInferenceService`` for controlled comparison.
    """

    def __init__(
        self,
        *,
        endpoints: Sequence[SharedInferenceEndpoint],
        replica: DoubleBufferedPolicyReplica,
        infer_fn: InferenceFn,
        max_batch_items: int,
        min_batch_items: int = 1,
        max_wait_ms: float = 2.0,
        response_timeout_seconds: float = 5.0,
        notification_drain_limit: int = 64,
        notification_wait_ms: float = 50.0,
        collector_poll_ms: float | None = None,
        max_drain_per_endpoint: int | None = None,
        mp_context: BaseContext | None = None,
    ) -> None:
        super().__init__(
            endpoints=endpoints,
            replica=replica,
            infer_fn=infer_fn,
            max_batch_items=max_batch_items,
            min_batch_items=min_batch_items,
            max_wait_ms=max_wait_ms,
            response_timeout_seconds=response_timeout_seconds,
        )
        if notification_drain_limit <= 0:
            raise ValueError("notification_drain_limit must be positive")
        if notification_wait_ms <= 0:
            raise ValueError("notification_wait_ms must be positive")
        # These two keyword arguments belonged to the preceding polling implementation.
        # Accept them during the M1 migration so existing callers and regression tests do not
        # break when the default service changes. The event-driven collector does not poll,
        # while the old per-endpoint drain cap maps to the closest node-level burst limit.
        if collector_poll_ms is not None and collector_poll_ms <= 0:
            raise ValueError("collector_poll_ms must be positive when provided")
        if max_drain_per_endpoint is not None:
            if max_drain_per_endpoint <= 0:
                raise ValueError("max_drain_per_endpoint must be positive when provided")
            notification_drain_limit = max(
                int(notification_drain_limit), int(max_drain_per_endpoint)
            )
        context = mp_context or mp.get_context()
        self.compatibility_collector_poll_ms = (
            None if collector_poll_ms is None else float(collector_poll_ms)
        )
        self.notification_drain_limit = int(notification_drain_limit)
        self.notification_wait_seconds = float(notification_wait_ms) / 1000.0
        self._notification_queue = context.Queue()
        self._event_metrics = _EventMetrics()
        self._endpoints_by_actor = {endpoint.actor_id: endpoint for endpoint in self.endpoints}
        self._original_ready_queues: dict[int, Any] = {}
        self._notifiers_installed = False
        self._install_notifiers()

    def _install_notifiers(self) -> None:
        if self._notifiers_installed:
            return
        installed: list[SharedInferenceEndpoint] = []
        try:
            for endpoint in self.endpoints:
                original = endpoint.request._ready_queue
                self._original_ready_queues[endpoint.actor_id] = original
                endpoint.request._ready_queue = _NotifyingReadyQueue(
                    original,
                    self._notification_queue,
                    endpoint.actor_id,
                )
                installed.append(endpoint)
        except BaseException:
            for endpoint in installed:
                endpoint.request._ready_queue = self._original_ready_queues[endpoint.actor_id]
            self._original_ready_queues.clear()
            raise
        self._notifiers_installed = True

    def _restore_notifiers(self) -> None:
        if not self._notifiers_installed:
            return
        for endpoint in self.endpoints:
            original = self._original_ready_queues.get(endpoint.actor_id)
            if original is not None:
                endpoint.request._ready_queue = original
        self._original_ready_queues.clear()
        self._notifiers_installed = False

    def event_metrics(self) -> EventDrivenInferenceMetricsSnapshot:
        return self._event_metrics.snapshot()

    @staticmethod
    def _claim(channel: SharedTensorChannel, message: SlotMessage) -> bool:
        if message.slot_id < 0 or message.slot_id >= channel.slot_count:
            return False
        with channel._lock:
            generation = int(channel._generations[message.slot_id])
            state = SlotState(int(channel._states[message.slot_id]))
            if generation != message.generation or state is not SlotState.READY:
                return False
            channel._states[message.slot_id] = int(SlotState.READING)
            channel._timestamps[message.slot_id] = time.monotonic()
            return True

    def _submit_token(self, token: _ReadyToken) -> bool:
        endpoint = self._endpoints_by_actor.get(int(token.actor_id))
        if endpoint is None:
            self._event_metrics.record_invalid_actor()
            raise RuntimeError(f"notification references unknown actor {token.actor_id}")
        if not self._claim(endpoint.request, token.message):
            self._event_metrics.record_stale()
            return False
        metadata_actor = int(token.message.metadata.get("actor_id", token.actor_id))
        if metadata_actor != token.actor_id:
            self._event_metrics.record_invalid_actor()
            endpoint.request.release(token.message)
            raise RuntimeError(
                f"notification actor {token.actor_id} disagrees with metadata {metadata_actor}"
            )
        pending = _PendingRequest(endpoint, token.message, time.monotonic())
        try:
            self._batcher.submit(
                pending,
                item_count=token.message.item_count,
                timeout=0.5,
            )
        except BaseException:
            try:
                endpoint.request.release(token.message)
            except RuntimeError:
                pass
            raise
        return True

    def _collector_loop(self) -> None:
        while not self._stop.is_set():
            try:
                first: _ReadyToken = self._notification_queue.get(
                    timeout=self.notification_wait_seconds
                )
            except queue.Empty:
                self._optimization_metrics.record_poll(progressed=False)
                continue
            except (OSError, ValueError) as error:
                if self._stop.is_set():
                    return
                self._last_error = error
                self._metrics.record_error()
                return

            tokens = [first]
            while len(tokens) < self.notification_drain_limit:
                try:
                    tokens.append(self._notification_queue.get_nowait())
                except queue.Empty:
                    break
                except (OSError, ValueError) as error:
                    self._last_error = error
                    self._metrics.record_error()
                    return

            progressed = False
            try:
                for token in tokens:
                    progressed = self._submit_token(token) or progressed
            except BaseException as error:
                if self._stop.is_set():
                    return
                self._last_error = error
                self._metrics.record_error()
                return
            finally:
                self._event_metrics.record_batch(token_count=len(tokens))
                self._optimization_metrics.record_poll(progressed=progressed)

    def stop(self, *, timeout: float = 5.0) -> None:
        try:
            super().stop(timeout=timeout)
        finally:
            self._restore_notifiers()
            try:
                self._notification_queue.close()
                self._notification_queue.join_thread()
            except (AttributeError, ValueError):
                pass
