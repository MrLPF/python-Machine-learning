from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
import queue
import time
import traceback
from typing import Any, Callable, Mapping, Sequence


SupervisedTarget = Callable[..., None]


class WorkerStartError(RuntimeError):
    """Raised when a supervised component exits before reporting readiness."""


@dataclass(frozen=True, slots=True)
class RestartRecord:
    generation: int
    pid: int
    exitcode: int | None
    reason: str
    started_at: float
    stopped_at: float


def _supervised_entry(
    target: SupervisedTarget,
    generation: int,
    ready_event: Any,
    stop_event: Any,
    status_queue: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    try:
        target(
            generation,
            ready_event,
            stop_event,
            status_queue,
            *args,
            **kwargs,
        )
    except BaseException:
        status_queue.put(
            {
                "type": "worker_error",
                "generation": generation,
                "traceback": traceback.format_exc(),
            }
        )
        raise


class RestartableProcess:
    """Small process supervisor used by M2 component-restart recovery.

    The target receives ``generation``, ``ready_event``, ``stop_event`` and ``status_queue`` before
    its configured arguments. A replacement process always receives a strictly newer generation.
    The supervisor intentionally does not hide component state: Actors rely on request idempotency,
    inference replicas reload the latest policy, and learner groups restore a completed checkpoint.
    """

    def __init__(
        self,
        target: SupervisedTarget,
        *,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
        name: str = "forge-rl-component",
        start_method: str = "spawn",
        graceful_timeout: float = 5.0,
    ) -> None:
        if graceful_timeout <= 0:
            raise ValueError("graceful_timeout must be positive")
        self.target = target
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.name = str(name)
        self.graceful_timeout = float(graceful_timeout)
        self._context: BaseContext = mp.get_context(start_method)
        self._process: BaseProcess | None = None
        self._ready_event: Any | None = None
        self._stop_event: Any | None = None
        self._status_queue: Any | None = None
        self._generation = -1
        self._started_at = 0.0
        self._history: list[RestartRecord] = []

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def pid(self) -> int | None:
        process = self._process
        return None if process is None else process.pid

    @property
    def alive(self) -> bool:
        process = self._process
        return process is not None and process.is_alive()

    @property
    def restart_count(self) -> int:
        return max(0, self._generation)

    @property
    def history(self) -> tuple[RestartRecord, ...]:
        return tuple(self._history)

    def start(self) -> int:
        if self.alive:
            raise RuntimeError("supervised process is already running")
        self._generation += 1
        self._ready_event = self._context.Event()
        self._stop_event = self._context.Event()
        self._status_queue = self._context.Queue()
        self._process = self._context.Process(
            target=_supervised_entry,
            args=(
                self.target,
                self._generation,
                self._ready_event,
                self._stop_event,
                self._status_queue,
                self.args,
                self.kwargs,
            ),
            name=f"{self.name}-g{self._generation}",
            daemon=False,
        )
        self._started_at = time.monotonic()
        self._process.start()
        return self._generation

    def wait_ready(self, *, timeout: float = 10.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        process = self._require_process()
        ready_event = self._ready_event
        assert ready_event is not None
        deadline = time.monotonic() + timeout
        while not ready_event.is_set():
            if not process.is_alive():
                process.join(timeout=0.1)
                detail = self._drain_error_message()
                raise WorkerStartError(
                    f"{self.name} generation {self._generation} exited before readiness: {detail}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"timed out waiting for {self.name} generation {self._generation}"
                )
            ready_event.wait(min(0.05, remaining))

    def receive_status(self, *, timeout: float = 5.0) -> Any:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        status_queue = self._status_queue
        if status_queue is None:
            raise RuntimeError("supervised process has not been started")
        try:
            return status_queue.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError(f"no status received from {self.name}") from error

    def inject_failure(self, *, reason: str = "injected_failure") -> RestartRecord:
        process = self._require_process()
        if process.is_alive():
            process.terminate()
            process.join(timeout=self.graceful_timeout)
        if process.is_alive():
            process.kill()
            process.join(timeout=self.graceful_timeout)
        if process.is_alive():
            raise RuntimeError(f"failed to terminate {self.name}")
        return self._record_stop(reason)

    def restart(self, *, timeout: float = 10.0, reason: str = "injected_failure") -> int:
        if self._process is not None and self._process.is_alive():
            self.inject_failure(reason=reason)
        generation = self.start()
        self.wait_ready(timeout=timeout)
        return generation

    def stop(self, *, reason: str = "graceful_stop") -> RestartRecord | None:
        process = self._process
        if process is None:
            return None
        stop_event = self._stop_event
        if process.is_alive() and stop_event is not None:
            stop_event.set()
            process.join(timeout=self.graceful_timeout)
        if process.is_alive():
            process.terminate()
            process.join(timeout=self.graceful_timeout)
        if process.is_alive():
            process.kill()
            process.join(timeout=self.graceful_timeout)
        if process.is_alive():
            raise RuntimeError(f"failed to stop {self.name}")
        return self._record_stop(reason)

    def close(self) -> None:
        self.stop()
        status_queue = self._status_queue
        if status_queue is not None:
            status_queue.close()
            status_queue.join_thread()
        self._status_queue = None
        self._ready_event = None
        self._stop_event = None
        self._process = None

    def __enter__(self) -> "RestartableProcess":
        self.start()
        self.wait_ready()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback_value: object) -> None:
        del exc_type, exc, traceback_value
        self.close()

    def _require_process(self) -> BaseProcess:
        process = self._process
        if process is None:
            raise RuntimeError("supervised process has not been started")
        return process

    def _drain_error_message(self) -> str:
        status_queue = self._status_queue
        if status_queue is None:
            return "no status queue"
        messages: list[str] = []
        while True:
            try:
                message = status_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(message, dict) and message.get("type") == "worker_error":
                messages.append(str(message.get("traceback", "worker error")))
        return "\n".join(messages) if messages else "no worker error payload"

    def _record_stop(self, reason: str) -> RestartRecord:
        process = self._require_process()
        record = RestartRecord(
            generation=self._generation,
            pid=int(process.pid or -1),
            exitcode=process.exitcode,
            reason=str(reason),
            started_at=self._started_at,
            stopped_at=time.monotonic(),
        )
        if not self._history or self._history[-1].generation != record.generation:
            self._history.append(record)
        return record
