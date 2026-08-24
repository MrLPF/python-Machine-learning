from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.context import BaseContext
import queue
import threading
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from forge_rl.transport import SharedTensorTreeArena, SharedTensorTreeDescriptor, TensorFieldSpec

from .fast_mailbox_inference import (
    FastMailboxInferenceClient,
    FastMailboxNodeLocalInferenceService,
)
from .inference import (
    DoubleBufferedPolicyReplica,
    InferenceFn,
    InferenceMetricsSnapshot,
    InferenceResponse,
)
from .mailbox_inference import MailboxInferenceEndpoint, MailboxTransportMetricsSnapshot
from .policy_registry import PolicyRegistry


@dataclass(frozen=True, slots=True)
class _ServiceEndpointDescriptor:
    actor_id: int
    max_items: int
    request: SharedTensorTreeDescriptor
    response: SharedTensorTreeDescriptor
    connection: Any


@dataclass(slots=True)
class _ChildEndpoint:
    actor_id: int
    max_items: int
    request: SharedTensorTreeArena
    response: SharedTensorTreeArena
    service_connection: Any

    @classmethod
    def attach(cls, descriptor: _ServiceEndpointDescriptor) -> "_ChildEndpoint":
        request = SharedTensorTreeArena.attach(descriptor.request)
        try:
            response = SharedTensorTreeArena.attach(descriptor.response)
        except BaseException:
            request.close()
            raise
        return cls(
            actor_id=descriptor.actor_id,
            max_items=descriptor.max_items,
            request=request,
            response=response,
            service_connection=descriptor.connection,
        )

    def close(self) -> None:
        self.request.close()
        self.response.close()
        try:
            self.service_connection.close()
        except OSError:
            pass


@dataclass(frozen=True, slots=True)
class _ProcessSpec:
    endpoints: tuple[_ServiceEndpointDescriptor, ...]
    module_type: type[nn.Module]
    module_args: tuple[Any, ...]
    module_kwargs: dict[str, Any]
    infer_fn: InferenceFn
    state_dict: dict[str, torch.Tensor]
    policy_version: int
    max_batch_items: int
    min_batch_items: int
    max_wait_ms: float
    idle_wait_ms: float
    device: str
    amp_dtype: str | None


@dataclass(frozen=True, slots=True)
class _ReadyReply:
    policy_version: int
    error: str = ""


@dataclass(frozen=True, slots=True)
class _UpdateCommand:
    version: int
    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class _MetricsCommand:
    token: int


@dataclass(frozen=True, slots=True)
class _StopCommand:
    token: int


@dataclass(frozen=True, slots=True)
class _CommandReply:
    token: int
    kind: str
    policy_version: int
    inference: InferenceMetricsSnapshot | None = None
    transport: MailboxTransportMetricsSnapshot | None = None
    error: str = ""


def _process_main(spec: _ProcessSpec, control: Any, replies: Any, ready: Any) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    endpoints: list[_ChildEndpoint] = []
    service: FastMailboxNodeLocalInferenceService | None = None
    startup_complete = False
    try:
        endpoints = [_ChildEndpoint.attach(descriptor) for descriptor in spec.endpoints]
        device = torch.device(spec.device)
        amp_dtype = None if spec.amp_dtype is None else getattr(torch, spec.amp_dtype)

        def factory() -> nn.Module:
            return spec.module_type(*spec.module_args, **spec.module_kwargs)

        registry = PolicyRegistry(history_size=2)
        snapshot = registry.publish(spec.state_dict, version=spec.policy_version)
        service = FastMailboxNodeLocalInferenceService(
            endpoints=endpoints,
            replica=DoubleBufferedPolicyReplica(
                factory,
                device=device,
                amp_dtype=amp_dtype,
            ),
            infer_fn=spec.infer_fn,
            max_batch_items=spec.max_batch_items,
            min_batch_items=spec.min_batch_items,
            max_wait_ms=spec.max_wait_ms,
            idle_wait_ms=spec.idle_wait_ms,
        )
        service.refresh_policy(snapshot)
        service.start()
        replies.put(_ReadyReply(service.replica.active_version))
        ready.set()
        startup_complete = True

        while True:
            command = control.get()
            if isinstance(command, _UpdateCommand):
                try:
                    snapshot = registry.publish(
                        command.state_dict,
                        version=command.version,
                    )
                    activated = service.refresh_policy(snapshot)
                    replies.put(
                        _CommandReply(
                            token=command.version,
                            kind="update",
                            policy_version=activated,
                        )
                    )
                except BaseException as error:
                    replies.put(
                        _CommandReply(
                            token=command.version,
                            kind="update",
                            policy_version=service.replica.active_version,
                            error=f"{type(error).__name__}: {error}",
                        )
                    )
            elif isinstance(command, _MetricsCommand):
                last_error = service.last_error
                replies.put(
                    _CommandReply(
                        token=command.token,
                        kind="metrics",
                        policy_version=service.replica.active_version,
                        inference=service.metrics(),
                        transport=service.transport_metrics(),
                        error=(
                            ""
                            if last_error is None
                            else f"{type(last_error).__name__}: {last_error}"
                        ),
                    )
                )
            elif isinstance(command, _StopCommand):
                replies.put(
                    _CommandReply(
                        token=command.token,
                        kind="stop",
                        policy_version=service.replica.active_version,
                    )
                )
                return
            else:
                raise RuntimeError(f"unsupported process mailbox command: {type(command)!r}")
    except BaseException as error:
        if not startup_complete:
            replies.put(_ReadyReply(-1, f"{type(error).__name__}: {error}"))
            ready.set()
        else:
            replies.put(
                _CommandReply(
                    token=-1,
                    kind="fatal",
                    policy_version=(
                        -1 if service is None else service.replica.active_version
                    ),
                    error=f"{type(error).__name__}: {error}",
                )
            )
    finally:
        if service is not None:
            try:
                service.stop(timeout=10.0)
            except BaseException:
                pass
        for endpoint in endpoints:
            endpoint.close()


class ProcessMailboxInferenceRuntime:
    """Process-isolated predictor with shared-memory synchronous Actor mailboxes.

    Actor Python remains in the parent process. Descriptor collection, batch assembly, PyTorch
    inference and response publication run in a dedicated child process, eliminating the GIL
    contention that limited the in-process mailbox at target Actor counts. Tensor payloads remain
    in shared memory; control queues carry only infrequent policy, metrics and lifecycle messages.
    """

    def __init__(
        self,
        *,
        actor_count: int,
        max_items: int,
        request_fields: Mapping[str, TensorFieldSpec | tuple[tuple[int, ...], Any]],
        response_fields: Mapping[str, TensorFieldSpec | tuple[tuple[int, ...], Any]],
        module_type: type[nn.Module],
        module_args: Sequence[Any] = (),
        module_kwargs: Mapping[str, Any] | None = None,
        infer_fn: InferenceFn,
        state_dict: Mapping[str, torch.Tensor],
        policy_version: int = 0,
        max_batch_items: int = 128,
        min_batch_items: int = 1,
        max_wait_ms: float = 2.0,
        idle_wait_ms: float = 50.0,
        device: torch.device | str = "cpu",
        amp_dtype: torch.dtype | None = None,
        copy_outputs: bool = False,
        mp_context: BaseContext | None = None,
    ) -> None:
        if actor_count <= 0:
            raise ValueError("actor_count must be positive")
        selected_device = torch.device(device)
        if selected_device.type not in {"cpu", "cuda"}:
            raise ValueError("process mailbox device must be cpu or cuda")
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA process mailbox requested but CUDA is unavailable")
        if amp_dtype not in {None, torch.float16, torch.bfloat16}:
            raise ValueError("amp_dtype must be float16, bfloat16 or None")
        if selected_device.type != "cuda" and amp_dtype is not None:
            raise ValueError("AMP process mailbox is supported only on CUDA")

        self.context = mp_context or mp.get_context("spawn")
        self.endpoints = [
            MailboxInferenceEndpoint.create(
                actor_id=actor_id,
                max_items=max_items,
                request_fields=request_fields,
                response_fields=response_fields,
                mp_context=self.context,
            )
            for actor_id in range(actor_count)
        ]
        self.clients = [
            FastMailboxInferenceClient(endpoint, copy_outputs=copy_outputs)
            for endpoint in self.endpoints
        ]
        descriptors = tuple(
            _ServiceEndpointDescriptor(
                endpoint.actor_id,
                endpoint.max_items,
                endpoint.request.descriptor,
                endpoint.response.descriptor,
                endpoint.service_connection,
            )
            for endpoint in self.endpoints
        )
        state = {
            name: value.detach().cpu().clone(memory_format=torch.preserve_format)
            for name, value in state_dict.items()
        }
        self._spec = _ProcessSpec(
            endpoints=descriptors,
            module_type=module_type,
            module_args=tuple(module_args),
            module_kwargs=dict(module_kwargs or {}),
            infer_fn=infer_fn,
            state_dict=state,
            policy_version=int(policy_version),
            max_batch_items=int(max_batch_items),
            min_batch_items=int(min_batch_items),
            max_wait_ms=float(max_wait_ms),
            idle_wait_ms=float(idle_wait_ms),
            device=str(selected_device),
            amp_dtype=(
                None if amp_dtype is None else str(amp_dtype).removeprefix("torch.")
            ),
        )
        self._control = self.context.Queue()
        self._replies = self.context.Queue()
        self._ready = self.context.Event()
        self._process = self.context.Process(
            target=_process_main,
            args=(self._spec, self._control, self._replies, self._ready),
            name="forge-mailbox-predictor",
            daemon=True,
        )
        self._control_lock = threading.Lock()
        self._token = 0
        self._started = False
        self._closed = False
        self._policy_version = int(policy_version)

    @property
    def policy_version(self) -> int:
        return self._policy_version

    @property
    def process_pid(self) -> int | None:
        return self._process.pid

    def start(self, *, timeout: float = 20.0) -> None:
        if self._started:
            raise RuntimeError("process mailbox runtime is already started")
        self._process.start()
        if not self._ready.wait(timeout):
            self._terminate()
            raise TimeoutError("process mailbox predictor did not become ready")
        try:
            reply = self._replies.get(timeout=timeout)
        except queue.Empty as error:
            self._terminate()
            raise TimeoutError("process mailbox predictor omitted its ready reply") from error
        if not isinstance(reply, _ReadyReply):
            self._terminate()
            raise RuntimeError("unexpected process mailbox startup reply")
        if reply.error:
            self._terminate()
            raise RuntimeError(reply.error)
        if reply.policy_version != self._policy_version:
            self._terminate()
            raise RuntimeError("process mailbox startup policy version mismatch")
        self._started = True

    def infer(
        self,
        actor_id: int,
        inputs: Mapping[str, np.ndarray],
        *,
        min_policy_version: int = -1,
        timeout: float = 15.0,
    ) -> InferenceResponse:
        if not self._started or self._closed:
            raise RuntimeError("process mailbox runtime is not active")
        if actor_id < 0 or actor_id >= len(self.clients):
            raise IndexError("actor_id outside process mailbox runtime")
        return self.clients[actor_id].infer(
            inputs,
            min_policy_version=min_policy_version,
            timeout=timeout,
        )

    def _next_token(self) -> int:
        self._token += 1
        return self._token

    def _command(self, command: Any, *, timeout: float) -> _CommandReply:
        if not self._started or self._closed:
            raise RuntimeError("process mailbox runtime is not active")
        with self._control_lock:
            if not self._process.is_alive():
                raise RuntimeError(
                    f"process mailbox predictor exited with code {self._process.exitcode}"
                )
            self._control.put(command, timeout=timeout)
            try:
                reply = self._replies.get(timeout=timeout)
            except queue.Empty as error:
                raise TimeoutError("timed out waiting for process mailbox control reply") from error
            if not isinstance(reply, _CommandReply):
                raise RuntimeError("unexpected process mailbox control reply")
            if reply.kind == "fatal":
                raise RuntimeError(reply.error)
            return reply

    def update_policy(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        version: int,
        timeout: float = 30.0,
    ) -> int:
        selected = int(version)
        if selected <= self._policy_version:
            raise ValueError("policy version must increase monotonically")
        state = {
            name: value.detach().cpu().clone(memory_format=torch.preserve_format)
            for name, value in state_dict.items()
        }
        reply = self._command(
            _UpdateCommand(selected, state),
            timeout=timeout,
        )
        if reply.kind != "update" or reply.token != selected:
            raise RuntimeError("process mailbox policy update reply mismatch")
        if reply.error:
            raise RuntimeError(reply.error)
        if reply.policy_version != selected:
            raise RuntimeError("process mailbox activated an unexpected policy version")
        self._policy_version = selected
        return selected

    def _metrics_reply(self, *, timeout: float = 10.0) -> _CommandReply:
        token = self._next_token()
        reply = self._command(_MetricsCommand(token), timeout=timeout)
        if reply.kind != "metrics" or reply.token != token:
            raise RuntimeError("process mailbox metrics reply mismatch")
        if reply.error:
            raise RuntimeError(reply.error)
        if reply.inference is None or reply.transport is None:
            raise RuntimeError("process mailbox metrics reply is incomplete")
        return reply

    def metrics(self, *, timeout: float = 10.0) -> InferenceMetricsSnapshot:
        reply = self._metrics_reply(timeout=timeout)
        assert reply.inference is not None
        return reply.inference

    def metrics_pair(
        self, *, timeout: float = 10.0
    ) -> tuple[InferenceMetricsSnapshot, MailboxTransportMetricsSnapshot]:
        reply = self._metrics_reply(timeout=timeout)
        assert reply.inference is not None and reply.transport is not None
        return reply.inference, reply.transport

    def close(self, *, timeout: float = 15.0) -> None:
        if self._closed:
            return
        try:
            if self._started and self._process.is_alive():
                token = self._next_token()
                try:
                    reply = self._command(_StopCommand(token), timeout=timeout)
                    if reply.kind != "stop" or reply.token != token:
                        raise RuntimeError("process mailbox stop reply mismatch")
                except BaseException:
                    self._terminate()
                    raise
            self._process.join(timeout)
            if self._process.is_alive():
                self._terminate()
                raise TimeoutError("process mailbox predictor did not stop")
        finally:
            self._closed = True
            for client in self.clients:
                client.close()
            for endpoint in self.endpoints:
                endpoint.close()
                endpoint.unlink()
            for channel in (self._control, self._replies):
                try:
                    channel.close()
                    channel.join_thread()
                except (AttributeError, ValueError):
                    pass

    def _terminate(self) -> None:
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(5.0)
        if self._process.is_alive() and hasattr(self._process, "kill"):
            self._process.kill()
            self._process.join(5.0)
