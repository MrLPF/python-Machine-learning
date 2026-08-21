from __future__ import annotations

import time
from typing import Mapping, Sequence

import numpy as np

from .inference import InferenceResponse
from .mailbox_inference import (
    MailboxInferenceClient,
    MailboxNodeLocalInferenceService,
    _MailboxRequest,
    _REQUEST,
    _RESPONSE,
)


class FastMailboxInferenceClient(MailboxInferenceClient):
    """Mailbox client with persistent tensor views and reusable binary headers."""

    def __init__(self, endpoint, *, copy_outputs: bool = False) -> None:
        super().__init__(endpoint, copy_outputs=copy_outputs)
        self._request_views = self._request.read(
            0, item_count=self.max_items, copy=False
        )
        self._response_views = self._response.read(
            0, item_count=self.max_items, copy=False
        )
        self._request_names = tuple(sorted(self._request_views))
        self._request_header = bytearray(_REQUEST.size)

    def infer(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        min_policy_version: int = -1,
        timeout: float = 5.0,
    ) -> InferenceResponse:
        if self._broken:
            raise RuntimeError("mailbox client is unusable after a failed exchange")
        if tuple(sorted(inputs)) != self._request_names:
            raise ValueError("input tensor fields differ from mailbox schema")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        item_count = -1
        prepared: list[tuple[np.ndarray, np.ndarray]] = []
        for name in self._request_names:
            target = self._request_views[name]
            source = np.asarray(inputs[name], dtype=target.dtype)
            if source.ndim == 0:
                raise ValueError(f"input field {name!r} has no leading dimension")
            if item_count < 0:
                item_count = int(source.shape[0])
            elif int(source.shape[0]) != item_count:
                raise ValueError("input tensors have inconsistent leading dimensions")
            if source.shape[1:] != target.shape[1:]:
                raise ValueError(
                    f"input field {name!r} expected trailing shape {target.shape[1:]}"
                )
            prepared.append((target, source))
        if item_count <= 0 or item_count > self.max_items:
            raise ValueError("item_count outside mailbox capacity")

        with self._lock:
            sequence_id = self._sequence
            self._sequence += 1
            submitted_at = time.monotonic()
            for target, source in prepared:
                np.copyto(target[:item_count], source, casting="no")
            _REQUEST.pack_into(
                self._request_header,
                0,
                sequence_id,
                item_count,
                int(min_policy_version),
                submitted_at,
            )
            self._connection.send_bytes(self._request_header)
            if not self._connection.poll(timeout):
                self._broken = True
                raise TimeoutError("timed out waiting for mailbox response")
            payload = self._connection.recv_bytes()
            if len(payload) < _RESPONSE.size:
                self._broken = True
                raise RuntimeError("truncated mailbox response")
            returned, policy_version, _completed, error_size = _RESPONSE.unpack_from(payload)
            error = payload[_RESPONSE.size :]
            if returned != sequence_id or len(error) != error_size:
                self._broken = True
                raise RuntimeError("mailbox response identity or length mismatch")
            if error:
                raise RuntimeError(error.decode("utf-8", errors="replace"))
            if policy_version < min_policy_version:
                self._broken = True
                raise RuntimeError("mailbox response used a stale policy")
            outputs = {
                name: value[:item_count].copy() if self.copy_outputs else value[:item_count]
                for name, value in self._response_views.items()
            }
            return InferenceResponse(
                outputs=outputs,
                policy_version=int(policy_version),
                sequence_id=sequence_id,
                latency_seconds=max(0.0, time.monotonic() - submitted_at),
            )


class FastMailboxNodeLocalInferenceService(MailboxNodeLocalInferenceService):
    """Prebound mailbox views and two-phase response release."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._request_views = {
            endpoint.actor_id: endpoint.request.read(
                0, item_count=endpoint.max_items, copy=False
            )
            for endpoint in self.endpoints
        }
        self._response_views = {
            endpoint.actor_id: endpoint.response.read(
                0, item_count=endpoint.max_items, copy=False
            )
            for endpoint in self.endpoints
        }
        self._response_headers = {
            endpoint.actor_id: bytearray(_RESPONSE.size)
            for endpoint in self.endpoints
        }

    def _execute(self, requests: Sequence[_MailboxRequest]) -> None:
        views = [
            {
                name: value[: request.item_count]
                for name, value in self._request_views[
                    request.endpoint.actor_id
                ].items()
            }
            for request in requests
        ]
        merged, total_items, copied_bytes, zero_copy, allocated = self._assembler.assemble(
            views
        )
        requested_min = max(request.min_policy_version for request in requests)
        if self.replica.active_version < requested_min:
            raise RuntimeError("active policy is older than requested")
        outputs, policy_version = self.replica.infer(merged, self.infer_fn)
        completed = time.monotonic()
        offset = 0
        latencies: list[float] = []

        for request in requests:
            stop = offset + request.item_count
            response_views = self._response_views[request.endpoint.actor_id]
            if tuple(sorted(response_views)) != tuple(sorted(outputs)):
                raise RuntimeError("inference output fields differ from mailbox schema")
            for name, output in outputs.items():
                target = response_views[name]
                source = output[offset:stop]
                if source.shape[1:] != target.shape[1:]:
                    raise RuntimeError(
                        f"output field {name!r} differs from mailbox schema"
                    )
                np.copyto(target[: request.item_count], source, casting="no")
            latencies.append(max(0.0, completed - request.submitted_at))
            offset = stop
        if offset != total_items:
            raise RuntimeError("mailbox response split did not consume the batch")

        for request in requests:
            header = self._response_headers[request.endpoint.actor_id]
            _RESPONSE.pack_into(
                header,
                0,
                request.sequence_id,
                policy_version,
                completed,
                0,
            )
            request.endpoint.service_connection.send_bytes(header)

        with self._transport_lock:
            self._response_signals += len(requests)
            self._assembly_allocations += int(allocated)
            self._assembly_copy_bytes += int(copied_bytes)
            self._reused_buffer_batches += int(not zero_copy and not allocated)
            self._single_request_zero_copy_batches += int(zero_copy)
        self._metrics.record_success(
            request_messages=len(requests),
            items=total_items,
            latencies=latencies,
        )
