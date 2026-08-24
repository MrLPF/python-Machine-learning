from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import socket
import socketserver
import struct
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlparse

import numpy as np
import torch

from .experience import ExperienceItem, OnPolicyExperienceQueue
from .policy_registry import PolicyRegistry, PolicySnapshot
from .transition import TransitionBatch

_MAGIC = b"FRL2"
_PROTOCOL_VERSION = 1
_FRAME_HEADER = struct.Struct("!4sBII")
_DEFAULT_MAX_METADATA_BYTES = 1 << 20
_DEFAULT_MAX_PAYLOAD_BYTES = 256 << 20
_DEFAULT_SOCKET_TIMEOUT = 30.0


class DataChannelError(RuntimeError):
    """A deterministic direct-data-channel error returned to a client."""

    def __init__(self, *, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(f"data channel {code}: {message}")
        self.code = str(code)
        self.message = str(message)
        self.retryable = bool(retryable)


class _RequestError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.retryable = bool(retryable)


@dataclass(frozen=True, slots=True)
class DataChannelMetricsSnapshot:
    accepted_requests: int
    rejected_requests: int
    received_payload_bytes: int
    accepted_rows: int
    published_versions: int


class _DataChannelMetrics:
    def __init__(self) -> None:
        self._accepted_requests = 0
        self._rejected_requests = 0
        self._received_payload_bytes = 0
        self._accepted_rows = 0
        self._published_versions = 0
        self._lock = threading.Lock()

    def accepted(self, *, payload_bytes: int, rows: int = 0, versions: int = 0) -> None:
        with self._lock:
            self._accepted_requests += 1
            self._received_payload_bytes += int(payload_bytes)
            self._accepted_rows += int(rows)
            self._published_versions += int(versions)

    def rejected(self, *, payload_bytes: int) -> None:
        with self._lock:
            self._rejected_requests += 1
            self._received_payload_bytes += int(payload_bytes)

    def snapshot(self) -> DataChannelMetricsSnapshot:
        with self._lock:
            return DataChannelMetricsSnapshot(
                accepted_requests=self._accepted_requests,
                rejected_requests=self._rejected_requests,
                received_payload_bytes=self._received_payload_bytes,
                accepted_rows=self._accepted_rows,
                published_versions=self._published_versions,
            )


def _read_exact(
    connection: socket.socket,
    size: int,
    *,
    allow_clean_eof: bool = False,
) -> bytes | None:
    if size < 0:
        raise ValueError("size must be non-negative")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            if allow_clean_eof and not chunks:
                return None
            raise EOFError("connection closed in the middle of a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_frame(
    connection: socket.socket,
    metadata: Mapping[str, Any],
    payload: bytes = b"",
) -> None:
    encoded_metadata = json.dumps(
        dict(metadata),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded_metadata) > _DEFAULT_MAX_METADATA_BYTES:
        raise ValueError("frame metadata exceeds 1 MiB")
    header = _FRAME_HEADER.pack(
        _MAGIC,
        _PROTOCOL_VERSION,
        len(encoded_metadata),
        len(payload),
    )
    connection.sendall(header)
    connection.sendall(encoded_metadata)
    if payload:
        connection.sendall(payload)


def _receive_frame(
    connection: socket.socket,
    *,
    max_metadata_bytes: int,
    max_payload_bytes: int,
) -> tuple[dict[str, Any], bytes] | None:
    raw_header = _read_exact(
        connection,
        _FRAME_HEADER.size,
        allow_clean_eof=True,
    )
    if raw_header is None:
        return None
    magic, version, metadata_size, payload_size = _FRAME_HEADER.unpack(raw_header)
    if magic != _MAGIC:
        raise _RequestError("invalid_frame", "invalid ForgeRL frame magic")
    if version != _PROTOCOL_VERSION:
        raise _RequestError(
            "unsupported_protocol",
            f"protocol version {version} is unsupported",
        )
    if metadata_size <= 0 or metadata_size > int(max_metadata_bytes):
        raise _RequestError("invalid_frame", "metadata size is outside the configured limit")
    if payload_size > int(max_payload_bytes):
        raise _RequestError("payload_too_large", "payload exceeds the configured limit")
    raw_metadata = _read_exact(connection, metadata_size)
    raw_payload = _read_exact(connection, payload_size)
    assert raw_metadata is not None
    assert raw_payload is not None
    try:
        metadata = json.loads(raw_metadata.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _RequestError("invalid_metadata", "metadata must be valid UTF-8 JSON") from error
    if not isinstance(metadata, dict):
        raise _RequestError("invalid_metadata", "metadata must be a JSON object")
    return metadata, raw_payload


def _payload_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_digest(metadata: Mapping[str, Any], payload: bytes) -> None:
    expected = str(metadata.get("payload_sha256", ""))
    if not expected:
        raise _RequestError("missing_digest", "payload_sha256 is required")
    actual = _payload_digest(payload)
    if not hmac.compare_digest(expected, actual):
        raise _RequestError("checksum_mismatch", "payload SHA-256 does not match")


def _append_numpy_array(
    entries: list[dict[str, Any]],
    chunks: list[bytes],
    *,
    path: list[str],
    value: np.ndarray,
    offset: int,
) -> int:
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"object dtype is not supported for {path}")
    raw = memoryview(array).cast("B").tobytes()
    entries.append(
        {
            "path": list(path),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "offset": int(offset),
            "nbytes": len(raw),
        }
    )
    chunks.append(raw)
    return offset + len(raw)


def encode_transition_batch(batch: TransitionBatch) -> tuple[dict[str, Any], bytes]:
    """Encode a validated transition batch without pickle or executable payloads."""

    entries: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    offset = 0
    for group_name, tree in (
        ("obs", batch.obs),
        ("next_obs", batch.next_obs),
        ("action", batch.action),
        ("hidden_in", batch.hidden_in or {}),
        ("hidden_out", batch.hidden_out or {}),
        ("extras", batch.extras),
    ):
        for name in sorted(tree):
            offset = _append_numpy_array(
                entries,
                chunks,
                path=[group_name, name],
                value=tree[name],
                offset=offset,
            )
    for name in (
        "reward",
        "terminated",
        "truncated",
        "valid_mask",
        "behavior_log_prob",
        "behavior_value",
        "policy_version",
        "episode_id",
        "step_id",
        "actor_id",
        "env_id",
    ):
        offset = _append_numpy_array(
            entries,
            chunks,
            path=[name],
            value=getattr(batch, name),
            offset=offset,
        )
    payload = b"".join(chunks)
    metadata = {
        "codec": "forge_rl_transition_batch_v1",
        "row_count": batch.size,
        "arrays": entries,
        "payload_sha256": _payload_digest(payload),
    }
    return metadata, payload


def _decode_numpy_entries(
    entries: Any,
    payload: bytes,
) -> dict[tuple[str, ...], np.ndarray]:
    if not isinstance(entries, list) or not entries:
        raise _RequestError("invalid_payload", "array metadata must be a non-empty list")
    decoded: dict[tuple[str, ...], np.ndarray] = {}
    ranges: list[tuple[int, int]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise _RequestError("invalid_payload", "array entry must be an object")
        raw_path = raw_entry.get("path")
        if not isinstance(raw_path, list) or not raw_path or len(raw_path) > 2:
            raise _RequestError("invalid_payload", "array path must contain one or two names")
        path = tuple(str(part) for part in raw_path)
        if any(not part for part in path) or path in decoded:
            raise _RequestError("invalid_payload", f"invalid or duplicate array path {path}")
        try:
            dtype = np.dtype(str(raw_entry["dtype"]))
            shape = tuple(int(value) for value in raw_entry["shape"])
            offset = int(raw_entry["offset"])
            nbytes = int(raw_entry["nbytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise _RequestError("invalid_payload", f"malformed array entry for {path}") from error
        if dtype.hasobject or any(value < 0 for value in shape) or offset < 0 or nbytes < 0:
            raise _RequestError("invalid_payload", f"unsafe array metadata for {path}")
        element_count = math.prod(shape)
        expected_bytes = element_count * dtype.itemsize
        if expected_bytes != nbytes or offset + nbytes > len(payload):
            raise _RequestError("invalid_payload", f"array byte range is invalid for {path}")
        ranges.append((offset, offset + nbytes))
        array = np.frombuffer(
            payload,
            dtype=dtype,
            count=element_count,
            offset=offset,
        ).copy()
        decoded[path] = array.reshape(shape)
    ordered_ranges = sorted(ranges)
    if any(
        right_start < left_end
        for (_, left_end), (right_start, _) in zip(
            ordered_ranges,
            ordered_ranges[1:],
            strict=False,
        )
    ):
        raise _RequestError("invalid_payload", "array byte ranges overlap")
    return decoded


def decode_transition_batch(metadata: Mapping[str, Any], payload: bytes) -> TransitionBatch:
    if metadata.get("codec") != "forge_rl_transition_batch_v1":
        raise _RequestError("unsupported_codec", "unsupported transition codec")
    _validate_digest(metadata, payload)
    arrays = _decode_numpy_entries(metadata.get("arrays"), payload)

    def tree(name: str, *, optional: bool = False) -> dict[str, np.ndarray] | None:
        selected = {
            path[1]: value
            for path, value in arrays.items()
            if len(path) == 2 and path[0] == name
        }
        if selected:
            return selected
        return None if optional else {}

    def scalar(name: str) -> np.ndarray:
        try:
            return arrays[(name,)]
        except KeyError as error:
            raise _RequestError(
                "invalid_payload",
                f"required transition array {name} is missing",
            ) from error

    try:
        batch = TransitionBatch(
            obs=tree("obs") or {},
            next_obs=tree("next_obs") or {},
            action=tree("action") or {},
            reward=scalar("reward"),
            terminated=scalar("terminated"),
            truncated=scalar("truncated"),
            valid_mask=scalar("valid_mask"),
            behavior_log_prob=scalar("behavior_log_prob"),
            behavior_value=scalar("behavior_value"),
            policy_version=scalar("policy_version"),
            episode_id=scalar("episode_id"),
            step_id=scalar("step_id"),
            actor_id=scalar("actor_id"),
            env_id=scalar("env_id"),
            hidden_in=tree("hidden_in", optional=True),
            hidden_out=tree("hidden_out", optional=True),
            extras=tree("extras") or {},
        )
    except (TypeError, ValueError) as error:
        raise _RequestError("invalid_transition", str(error)) from error
    declared_rows = int(metadata.get("row_count", -1))
    if declared_rows != batch.size:
        raise _RequestError("invalid_transition", "declared row count does not match payload")
    return batch


_TORCH_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}


def encode_policy_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], bytes]:
    if not state_dict:
        raise ValueError("state_dict must not be empty")
    entries: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_dict value {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        if dtype_name not in _TORCH_DTYPES:
            raise ValueError(f"unsupported tensor dtype {tensor.dtype} for {name!r}")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        entries.append(
            {
                "name": str(name),
                "dtype": dtype_name,
                "shape": list(tensor.shape),
                "offset": offset,
                "nbytes": len(raw),
            }
        )
        chunks.append(raw)
        offset += len(raw)
    payload = b"".join(chunks)
    return {
        "codec": "forge_rl_state_dict_v1",
        "tensors": entries,
        "payload_sha256": _payload_digest(payload),
    }, payload


def decode_policy_state_dict(
    metadata: Mapping[str, Any],
    payload: bytes,
) -> dict[str, torch.Tensor]:
    if metadata.get("codec") != "forge_rl_state_dict_v1":
        raise _RequestError("unsupported_codec", "unsupported policy codec")
    _validate_digest(metadata, payload)
    entries = metadata.get("tensors")
    if not isinstance(entries, list) or not entries:
        raise _RequestError("invalid_payload", "tensor metadata must be a non-empty list")
    state: dict[str, torch.Tensor] = {}
    ranges: list[tuple[int, int]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise _RequestError("invalid_payload", "tensor entry must be an object")
        try:
            name = str(raw_entry["name"])
            dtype_name = str(raw_entry["dtype"])
            shape = tuple(int(value) for value in raw_entry["shape"])
            offset = int(raw_entry["offset"])
            nbytes = int(raw_entry["nbytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise _RequestError("invalid_payload", "malformed tensor entry") from error
        if not name or name in state or dtype_name not in _TORCH_DTYPES:
            raise _RequestError("invalid_payload", f"invalid tensor identity {name!r}")
        if any(value < 0 for value in shape) or offset < 0 or nbytes < 0:
            raise _RequestError("invalid_payload", f"invalid tensor metadata for {name!r}")
        dtype = _TORCH_DTYPES[dtype_name]
        expected_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if expected_bytes != nbytes or offset + nbytes > len(payload):
            raise _RequestError(
                "invalid_payload",
                f"tensor byte range is invalid for {name!r}",
            )
        ranges.append((offset, offset + nbytes))
        raw = np.frombuffer(payload, dtype=np.uint8, count=nbytes, offset=offset).copy()
        byte_tensor = torch.from_numpy(raw)
        tensor = byte_tensor.view(dtype)
        state[name] = tensor.reshape(shape).clone()
    ordered_ranges = sorted(ranges)
    if any(
        right_start < left_end
        for (_, left_end), (right_start, _) in zip(
            ordered_ranges,
            ordered_ranges[1:],
            strict=False,
        )
    ):
        raise _RequestError("invalid_payload", "tensor byte ranges overlap")
    return state


def _parse_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    selected = endpoint if "://" in endpoint else f"tcp://{endpoint}"
    parsed = urlparse(selected)
    if parsed.scheme != "tcp" or parsed.hostname is None or parsed.port is None:
        raise ValueError("endpoint must be tcp://host:port")
    return parsed.hostname, int(parsed.port)


class _DataPlaneTCPServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    service: "_BaseDataService"


class _DataPlaneRequestHandler(socketserver.BaseRequestHandler):
    server: _DataPlaneTCPServer

    def handle(self) -> None:
        connection = self.request
        assert isinstance(connection, socket.socket)
        connection.settimeout(self.server.service.socket_timeout)
        while True:
            try:
                frame = _receive_frame(
                    connection,
                    max_metadata_bytes=self.server.service.max_metadata_bytes,
                    max_payload_bytes=self.server.service.max_payload_bytes,
                )
            except (EOFError, OSError, socket.timeout):
                return
            except _RequestError as error:
                _send_frame(connection, self.server.service.error_payload("", error))
                return
            if frame is None:
                return
            metadata, payload = frame
            request_id = str(metadata.get("request_id", ""))
            try:
                response = self.server.service.dispatch(metadata, payload)
            except _RequestError as error:
                self.server.service.metrics_counter.rejected(payload_bytes=len(payload))
                response = self.server.service.error_payload(request_id, error)
            except Exception as error:  # pragma: no cover - defensive protocol boundary
                self.server.service.metrics_counter.rejected(payload_bytes=len(payload))
                response = self.server.service.error_payload(
                    request_id,
                    _RequestError("internal_error", str(error), retryable=True),
                )
            try:
                _send_frame(connection, response)
            except OSError:
                return


class _BaseDataService:
    operation: str

    def __init__(
        self,
        *,
        host: str,
        port: int,
        bearer_token: str | None,
        max_metadata_bytes: int,
        max_payload_bytes: int,
        socket_timeout: float,
        idempotency_cache_size: int,
    ) -> None:
        if max_metadata_bytes <= 0 or max_payload_bytes <= 0 or socket_timeout <= 0:
            raise ValueError("frame limits and socket timeout must be positive")
        if idempotency_cache_size <= 0:
            raise ValueError("idempotency_cache_size must be positive")
        self.bearer_token = bearer_token
        self.max_metadata_bytes = int(max_metadata_bytes)
        self.max_payload_bytes = int(max_payload_bytes)
        self.socket_timeout = float(socket_timeout)
        self.idempotency_cache_size = int(idempotency_cache_size)
        self.metrics_counter = _DataChannelMetrics()
        self._acks: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._request_lock = threading.Lock()
        self._server = _DataPlaneTCPServer((host, int(port)), _DataPlaneRequestHandler)
        self._server.service = self
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def endpoint(self) -> str:
        host, port = self._server.server_address[:2]
        selected_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else str(host)
        return f"tcp://{selected_host}:{int(port)}"

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("data service is closed")
        if self._thread is not None:
            raise RuntimeError("data service is already running")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"forge-rl-{self.operation}",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread is not None and self._thread.is_alive():
            self._server.shutdown()
            self._thread.join(timeout=5.0)
        self._server.server_close()
        self._thread = None

    def __enter__(self) -> "_BaseDataService":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def metrics(self) -> DataChannelMetricsSnapshot:
        return self.metrics_counter.snapshot()

    def _authorize(self, metadata: Mapping[str, Any]) -> None:
        if self.bearer_token is None:
            return
        provided = str(metadata.get("bearer_token", ""))
        if not hmac.compare_digest(provided, self.bearer_token):
            raise _RequestError("unauthorized", "invalid bearer token")

    def _validate_request(self, metadata: Mapping[str, Any]) -> str:
        self._authorize(metadata)
        if metadata.get("operation") != self.operation:
            raise _RequestError("unsupported_operation", "unexpected data-channel operation")
        request_id = str(metadata.get("request_id", ""))
        if not request_id or len(request_id) > 128:
            raise _RequestError("invalid_request_id", "request_id must contain 1-128 characters")
        return request_id

    def _cache_ack(self, request_id: str, response: dict[str, Any]) -> None:
        self._acks[request_id] = response
        self._acks.move_to_end(request_id)
        while len(self._acks) > self.idempotency_cache_size:
            self._acks.popitem(last=False)

    def error_payload(self, request_id: str, error: _RequestError) -> dict[str, Any]:
        return {
            "ok": False,
            "request_id": request_id,
            "error": {
                "code": error.code,
                "message": error.message,
                "retryable": error.retryable,
            },
        }

    def dispatch(self, metadata: Mapping[str, Any], payload: bytes) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ExperiencePushAck:
    request_id: str
    accepted_rows: int
    policy_version_min: int
    policy_version_max: int
    queue_rows: int


class NetworkExperienceService(_BaseDataService):
    """Direct EnvRunner-to-Experience receiver with validation and backpressure."""

    operation = "put_transition_batch"

    def __init__(
        self,
        experience_queue: OnPolicyExperienceQueue,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        bearer_token: str | None = None,
        put_timeout: float = 0.0,
        max_metadata_bytes: int = _DEFAULT_MAX_METADATA_BYTES,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
        socket_timeout: float = _DEFAULT_SOCKET_TIMEOUT,
        idempotency_cache_size: int = 4096,
    ) -> None:
        if put_timeout < 0:
            raise ValueError("put_timeout cannot be negative")
        self.experience_queue = experience_queue
        self.put_timeout = float(put_timeout)
        super().__init__(
            host=host,
            port=port,
            bearer_token=bearer_token,
            max_metadata_bytes=max_metadata_bytes,
            max_payload_bytes=max_payload_bytes,
            socket_timeout=socket_timeout,
            idempotency_cache_size=idempotency_cache_size,
        )

    def dispatch(self, metadata: Mapping[str, Any], payload: bytes) -> dict[str, Any]:
        request_id = self._validate_request(metadata)
        with self._request_lock:
            cached = self._acks.get(request_id)
            if cached is not None:
                return dict(cached)
            batch = decode_transition_batch(metadata, payload)
            valid = batch.valid_mask > 0.0
            accepted_rows = int(valid.sum())
            if accepted_rows <= 0:
                raise _RequestError("empty_batch", "transition batch contains no valid rows")
            versions = batch.policy_version[valid]
            actors = np.unique(batch.actor_id[valid])
            item = ExperienceItem.create(
                batch,
                train_rows=accepted_rows,
                policy_version_min=int(versions.min()),
                policy_version_max=int(versions.max()),
                actor_id=int(actors[0]) if actors.size == 1 else -1,
            )
            try:
                self.experience_queue.put(item, timeout=self.put_timeout)
            except TimeoutError as error:
                raise _RequestError(
                    "backpressure",
                    "experience queue has no capacity",
                    retryable=True,
                ) from error
            response = {
                "ok": True,
                "request_id": request_id,
                "accepted_rows": accepted_rows,
                "policy_version_min": item.policy_version_min,
                "policy_version_max": item.policy_version_max,
                "queue_rows": self.experience_queue.rows,
            }
            self._cache_ack(request_id, response)
            self.metrics_counter.accepted(payload_bytes=len(payload), rows=accepted_rows)
            return dict(response)


@dataclass(frozen=True, slots=True)
class PolicyPublishAck:
    request_id: str
    version: int
    tensor_count: int


class NetworkPolicyService(_BaseDataService):
    """Direct Learner-to-Policy receiver backed by a monotonic PolicyRegistry."""

    operation = "publish_policy"

    def __init__(
        self,
        registry: PolicyRegistry,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        bearer_token: str | None = None,
        max_metadata_bytes: int = _DEFAULT_MAX_METADATA_BYTES,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
        socket_timeout: float = _DEFAULT_SOCKET_TIMEOUT,
        idempotency_cache_size: int = 1024,
    ) -> None:
        self.registry = registry
        super().__init__(
            host=host,
            port=port,
            bearer_token=bearer_token,
            max_metadata_bytes=max_metadata_bytes,
            max_payload_bytes=max_payload_bytes,
            socket_timeout=socket_timeout,
            idempotency_cache_size=idempotency_cache_size,
        )

    def dispatch(self, metadata: Mapping[str, Any], payload: bytes) -> dict[str, Any]:
        request_id = self._validate_request(metadata)
        with self._request_lock:
            cached = self._acks.get(request_id)
            if cached is not None:
                return dict(cached)
            if "version" not in metadata:
                raise _RequestError("invalid_version", "policy version is required")
            version = int(metadata["version"])
            state_dict = decode_policy_state_dict(metadata, payload)
            user_metadata = metadata.get("policy_metadata", {})
            if not isinstance(user_metadata, dict):
                raise _RequestError("invalid_metadata", "policy_metadata must be an object")
            try:
                snapshot = self.registry.publish(
                    state_dict,
                    version=version,
                    metadata=user_metadata,
                )
            except ValueError as error:
                raise _RequestError("policy_version_conflict", str(error)) from error
            response = {
                "ok": True,
                "request_id": request_id,
                "version": snapshot.version,
                "tensor_count": len(snapshot.state_dict),
            }
            self._cache_ack(request_id, response)
            self.metrics_counter.accepted(payload_bytes=len(payload), versions=1)
            return dict(response)


class _PersistentDataClient:
    def __init__(
        self,
        endpoint: str,
        *,
        bearer_token: str | None,
        timeout: float,
        max_metadata_bytes: int,
        max_payload_bytes: int,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.host, self.port = _parse_tcp_endpoint(endpoint)
        self.bearer_token = bearer_token
        self.timeout = float(timeout)
        self.max_metadata_bytes = int(max_metadata_bytes)
        self.max_payload_bytes = int(max_payload_bytes)
        self._connection: socket.socket | None = None
        self._lock = threading.Lock()
        self._counter = 0
        self._prefix = f"{time.time_ns():x}-{id(self):x}"

    def _next_request_id(self) -> str:
        self._counter += 1
        return f"{self._prefix}-{self._counter}"

    def _connect(self) -> socket.socket:
        if self._connection is None:
            connection = socket.create_connection((self.host, self.port), timeout=self.timeout)
            connection.settimeout(self.timeout)
            self._connection = connection
        return self._connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self._connection.close()
                self._connection = None

    def __enter__(self) -> "_PersistentDataClient":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def _request(
        self,
        *,
        operation: str,
        metadata: Mapping[str, Any],
        payload: bytes,
        request_id: str | None,
    ) -> dict[str, Any]:
        selected_request_id = request_id or self._next_request_id()
        envelope = {
            **dict(metadata),
            "operation": operation,
            "request_id": selected_request_id,
        }
        if self.bearer_token is not None:
            envelope["bearer_token"] = self.bearer_token
        with self._lock:
            try:
                connection = self._connect()
                _send_frame(connection, envelope, payload)
                frame = _receive_frame(
                    connection,
                    max_metadata_bytes=self.max_metadata_bytes,
                    max_payload_bytes=self.max_payload_bytes,
                )
                if frame is None:
                    raise EOFError("server closed without a response")
                response, response_payload = frame
                if response_payload:
                    raise DataChannelError(
                        code="protocol_error",
                        message="acknowledgement payload must be empty",
                    )
            except (EOFError, OSError, socket.timeout, _RequestError) as error:
                if self._connection is not None:
                    self._connection.close()
                    self._connection = None
                raise DataChannelError(
                    code="transport_error",
                    message=str(error),
                    retryable=True,
                ) from error
        if response.get("request_id") != selected_request_id:
            raise DataChannelError(
                code="protocol_error",
                message="response request_id does not match",
            )
        if not bool(response.get("ok")):
            detail = response.get("error", {})
            if not isinstance(detail, dict):
                detail = {}
            raise DataChannelError(
                code=str(detail.get("code", "remote_error")),
                message=str(detail.get("message", "remote data-channel error")),
                retryable=bool(detail.get("retryable", False)),
            )
        return response


class NetworkExperienceClient(_PersistentDataClient):
    def __init__(
        self,
        endpoint: str,
        *,
        bearer_token: str | None = None,
        timeout: float = _DEFAULT_SOCKET_TIMEOUT,
        max_metadata_bytes: int = _DEFAULT_MAX_METADATA_BYTES,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        super().__init__(
            endpoint,
            bearer_token=bearer_token,
            timeout=timeout,
            max_metadata_bytes=max_metadata_bytes,
            max_payload_bytes=max_payload_bytes,
        )

    def put(
        self,
        batch: TransitionBatch,
        *,
        request_id: str | None = None,
    ) -> ExperiencePushAck:
        metadata, payload = encode_transition_batch(batch)
        response = self._request(
            operation="put_transition_batch",
            metadata=metadata,
            payload=payload,
            request_id=request_id,
        )
        return ExperiencePushAck(
            request_id=str(response["request_id"]),
            accepted_rows=int(response["accepted_rows"]),
            policy_version_min=int(response["policy_version_min"]),
            policy_version_max=int(response["policy_version_max"]),
            queue_rows=int(response["queue_rows"]),
        )


class NetworkPolicyClient(_PersistentDataClient):
    def __init__(
        self,
        endpoint: str,
        *,
        bearer_token: str | None = None,
        timeout: float = _DEFAULT_SOCKET_TIMEOUT,
        max_metadata_bytes: int = _DEFAULT_MAX_METADATA_BYTES,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        super().__init__(
            endpoint,
            bearer_token=bearer_token,
            timeout=timeout,
            max_metadata_bytes=max_metadata_bytes,
            max_payload_bytes=max_payload_bytes,
        )

    def publish(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        version: int,
        metadata: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> PolicyPublishAck:
        envelope, payload = encode_policy_state_dict(state_dict)
        envelope.update(
            {
                "version": int(version),
                "policy_metadata": dict(metadata or {}),
            }
        )
        response = self._request(
            operation="publish_policy",
            metadata=envelope,
            payload=payload,
            request_id=request_id,
        )
        return PolicyPublishAck(
            request_id=str(response["request_id"]),
            version=int(response["version"]),
            tensor_count=int(response["tensor_count"]),
        )
