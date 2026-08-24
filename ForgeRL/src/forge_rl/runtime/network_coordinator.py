from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import threading
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen

from .coordinator import (
    InMemoryCoordinator,
    NodeLease,
    NodeNotFoundError,
    NodeRole,
    PolicyVersionConflictError,
    StaleGenerationError,
)

_MAX_REQUEST_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class RemoteNodeLease:
    node_id: str
    role: NodeRole
    endpoint: str
    resources: dict[str, float]
    metadata: dict[str, Any]
    generation: int
    expires_in_seconds: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RemoteNodeLease":
        return cls(
            node_id=str(payload["node_id"]),
            role=NodeRole(str(payload["role"])),
            endpoint=str(payload["endpoint"]),
            resources={
                str(key): float(value)
                for key, value in dict(payload.get("resources", {})).items()
            },
            metadata=dict(payload.get("metadata", {})),
            generation=int(payload["generation"]),
            expires_in_seconds=float(payload["expires_in_seconds"]),
        )


class CoordinatorHTTPError(RuntimeError):
    def __init__(self, *, status: int, code: str, message: str) -> None:
        super().__init__(f"coordinator HTTP {status} {code}: {message}")
        self.status = int(status)
        self.code = str(code)
        self.message = str(message)


class _CoordinatorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    service: "NetworkCoordinatorService"


class _CoordinatorRequestHandler(BaseHTTPRequestHandler):
    server: _CoordinatorHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._write_error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "use GET or POST")

    def do_DELETE(self) -> None:
        self._write_error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "use GET or POST")

    def _dispatch(self, method: str) -> None:
        if not self.server.service.authorized(self.headers.get("Authorization", "")):
            self._write_error(HTTPStatus.UNAUTHORIZED, "unauthorized", "invalid bearer token")
            return

        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if method == "GET" and path == "/healthz":
                self._write_json(HTTPStatus.OK, {"status": "ok"})
                return
            if method == "GET" and path == "/v1/state":
                self._write_json(HTTPStatus.OK, self.server.service.state_payload())
                return
            if method == "GET" and path == "/v1/nodes":
                query = parse_qs(parsed.query)
                role = query.get("role", [None])[0]
                nodes = self.server.service.coordinator.active_nodes(role)
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "nodes": [self.server.service.lease_payload(lease) for lease in nodes],
                        "policy_version": self.server.service.coordinator.policy_version,
                    },
                )
                return
            if method == "POST" and path == "/v1/nodes/register":
                payload = self._read_json()
                lease = self.server.service.coordinator.register(
                    node_id=str(payload.get("node_id", "")),
                    role=str(payload.get("role", "")),
                    endpoint=str(payload.get("endpoint", "")),
                    resources=self._mapping(payload.get("resources"), "resources"),
                    metadata=self._mapping(payload.get("metadata"), "metadata"),
                )
                self._write_json(
                    HTTPStatus.CREATED,
                    {"lease": self.server.service.lease_payload(lease)},
                )
                return
            if method == "POST" and path == "/v1/policy/commit":
                payload = self._read_json()
                if "version" not in payload:
                    raise ValueError("version is required")
                version = self.server.service.coordinator.commit_policy_version(
                    int(payload["version"])
                )
                self._write_json(HTTPStatus.OK, {"policy_version": version})
                return
            prefix = "/v1/nodes/"
            suffix = "/heartbeat"
            if method == "POST" and path.startswith(prefix) and path.endswith(suffix):
                node_id = unquote(path[len(prefix) : -len(suffix)]).strip("/")
                if not node_id:
                    raise ValueError("node_id is required")
                payload = self._read_json()
                if "generation" not in payload:
                    raise ValueError("generation is required")
                lease = self.server.service.coordinator.heartbeat(
                    node_id,
                    generation=int(payload["generation"]),
                )
                self._write_json(
                    HTTPStatus.OK,
                    {"lease": self.server.service.lease_payload(lease)},
                )
                return
            self._write_error(HTTPStatus.NOT_FOUND, "not_found", "unknown coordinator route")
        except NodeNotFoundError as error:
            self._write_error(HTTPStatus.NOT_FOUND, "node_not_found", str(error.args[0]))
        except StaleGenerationError as error:
            self._write_error(HTTPStatus.CONFLICT, "stale_generation", str(error))
        except PolicyVersionConflictError as error:
            self._write_error(HTTPStatus.CONFLICT, "policy_version_conflict", str(error))
        except (TypeError, ValueError) as error:
            self._write_error(HTTPStatus.BAD_REQUEST, "invalid_request", str(error))
        except Exception as error:  # pragma: no cover - defensive protocol boundary
            self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", str(error))

    @staticmethod
    def _mapping(value: Any, name: str) -> Mapping[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a JSON object")
        return value

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length <= 0:
            raise ValueError("JSON request body is required")
        if length > _MAX_REQUEST_BYTES:
            raise ValueError("request body exceeds 1 MiB")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("request body must be valid UTF-8 JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _write_error(self, status: HTTPStatus, code: str, message: str) -> None:
        self._write_json(status, {"error": {"code": code, "message": message}})

    def _write_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class NetworkCoordinatorService:
    """Dependency-free HTTP/JSON coordinator for M2 control-plane metadata."""

    def __init__(
        self,
        *,
        coordinator: InMemoryCoordinator | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        lease_seconds: float = 10.0,
        bearer_token: str | None = None,
    ) -> None:
        self.coordinator = coordinator or InMemoryCoordinator(lease_seconds=lease_seconds)
        self.bearer_token = bearer_token
        self._server = _CoordinatorHTTPServer((host, int(port)), _CoordinatorRequestHandler)
        self._server.service = self
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def endpoint(self) -> str:
        host, port = self._server.server_address[:2]
        selected_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else str(host)
        return f"http://{selected_host}:{int(port)}"

    def authorized(self, authorization: str) -> bool:
        if self.bearer_token is None:
            return True
        expected = f"Bearer {self.bearer_token}"
        return hmac.compare_digest(str(authorization), expected)

    def lease_payload(self, lease: NodeLease) -> dict[str, Any]:
        return {
            "node_id": lease.node_id,
            "role": lease.role.value,
            "endpoint": lease.endpoint,
            "resources": dict(lease.resources),
            "metadata": dict(lease.metadata),
            "generation": lease.generation,
            "expires_in_seconds": max(0.0, lease.expires_at - time.monotonic()),
        }

    def state_payload(self) -> dict[str, Any]:
        nodes = self.coordinator.active_nodes()
        return {
            "policy_version": self.coordinator.policy_version,
            "lease_seconds": self.coordinator.lease_seconds,
            "nodes": [self.lease_payload(lease) for lease in nodes],
        }

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("network coordinator is closed")
        if self._thread is not None:
            raise RuntimeError("network coordinator is already running")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="forge-rl-network-coordinator",
            daemon=True,
        )
        self._thread.start()

    def serve_forever(self) -> None:
        if self._closed:
            raise RuntimeError("network coordinator is closed")
        self._server.serve_forever()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            self._server.shutdown()
            thread.join(timeout=5.0)
        self._server.server_close()
        self._thread = None

    def __enter__(self) -> "NetworkCoordinatorService":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


class NetworkCoordinatorClient:
    """Small synchronous client for coordinator registration, leases and discovery."""

    def __init__(
        self,
        endpoint: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.endpoint = endpoint.rstrip("/")
        self.bearer_token = bearer_token
        self.timeout = float(timeout)

    def register(
        self,
        *,
        node_id: str,
        role: NodeRole | str,
        endpoint: str,
        resources: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RemoteNodeLease:
        payload = self._request(
            "POST",
            "/v1/nodes/register",
            {
                "node_id": node_id,
                "role": role.value if isinstance(role, NodeRole) else str(role),
                "endpoint": endpoint,
                "resources": dict(resources or {}),
                "metadata": dict(metadata or {}),
            },
        )
        return RemoteNodeLease.from_payload(payload["lease"])

    def heartbeat(self, node_id: str, *, generation: int) -> RemoteNodeLease:
        payload = self._request(
            "POST",
            f"/v1/nodes/{quote(node_id, safe='')}/heartbeat",
            {"generation": int(generation)},
        )
        return RemoteNodeLease.from_payload(payload["lease"])

    def active_nodes(self, role: NodeRole | str | None = None) -> list[RemoteNodeLease]:
        path = "/v1/nodes"
        if role is not None:
            selected = role.value if isinstance(role, NodeRole) else str(role)
            path += f"?role={quote(selected, safe='')}"
        payload = self._request("GET", path)
        return [RemoteNodeLease.from_payload(row) for row in payload["nodes"]]

    def commit_policy_version(self, version: int) -> int:
        payload = self._request("POST", "/v1/policy/commit", {"version": int(version)})
        return int(payload["policy_version"])

    def state(self) -> dict[str, Any]:
        return self._request("GET", "/v1/state")

    def health(self) -> bool:
        return self._request("GET", "/healthz").get("status") == "ok"

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.bearer_token is not None:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = Request(
            f"{self.endpoint}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raw = error.read()
            try:
                decoded_error = json.loads(raw.decode("utf-8"))
                detail = decoded_error.get("error", {})
                code = str(detail.get("code", "http_error"))
                message = str(detail.get("message", error.reason))
            except (UnicodeDecodeError, json.JSONDecodeError):
                code = "http_error"
                message = str(error.reason)
            raise CoordinatorHTTPError(
                status=error.code,
                code=code,
                message=message,
            ) from error
        except URLError as error:
            raise CoordinatorHTTPError(
                status=0,
                code="transport_error",
                message=str(error.reason),
            ) from error
        if not isinstance(decoded, dict):
            raise CoordinatorHTTPError(
                status=0,
                code="protocol_error",
                message="coordinator response must be a JSON object",
            )
        return decoded
