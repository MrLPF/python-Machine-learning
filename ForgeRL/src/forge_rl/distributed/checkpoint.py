from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence
import uuid

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.optim import Optimizer
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)

_MANIFEST_NAME = "forgerl_manifest.json"
_CHECKPOINT_SCHEMA_VERSION = 1


class DistributedCheckpointError(RuntimeError):
    """Raised when a ForgeRL distributed checkpoint cannot be committed or restored."""


class IncompleteCheckpointError(DistributedCheckpointError):
    """Raised when a checkpoint directory does not contain a valid completion manifest."""


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    schema_version: int
    checkpoint_id: str
    global_step: int
    policy_version: int
    saved_world_size: int
    created_at_unix_ns: int
    model_class: str
    optimizer_classes: tuple[str, ...]
    metadata: dict[str, Any]

    def validate(self) -> None:
        if self.schema_version != _CHECKPOINT_SCHEMA_VERSION:
            raise IncompleteCheckpointError(
                f"unsupported checkpoint schema {self.schema_version}"
            )
        if not self.checkpoint_id:
            raise IncompleteCheckpointError("checkpoint_id is missing")
        if self.global_step < 0 or self.policy_version < 0:
            raise IncompleteCheckpointError("global_step and policy_version must be non-negative")
        if self.saved_world_size <= 0:
            raise IncompleteCheckpointError("saved_world_size must be positive")
        if self.created_at_unix_ns <= 0:
            raise IncompleteCheckpointError("created_at_unix_ns must be positive")
        if not self.model_class or not self.optimizer_classes:
            raise IncompleteCheckpointError("model and optimizer class metadata are required")
        try:
            json.dumps(self.metadata, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise IncompleteCheckpointError("checkpoint metadata is not valid JSON") from error

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["optimizer_classes"] = list(self.optimizer_classes)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointManifest":
        try:
            manifest = cls(
                schema_version=int(payload["schema_version"]),
                checkpoint_id=str(payload["checkpoint_id"]),
                global_step=int(payload["global_step"]),
                policy_version=int(payload["policy_version"]),
                saved_world_size=int(payload["saved_world_size"]),
                created_at_unix_ns=int(payload["created_at_unix_ns"]),
                model_class=str(payload["model_class"]),
                optimizer_classes=tuple(
                    str(value) for value in payload["optimizer_classes"]
                ),
                metadata=dict(payload.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IncompleteCheckpointError("checkpoint manifest is malformed") from error
        manifest.validate()
        return manifest


def _qualified_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _base_model(model: nn.Module) -> nn.Module:
    candidate = getattr(model, "module", None)
    return candidate if isinstance(candidate, nn.Module) else model


def _normalise_optimizers(
    optimizers: Optimizer | Sequence[Optimizer],
) -> tuple[tuple[Optimizer, ...], Optimizer | tuple[Optimizer, ...]]:
    if isinstance(optimizers, Optimizer):
        selected = (optimizers,)
    else:
        selected = tuple(optimizers)
    if not selected or any(not isinstance(value, Optimizer) for value in selected):
        raise ValueError("at least one torch Optimizer is required")
    api_value: Optimizer | tuple[Optimizer, ...] = (
        selected[0] if len(selected) == 1 else selected
    )
    return selected, api_value


def _distributed_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank_world_size() -> tuple[int, int]:
    if _distributed_active():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _collective_device() -> torch.device | None:
    if not _distributed_active() or dist.get_backend() != "nccl":
        return None
    return torch.device("cuda", torch.cuda.current_device())


def _broadcast_object(value: Any, *, source_rank: int) -> Any:
    if not _distributed_active():
        return value
    rank = dist.get_rank()
    values = [value if rank == source_rank else None]
    dist.broadcast_object_list(
        values,
        src=source_rank,
        device=_collective_device(),
    )
    return values[0]


def _validate_checkpoint_id(checkpoint_id: str) -> str:
    selected = str(checkpoint_id).strip()
    path = Path(selected)
    if (
        not selected
        or path.is_absolute()
        or len(path.parts) != 1
        or selected in {".", ".."}
    ):
        raise ValueError("checkpoint_id must be one relative path component")
    return selected


def _json_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    selected = dict(metadata or {})
    try:
        encoded = json.dumps(selected, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint metadata must contain JSON-safe values") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    return decoded


def _write_manifest(path: Path, manifest: CheckpointManifest) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = json.dumps(
        manifest.to_dict(),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_manifest(path: Path) -> CheckpointManifest:
    if not path.is_file():
        raise IncompleteCheckpointError(
            f"checkpoint completion manifest is missing: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IncompleteCheckpointError("checkpoint manifest cannot be decoded") from error
    if not isinstance(payload, dict):
        raise IncompleteCheckpointError("checkpoint manifest must be a JSON object")
    return CheckpointManifest.from_dict(payload)


class DistributedCheckpointManager:
    """Atomic PyTorch DCP wrapper with topology-independent model/optimizer restore.

    Every participating learner rank calls ``save`` and ``load``. PyTorch Distributed Checkpoint
    writes or reads local shards in parallel and performs load-time resharding when the current
    world size or parallelism differs from the saved topology. ForgeRL adds a JSON completion
    manifest, monotonic training metadata and an atomic directory commit on the coordinator rank.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        coordinator_rank: int = 0,
    ) -> None:
        self.root = Path(root)
        self.coordinator_rank = int(coordinator_rank)
        if self.coordinator_rank < 0:
            raise ValueError("coordinator_rank must be non-negative")

    def checkpoint_path(self, checkpoint_id: str) -> Path:
        return self.root / _validate_checkpoint_id(checkpoint_id)

    def read_manifest(self, checkpoint_id: str) -> CheckpointManifest:
        checkpoint = self.checkpoint_path(checkpoint_id)
        manifest = _read_manifest(checkpoint / _MANIFEST_NAME)
        if manifest.checkpoint_id != checkpoint.name:
            raise IncompleteCheckpointError(
                "manifest checkpoint_id does not match its directory"
            )
        return manifest

    def save(
        self,
        checkpoint_id: str,
        *,
        model: nn.Module,
        optimizers: Optimizer | Sequence[Optimizer],
        global_step: int,
        policy_version: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointManifest:
        selected_id = _validate_checkpoint_id(checkpoint_id)
        step = int(global_step)
        version = int(policy_version)
        if step < 0 or version < 0:
            raise ValueError("global_step and policy_version must be non-negative")
        optimizer_tuple, optimizer_api = _normalise_optimizers(optimizers)
        rank, world_size = _rank_world_size()
        if self.coordinator_rank >= world_size:
            raise ValueError("coordinator_rank is outside the current world size")

        setup: dict[str, Any] | None = None
        if rank == self.coordinator_rank:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                final_path = self.root / selected_id
                if final_path.exists():
                    raise FileExistsError(f"checkpoint already exists: {final_path}")
                setup = {
                    "ok": True,
                    "temporary_name": (
                        f".{selected_id}.incomplete-{uuid.uuid4().hex}"
                    ),
                    "created_at_unix_ns": time.time_ns(),
                }
            except Exception as error:
                setup = {"ok": False, "error": str(error)}
        setup = _broadcast_object(setup, source_rank=self.coordinator_rank)
        if not isinstance(setup, dict) or not bool(setup.get("ok")):
            message = "checkpoint setup failed"
            if isinstance(setup, dict):
                message = str(setup.get("error", message))
            raise DistributedCheckpointError(message)

        temporary_path = self.root / str(setup["temporary_name"])
        final_path = self.root / selected_id
        manifest = CheckpointManifest(
            schema_version=_CHECKPOINT_SCHEMA_VERSION,
            checkpoint_id=selected_id,
            global_step=step,
            policy_version=version,
            saved_world_size=world_size,
            created_at_unix_ns=int(setup["created_at_unix_ns"]),
            model_class=_qualified_name(_base_model(model)),
            optimizer_classes=tuple(
                _qualified_name(optimizer) for optimizer in optimizer_tuple
            ),
            metadata=_json_metadata(metadata),
        )
        manifest.validate()

        model_state, optimizer_state = get_state_dict(model, optimizer_api)
        dcp.save(
            {
                "model": model_state,
                "optimizer": optimizer_state,
            },
            checkpoint_id=temporary_path,
            no_dist=not _distributed_active(),
        )

        commit: dict[str, Any] | None = None
        if rank == self.coordinator_rank:
            try:
                _write_manifest(temporary_path / _MANIFEST_NAME, manifest)
                os.replace(temporary_path, final_path)
                commit = {"ok": True, "manifest": manifest.to_dict()}
            except Exception as error:
                shutil.rmtree(temporary_path, ignore_errors=True)
                commit = {"ok": False, "error": str(error)}
        commit = _broadcast_object(commit, source_rank=self.coordinator_rank)
        if not isinstance(commit, dict) or not bool(commit.get("ok")):
            message = "checkpoint commit failed"
            if isinstance(commit, dict):
                message = str(commit.get("error", message))
            raise DistributedCheckpointError(message)
        return CheckpointManifest.from_dict(commit["manifest"])

    def load(
        self,
        checkpoint_id: str,
        *,
        model: nn.Module,
        optimizers: Optimizer | Sequence[Optimizer],
        strict: bool = True,
    ) -> CheckpointManifest:
        selected_id = _validate_checkpoint_id(checkpoint_id)
        checkpoint_path = self.root / selected_id
        optimizer_tuple, optimizer_api = _normalise_optimizers(optimizers)
        del optimizer_tuple
        rank, world_size = _rank_world_size()
        if self.coordinator_rank >= world_size:
            raise ValueError("coordinator_rank is outside the current world size")

        manifest_result: dict[str, Any] | None = None
        if rank == self.coordinator_rank:
            try:
                manifest = _read_manifest(checkpoint_path / _MANIFEST_NAME)
                if manifest.checkpoint_id != selected_id:
                    raise IncompleteCheckpointError(
                        "manifest checkpoint_id does not match the requested checkpoint"
                    )
                manifest_result = {"ok": True, "manifest": manifest.to_dict()}
            except Exception as error:
                manifest_result = {"ok": False, "error": str(error)}
        manifest_result = _broadcast_object(
            manifest_result,
            source_rank=self.coordinator_rank,
        )
        if not isinstance(manifest_result, dict) or not bool(
            manifest_result.get("ok")
        ):
            message = "checkpoint manifest validation failed"
            if isinstance(manifest_result, dict):
                message = str(manifest_result.get("error", message))
            raise IncompleteCheckpointError(message)
        manifest = CheckpointManifest.from_dict(manifest_result["manifest"])

        model_state, optimizer_state = get_state_dict(model, optimizer_api)
        dcp.load(
            {
                "model": model_state,
                "optimizer": optimizer_state,
            },
            checkpoint_id=checkpoint_path,
            no_dist=not _distributed_active(),
        )
        incompatible = set_state_dict(
            model,
            optimizer_api,
            model_state_dict=model_state,
            optim_state_dict=optimizer_state,
            options=StateDictOptions(strict=bool(strict)),
        )
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise DistributedCheckpointError(
                "checkpoint model keys do not match: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        if _distributed_active():
            dist.barrier()
        return manifest
