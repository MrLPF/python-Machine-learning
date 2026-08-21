from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import os
from typing import Iterator, Mapping

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    backend: str

    @classmethod
    def from_env(cls, *, backend: str | None = None) -> "DistributedContext":
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        selected_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        return cls(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            backend=selected_backend,
        )

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def device(self) -> torch.device:
        if self.backend == "nccl":
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    def initialize(
        self,
        *,
        init_method: str = "env://",
        timeout_seconds: float = 120.0,
    ) -> None:
        if not self.distributed or dist.is_initialized():
            return
        if self.backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL backend requires CUDA")
            torch.cuda.set_device(self.local_rank)
        dist.init_process_group(
            backend=self.backend,
            init_method=init_method,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=float(timeout_seconds)),
        )

    def wrap(self, module: nn.Module) -> nn.Module:
        module = module.to(self.device)
        if not self.distributed:
            return module
        if not dist.is_initialized():
            raise RuntimeError("initialize the process group before wrapping the model")
        if self.backend == "nccl":
            return DistributedDataParallel(
                module,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        return DistributedDataParallel(module)

    def barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


@contextmanager
def distributed_session(
    context: DistributedContext,
    *,
    init_method: str = "env://",
    timeout_seconds: float = 120.0,
) -> Iterator[DistributedContext]:
    context.initialize(init_method=init_method, timeout_seconds=timeout_seconds)
    try:
        yield context
    finally:
        context.close()


def all_reduce_mean(
    metrics: Mapping[str, float | torch.Tensor],
    *,
    device: torch.device | None = None,
) -> dict[str, float]:
    selected_device = device or (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available() and dist.is_initialized() and dist.get_backend() == "nccl"
        else torch.device("cpu")
    )
    keys = sorted(metrics)
    values = torch.tensor(
        [float(metrics[key]) for key in keys], dtype=torch.float64, device=selected_device
    )
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}
