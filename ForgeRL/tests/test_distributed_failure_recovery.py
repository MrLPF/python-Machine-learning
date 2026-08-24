from __future__ import annotations

import os
from pathlib import Path
import time

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.multiprocessing.spawn import ProcessExitedException

from forge_rl.distributed.checkpoint import DistributedCheckpointManager
from forge_rl.distributed.learner_group import DistributedContext


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))


def _train_step(model: nn.Module, optimizer: torch.optim.Optimizer, *, rank: int) -> None:
    inputs = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float32,
    ) + float(rank)
    loss = model(inputs).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def _faulted_group_worker(
    rank: int,
    world_size: int,
    init_method: str,
    checkpoint_root: str,
) -> None:
    context = DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        backend="gloo",
    )
    context.initialize(init_method=init_method, timeout_seconds=30)
    try:
        torch.manual_seed(4100)
        model = context.wrap(_model())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        _train_step(model, optimizer, rank=rank)
        manager = DistributedCheckpointManager(checkpoint_root)
        manager.save(
            "learner-recovery-point",
            model=model,
            optimizers=optimizer,
            global_step=1,
            policy_version=5,
            metadata={"failure_test": "learner_rank_restart", "base_seed": 4100},
        )
        context.barrier()
        if rank == 1:
            os._exit(17)
        time.sleep(0.2)
    finally:
        context.close()


def _restarted_group_worker(
    rank: int,
    world_size: int,
    init_method: str,
    checkpoint_root: str,
    output_dir: str,
) -> None:
    context = DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        backend="gloo",
    )
    context.initialize(init_method=init_method, timeout_seconds=30)
    try:
        torch.manual_seed(9999 + rank)
        model = context.wrap(_model())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        manager = DistributedCheckpointManager(checkpoint_root)
        manifest = manager.load(
            "learner-recovery-point",
            model=model,
            optimizers=optimizer,
        )
        assert manifest.global_step == 1
        assert manifest.policy_version == 5
        assert manifest.saved_world_size == 2

        _train_step(model, optimizer, rank=rank)
        context.barrier()
        torch.save(
            {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "restored_step": manifest.global_step,
                "restored_policy_version": manifest.policy_version,
            },
            Path(output_dir) / f"rank-{rank}.pt",
        )
        context.barrier()
    finally:
        context.close()


@pytest.mark.distributed
def test_learner_rank_failure_restarts_group_from_last_completed_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    failure_init = tmp_path / "failure-rendezvous"
    with pytest.raises(ProcessExitedException) as failure:
        mp.spawn(
            _faulted_group_worker,
            args=(2, f"file://{failure_init}", str(checkpoint_root)),
            nprocs=2,
            join=True,
        )
    assert failure.value.exit_code == 17

    manifest = DistributedCheckpointManager(checkpoint_root).read_manifest(
        "learner-recovery-point"
    )
    assert manifest.global_step == 1
    assert manifest.policy_version == 5
    assert manifest.saved_world_size == 2

    output_dir = tmp_path / "restarted"
    output_dir.mkdir()
    restart_init = tmp_path / "restart-rendezvous"
    mp.spawn(
        _restarted_group_worker,
        args=(
            2,
            f"file://{restart_init}",
            str(checkpoint_root),
            str(output_dir),
        ),
        nprocs=2,
        join=True,
    )

    restored = [
        torch.load(output_dir / f"rank-{rank}.pt", weights_only=True)
        for rank in range(2)
    ]
    assert all(row["restored_step"] == 1 for row in restored)
    assert all(row["restored_policy_version"] == 5 for row in restored)
    assert restored[0]["model"].keys() == restored[1]["model"].keys()
    for name in restored[0]["model"]:
        torch.testing.assert_close(
            restored[0]["model"][name],
            restored[1]["model"][name],
            rtol=0.0,
            atol=0.0,
        )
