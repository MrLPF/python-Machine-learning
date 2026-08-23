from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from forge_rl.distributed.checkpoint import (
    DistributedCheckpointManager,
    IncompleteCheckpointError,
)
from forge_rl.distributed.learner_group import DistributedContext


def _model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(3, 8),
        nn.Tanh(),
        nn.Linear(8, 2),
    )


def _train_step(model: nn.Module, optimizer: torch.optim.Optimizer, *, scale: float) -> None:
    inputs = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float32,
    ) * float(scale)
    loss = model(inputs).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_value, expected_value)
        return
    assert actual == expected


def _assert_model_equal(actual: nn.Module, expected_state: dict[str, torch.Tensor]) -> None:
    state = actual.state_dict()
    assert state.keys() == expected_state.keys()
    for name in state:
        torch.testing.assert_close(state[name], expected_state[name], rtol=0.0, atol=0.0)


def test_single_rank_checkpoint_is_atomic_and_restores_optimizer(tmp_path: Path) -> None:
    torch.manual_seed(3100)
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    _train_step(model, optimizer, scale=1.0)
    expected_model = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    expected_optimizer = optimizer.state_dict()

    manager = DistributedCheckpointManager(tmp_path / "checkpoints")
    manifest = manager.save(
        "step-12",
        model=model,
        optimizers=optimizer,
        global_step=12,
        policy_version=4,
        metadata={"base_seed": 3100, "environment": "CartPole-v1"},
    )
    assert manifest.saved_world_size == 1
    assert manifest.global_step == 12
    assert manifest.policy_version == 4
    assert manager.read_manifest("step-12") == manifest
    assert not list((tmp_path / "checkpoints").glob("*.incomplete-*"))

    restored_model = _model()
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=3e-4)
    restored_manifest = manager.load(
        "step-12",
        model=restored_model,
        optimizers=restored_optimizer,
    )
    assert restored_manifest == manifest
    _assert_model_equal(restored_model, expected_model)
    _assert_nested_equal(restored_optimizer.state_dict(), expected_optimizer)

    incomplete = tmp_path / "checkpoints" / "incomplete"
    incomplete.mkdir()
    with pytest.raises(IncompleteCheckpointError):
        manager.load(
            "incomplete",
            model=_model(),
            optimizers=torch.optim.Adam(_model().parameters(), lr=3e-4),
        )


def _save_two_rank_worker(
    rank: int,
    world_size: int,
    init_method: str,
    checkpoint_root: str,
    expected_path: str,
) -> None:
    context = DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        backend="gloo",
    )
    context.initialize(init_method=init_method, timeout_seconds=60)
    try:
        torch.manual_seed(3200)
        model = context.wrap(_model())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        _train_step(model, optimizer, scale=float(rank + 1))
        manager = DistributedCheckpointManager(checkpoint_root)
        manifest = manager.save(
            "from-two-ranks",
            model=model,
            optimizers=optimizer,
            global_step=20,
            policy_version=7,
            metadata={"saved_by": "two-rank-ddp"},
        )
        assert manifest.saved_world_size == 2
        if rank == 0:
            base_model = model.module
            torch.save(
                {
                    "model": base_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                expected_path,
            )
        context.barrier()
    finally:
        context.close()


def _load_two_rank_worker(
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
    context.initialize(init_method=init_method, timeout_seconds=60)
    try:
        model = context.wrap(_model())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        manager = DistributedCheckpointManager(checkpoint_root)
        manifest = manager.load(
            "from-one-rank",
            model=model,
            optimizers=optimizer,
        )
        assert manifest.saved_world_size == 1
        torch.save(
            {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": manifest.global_step,
                "policy_version": manifest.policy_version,
            },
            Path(output_dir) / f"rank-{rank}.pt",
        )
        context.barrier()
    finally:
        context.close()


@pytest.mark.distributed
def test_checkpoint_restores_across_one_and_two_rank_topologies(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    expected_two = tmp_path / "expected-two.pt"
    save_init = tmp_path / "save-rendezvous"
    mp.spawn(
        _save_two_rank_worker,
        args=(
            2,
            f"file://{save_init}",
            str(checkpoint_root),
            str(expected_two),
        ),
        nprocs=2,
        join=True,
    )

    expected = torch.load(expected_two, weights_only=True)
    single_model = _model()
    single_optimizer = torch.optim.Adam(single_model.parameters(), lr=1e-3)
    manager = DistributedCheckpointManager(checkpoint_root)
    manifest = manager.load(
        "from-two-ranks",
        model=single_model,
        optimizers=single_optimizer,
    )
    assert manifest.saved_world_size == 2
    assert manifest.global_step == 20
    assert manifest.policy_version == 7
    _assert_model_equal(single_model, expected["model"])
    _assert_nested_equal(single_optimizer.state_dict(), expected["optimizer"])

    _train_step(single_model, single_optimizer, scale=3.0)
    expected_one_model = {
        name: value.detach().clone()
        for name, value in single_model.state_dict().items()
    }
    expected_one_optimizer = single_optimizer.state_dict()
    manager.save(
        "from-one-rank",
        model=single_model,
        optimizers=single_optimizer,
        global_step=21,
        policy_version=8,
        metadata={"saved_by": "one-rank-restore"},
    )

    output_dir = tmp_path / "restored-ranks"
    output_dir.mkdir()
    load_init = tmp_path / "load-rendezvous"
    mp.spawn(
        _load_two_rank_worker,
        args=(
            2,
            f"file://{load_init}",
            str(checkpoint_root),
            str(output_dir),
        ),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        restored = torch.load(output_dir / f"rank-{rank}.pt", weights_only=True)
        assert restored["global_step"] == 21
        assert restored["policy_version"] == 8
        for name, value in expected_one_model.items():
            torch.testing.assert_close(
                restored["model"][name], value, rtol=0.0, atol=0.0
            )
        _assert_nested_equal(restored["optimizer"], expected_one_optimizer)
