from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from forge_rl.distributed.learner_group import DistributedContext, all_reduce_mean


def _ddp_worker(rank: int, world_size: int, init_method: str, output_dir: str) -> None:
    context = DistributedContext(rank=rank, world_size=world_size, local_rank=rank, backend="gloo")
    context.initialize(init_method=init_method, timeout_seconds=30)
    try:
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        model = context.wrap(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.tensor([[float(rank + 1)]])
        loss = model(x).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        reduced = all_reduce_mean({"loss": loss.detach()})
        weight = next(model.parameters()).detach().item()
        Path(output_dir, f"rank-{rank}.txt").write_text(f"{weight},{reduced['loss']}")
        context.barrier()
    finally:
        context.close()


@pytest.mark.distributed
def test_two_process_gloo_learner_parameters_stay_equal(tmp_path: Path) -> None:
    init_file = tmp_path / "rendezvous"
    init_method = f"file://{init_file}"
    mp.spawn(
        _ddp_worker,
        args=(2, init_method, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    values = [
        tuple(float(value) for value in (tmp_path / f"rank-{rank}.txt").read_text().split(","))
        for rank in range(2)
    ]
    assert values[0][0] == pytest.approx(values[1][0], abs=1e-7)
    assert values[0][1] == pytest.approx(values[1][1], abs=1e-7)
