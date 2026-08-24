#!/usr/bin/env python3
from __future__ import annotations

import json

import torch
from torch import nn

from forge_rl.distributed.learner_group import DistributedContext, all_reduce_mean


def main() -> None:
    context = DistributedContext.from_env()
    context.initialize(timeout_seconds=120)
    try:
        torch.manual_seed(7)
        model = context.wrap(nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 2)))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        inputs = torch.full((32, 8), float(context.rank + 1), device=context.device)
        targets = torch.zeros((32, 2), device=context.device)
        output = model(inputs)
        loss = (output - targets).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics = all_reduce_mean({"loss": loss.detach()}, device=context.device)
        context.barrier()
        print(
            json.dumps(
                {
                    "rank": context.rank,
                    "world_size": context.world_size,
                    "backend": context.backend,
                    **metrics,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        context.close()


if __name__ == "__main__":
    main()
