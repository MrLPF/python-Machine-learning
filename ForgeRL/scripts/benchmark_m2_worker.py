#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket

from forge_rl.benchmarks import M2WorkerConfig, run_m2_worker
from forge_rl.distributed import DistributedContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one ForgeRL M2 weak-scaling subject on the current learner rank."
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=50)
    parser.add_argument("--valid-rows-per-rank", type=int, default=128)
    parser.add_argument("--invalid-rows-per-batch", type=int, default=1)
    parser.add_argument("--observation-width", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--policy-publish-interval", type=int, default=10)
    parser.add_argument(
        "--advertise-host",
        default=os.environ.get("FORGERL_M2_ADVERTISE_HOST", socket.gethostname()),
    )
    parser.add_argument(
        "--physical-node-id",
        default=os.environ.get("FORGERL_M2_NODE_ID", socket.gethostname()),
    )
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--backend", choices=("gloo", "nccl"))
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = DistributedContext.from_env(backend=args.backend)
    context.initialize(timeout_seconds=args.timeout_seconds)
    try:
        report = run_m2_worker(
            context,
            M2WorkerConfig(
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
                valid_rows_per_rank=args.valid_rows_per_rank,
                invalid_rows_per_batch=args.invalid_rows_per_batch,
                observation_width=args.observation_width,
                hidden_size=args.hidden_size,
                learning_rate=args.learning_rate,
                policy_publish_interval=args.policy_publish_interval,
                advertise_host=args.advertise_host,
                physical_node_id=args.physical_node_id,
                controlled=args.controlled,
            ),
        )
        if report is not None:
            rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
            print(rendered, flush=True)
    finally:
        context.close()


if __name__ == "__main__":
    main()
