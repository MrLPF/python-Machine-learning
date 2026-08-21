#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from forge_rl.benchmarks import LearningBenchmarkConfig, run_learning_benchmark


def _integers(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled ForgeRL v1/v2 PPO learning-quality gate."
    )
    parser.add_argument("--seeds", type=_integers, default=(1, 2, 3, 4, 5))
    parser.add_argument(
        "--environments",
        type=_strings,
        default=("CartPole-v1", "Pendulum-v1"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--actors", type=int, default=4)
    parser.add_argument("--envs-per-actor", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--cartpole-updates", type=int)
    parser.add_argument("--pendulum-updates", type=int)
    parser.add_argument("--cartpole-eval-interval", type=int)
    parser.add_argument("--pendulum-eval-interval", type=int)
    parser.add_argument("--max-batch-items", type=int, default=128)
    parser.add_argument("--v2-min-batch-items", type=int, default=32)
    parser.add_argument("--legacy-min-batch-items", type=int, default=1)
    parser.add_argument("--max-wait-ms", type=float, default=2.0)
    parser.add_argument("--request-slots", type=int, default=8)
    parser.add_argument(
        "--output",
        default="benchmarks/results/m1-learning.json",
    )
    parser.add_argument("--require-learning-gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        config = LearningBenchmarkConfig.smoke_config(
            seeds=args.seeds,
            environments=args.environments,
            device=args.device,
        )
    else:
        config = LearningBenchmarkConfig(
            environments=args.environments,
            seeds=args.seeds,
            actors=args.actors,
            envs_per_actor=args.envs_per_actor,
            rollout_steps=args.rollout_steps,
            hidden_size=args.hidden_size,
            eval_episodes=args.eval_episodes,
            max_batch_items=args.max_batch_items,
            v2_min_batch_items=args.v2_min_batch_items,
            legacy_min_batch_items=args.legacy_min_batch_items,
            max_wait_ms=args.max_wait_ms,
            request_slots=args.request_slots,
            device=args.device,
            cartpole_max_updates=args.cartpole_updates,
            pendulum_max_updates=args.pendulum_updates,
            cartpole_eval_interval=args.cartpole_eval_interval,
            pendulum_eval_interval=args.pendulum_eval_interval,
        )
    report = run_learning_benchmark(config)
    report.save(args.output)
    print(
        json.dumps(
            {
                "status": report.status,
                "formal_eligible": report.formal_eligible,
                "learning_gate_passed": report.learning_gate_passed,
                "environments": list(report.environments),
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_learning_gate and not report.learning_gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
