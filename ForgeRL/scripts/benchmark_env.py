#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np


def _sample_action(space: Any, rng: np.random.Generator) -> Any:
    if hasattr(space, "n"):
        return int(rng.integers(int(space.n)))
    if hasattr(space, "low") and hasattr(space, "high"):
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        return rng.uniform(low, high).astype(getattr(space, "dtype", np.float32))
    return space.sample()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit("install ForgeRL with the gym extra: pip install -e '.[gym]'") from exc

    rng = np.random.default_rng(args.seed)
    envs = [gym.make(args.env) for _ in range(args.envs)]
    episodes = 0
    try:
        for index, env in enumerate(envs):
            env.reset(seed=args.seed + index)
        started = time.perf_counter()
        completed = 0
        while completed < args.steps:
            for env in envs:
                action = _sample_action(env.action_space, rng)
                _obs, _reward, terminated, truncated, _info = env.step(action)
                completed += 1
                if terminated or truncated:
                    env.reset()
                    episodes += 1
                if completed >= args.steps:
                    break
        elapsed = time.perf_counter() - started
    finally:
        for env in envs:
            env.close()
    print(
        json.dumps(
            {
                "environment": args.env,
                "env_instances": args.envs,
                "steps": completed,
                "episodes": episodes,
                "seconds": elapsed,
                "steps_per_second": completed / max(elapsed, 1e-12),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
