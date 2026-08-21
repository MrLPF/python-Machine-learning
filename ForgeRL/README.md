# ForgeRL

ForgeRL is a high-throughput distributed reinforcement-learning runtime. It keeps tensor
execution, autograd, mixed precision, DDP and future FSDP2 support in PyTorch, while ForgeRL
owns the RL-specific runtime:

- environment runners and fault isolation;
- node-local, deadline-aware inference batching;
- versioned policy publication;
- trajectory/replay consistency and backpressure;
- multi-process and multi-node learner orchestration;
- PPO/APPO/V-trace first, with extension points for MAPPO, SAC and R2D2.

This branch is the first ForgeRL v2 implementation milestone. It extracts a clean runtime
foundation from the existing v1 design and defines the contracts required for the staged v1
migration described in `docs/MIGRATION_FROM_V1.md`.
It is intentionally **not** a replacement for PyTorch or OneFlow's tensor/autograd layer.

## Implemented in milestone 1

- clean package installation without simulator-specific dependencies;
- explicit `terminated`, `truncated` and invalid-system-transition contracts;
- deadline-aware, overflow-safe dynamic batching;
- validated `TransitionBatch` contract with unique step identity;
- bounded on-policy experience queue with policy-lag and age filtering;
- versioned policy registry;
- shared-memory tensor arena for zero-copy local data transport;
- coordinator lease/heartbeat contract;
- PyTorch DDP learner-group utilities and a two-process Gloo smoke test;
- CPU CI, distributed CI and benchmark scripts.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,gym]'
pytest -q
python scripts/benchmark_env.py --env CartPole-v1 --steps 10000
python scripts/benchmark_runtime.py --items 200000
```

## Benchmark environments

The initial reproducible matrix uses environments without proprietary assets:

- `CartPole-v1`: discrete-action correctness and CI learning smoke test;
- `Pendulum-v1`: continuous-action protocol and throughput test;
- PettingZoo MPE `simple_spread_v3`: phase-2 multi-agent/MAPPO test;
- optional Gymnasium MuJoCo `Ant-v5`: phase-3 cloud scaling and continuous-control test.

The first two run on ordinary GitHub-hosted CPU runners. GPU and multi-node workflows are
provided as opt-in/self-hosted jobs so normal CI cannot silently incur cloud cost.

## Project status

The current code is an implementation foundation, not a claim that multi-node performance is
already production-ready. Acceptance gates and remaining work are tracked in
[`docs/ROADMAP.md`](docs/ROADMAP.md) and [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
