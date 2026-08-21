# ForgeRL

ForgeRL is a high-throughput distributed reinforcement-learning runtime built on PyTorch. PyTorch
continues to own tensor kernels, autograd, CUDA allocation, mixed precision, DDP and future FSDP2
support. ForgeRL owns the reinforcement-learning dataflow:

- vectorized environment execution and fault isolation;
- node-local, deadline-aware inference batching;
- versioned and atomically activated policies;
- trajectory/replay consistency and backpressure;
- multi-process and multi-node learner orchestration;
- PPO/APPO/V-trace first, with extension points for MAPPO, SAC and R2D2.

The implementation is being migrated incrementally from the uploaded ForgeRL v1 design. It is
intentionally **not** a replacement for PyTorch or OneFlow's tensor/autograd layer.

## Implemented foundations

### M0 — correctness and distributed foundation

- clean installation without simulator-specific dependencies;
- explicit `terminated`, `truncated` and invalid-system-transition contracts;
- validated `TransitionBatch` with unique step identity;
- bounded on-policy experience queue with policy-lag and age filtering;
- versioned policy registry;
- coordinator lease/heartbeat contract;
- PyTorch DDP learner utilities and two-process Gloo CI.

### M1 — local high-throughput runtime

- generation-safe shared-memory tensor-tree channels;
- compact slot descriptors instead of pickled tensor payloads;
- cross-actor deadline/minimum-size dynamic inference batching;
- active/staging double-buffered policy replicas;
- optional pinned-memory, non-blocking H2D and AMP inference path;
- `TrajectoryBuilder` with separate bootstrap state and no lost final transition;
- stable C ABI for vectorized C++ environments;
- deterministic C++ counter-environment reference implementation.

M1 code is implemented, but the performance go gate remains open until it is benchmarked against
the frozen v1 implementation on matching hardware and time-to-target tests.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,gym]'
pytest -q
python scripts/benchmark_env.py --env CartPole-v1 --steps 10000
python scripts/benchmark_runtime.py --items 200000
python scripts/benchmark_m1_inference.py --actors 4 --requests-per-actor 1000
```

## Initial benchmark environments

- `CartPole-v1`: discrete-action correctness and time-to-target;
- `Pendulum-v1`: continuous-action protocol and throughput;
- PettingZoo MPE `simple_spread_v3`: phase-2 multi-agent/MAPPO test;
- optional Gymnasium MuJoCo `Ant-v5`: later cloud scaling and continuous control.

The C++ counter environment is a runtime/ABI benchmark, not a learning-quality benchmark.

## Repository status

Acceptance gates and remaining work are tracked in [`docs/ROADMAP.md`](docs/ROADMAP.md). The
benchmark methodology is defined in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). The new C++
environment boundary is documented in
[`docs/CPP_VECTOR_ENV_ABI.md`](docs/CPP_VECTOR_ENV_ABI.md).
