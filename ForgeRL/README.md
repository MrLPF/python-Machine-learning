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
- versioned policy registry and coordinator lease/heartbeat contracts;
- PyTorch DDP learner utilities and two-process Gloo CI.

### M1 — local high-throughput runtime and acceptance tooling

- generation-safe shared-memory tensor-tree channels;
- compact slot descriptors instead of pickled tensor payloads;
- cross-actor deadline/minimum-size dynamic inference batching;
- one fair collector thread for all node-local actor endpoints;
- bounded per-endpoint draining to prevent a hot actor from starving peers;
- zero-copy one-request batches and reusable multi-request assembly buffers;
- active/staging double-buffered policy replicas;
- optional pinned-memory, non-blocking H2D and AMP inference path;
- lossless trajectory fragments with separate bootstrap state;
- stable C ABI for vectorized C++ environments and a deterministic reference environment;
- source-fingerprinted v1 Queue/pickle predictor reference;
- same-model v1/v2 synthetic transition audit;
- shared reference PPO core for CartPole/Pendulum learning-quality comparison;
- normal-CI learning plumbing smoke and a pinned self-hosted five-seed workflow.

The previous per-actor-thread implementation is retained as
`ThreadedNodeLocalInferenceService` for regression comparison. The default
`NodeLocalInferenceService` reports collector, allocation, copied-byte, reused-buffer and
single-request zero-copy counters.

M1 is **not performance accepted**. Acceptance still requires a pinned-hardware `>=2x` valid-row
result and a formal five-seed report for both environments whose final `m1-final.json` status is
`GO`. Green hosted CI alone is not that evidence.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,gym]'
pytest -q
python scripts/benchmark_env.py --env CartPole-v1 --steps 10000
python scripts/benchmark_runtime.py --items 200000
python scripts/benchmark_m1_inference.py \
  --actors 4 --requests-per-actor 1000 \
  --collector-poll-ms 0.25 --max-drain-per-endpoint 4
python scripts/benchmark_m1_acceptance.py \
  --actors 2 --requests-per-actor 8 --items-per-request 2 \
  --width 16 --max-batch-items 8 --v2-min-batch-items 4 \
  --throughput-gate 0
python scripts/benchmark_m1_learning.py \
  --smoke --seeds 7 \
  --environments CartPole-v1,Pendulum-v1 \
  --output benchmarks/results/m1-learning-smoke.json
```

## Formal M1 execution

The manual `.github/workflows/forge-rl-m1-formal.yml` workflow targets a pinned self-hosted runner
labelled `forgerl-benchmark`. It runs the formal paired five-seed learning gate, the controlled
synthetic throughput gate, and the final `--require-go` decision without replacing the runner's
pre-provisioned Python environment.

## Initial benchmark environments

- `CartPole-v1`: discrete-action correctness and time-to-target;
- `Pendulum-v1`: continuous-action correctness and time-to-target;
- PettingZoo MPE `simple_spread_v3`: later multi-agent/MAPPO testing;
- optional Gymnasium MuJoCo `Ant-v5`: later cloud scaling and continuous control.

The reference PPO is an M1 acceptance subject, not completion of the M3 algorithm-plugin work. The
C++ counter environment is a runtime/ABI benchmark, not a learning-quality benchmark.

## Repository status

Acceptance gates and remaining work are tracked in [`docs/ROADMAP.md`](docs/ROADMAP.md). The
benchmark methodology is defined in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), and the exact M1
procedure is in [`docs/M1_ACCEPTANCE.md`](docs/M1_ACCEPTANCE.md). The C++ environment boundary is
documented in [`docs/CPP_VECTOR_ENV_ABI.md`](docs/CPP_VECTOR_ENV_ABI.md).
