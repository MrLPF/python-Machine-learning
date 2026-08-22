# M1 acceptance procedure

M1 is accepted only after **correctness**, **stable synthetic throughput**, and **learning quality**
pass on the same controlled hardware. Hosted CI, a fast microbenchmark, or the existence of an
acceptance script cannot produce `GO`.

## Frozen v1 reference

The user-provided source archive is identified by `benchmarks/legacy_v1_manifest.json`. The v1
benchmark subject remains:

```text
Actor NumPy payload
  -> multiprocessing.Queue / pickle
  -> central predictor process
  -> per-Actor multiprocessing.Queue / pickle
```

The learning-only policy-update control message is acknowledged between rollout rounds and does
not change the frozen transport being measured.

## Formal v2 subject

The formal local v2 subject is the production vectorized topology:

```text
C++/vectorized EnvRunner
  -> one contiguous observation batch
  -> VectorBatchInferenceRuntime
  -> one policy forward
  -> per-Actor/per-environment output views
```

Batch membership is producer-defined. It does not depend on Python Actor arrival order, a timing
window, or reconstructing a batch after requests cross an IPC boundary. The runtime preserves:

- monotonic policy versions and atomic active/staging switches;
- one sequence ID per Actor wave;
- CPU/CUDA and optional CUDA AMP execution;
- explicit output partition checks;
- batch, item, error and latency metrics.

Event-driven shared-memory, polling, rendezvous and process-mailbox paths remain regression or
fallback subjects. They do not replace the vector-batch subject in the final M1 report.

## Correctness gates

Every formal synthetic and learning run must satisfy:

```text
missing valid rows = 0
duplicate collected rows = 0
duplicate trained rows = 0
unexpected trained rows = 0
invalid infrastructure rows in loss = 0
useful sample ratio = 1.0
runtime errors = 0
```

FP32 batched and sliced forward passes are compared with explicit floating-point tolerances rather
than bitwise equality. Shape, finite values, policy version, sequence identity and transition
identity remain exact checks.

The reference C++ pipeline test covers:

```text
CppVectorEnv.step
  -> VectorBatchInferenceRuntime
  -> TransitionBatch
  -> OnPolicyExperienceQueue
  -> one PPO optimizer update
  -> policy version refresh
```

## Synthetic throughput gate

The one-run final subject is invoked explicitly:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmark_m1_acceptance.py \
  --actors 8 --requests-per-actor 2000 --items-per-request 8 \
  --width 64 --max-batch-items 128 \
  --v2-min-batch-items 64 --legacy-min-batch-items 1 \
  --v2-runtime vector-batch \
  --throughput-gate 2.0 \
  --output benchmarks/results/m1-acceptance-synthetic.json
```

Before the final one-run gate, the controlled runner must demonstrate stability with one warmup and
at least three measured repeats:

```bash
taskset -c "$FORGERL_CPUSET" \
  python scripts/benchmark_m1_double_buffer.py \
    --actors 8 --requests-per-actor 2000 --items-per-request 8 \
    --width 64 --output-size 4 --max-batch-items 128 \
    --min-batch-items 64 --max-wait-ms 0.5 \
    --v2-runtime vector-batch \
    --repeats 3 --warmup-repeats 1 --controlled \
    --output benchmarks/results/m1-vector-batch-controlled.json
```

The controlled report must show:

```text
status = DOUBLE_BUFFER_CANDIDATE
formal_eligible = true
double_buffer_stable_two_x = true
speedup coefficient of variation <= 0.10
identical total valid work for compared cases
all correctness gates pass
```

Hosted-runner results remain screening signals only.

## Learning-quality gate

`benchmark_m1_learning.py` uses one PPO implementation, identical initial weights, GAE, minibatch
order, rollout budget and per-Actor random-number streams for v1 and v2. The v2 subject uses the
same `VectorBatchInferenceRuntime` as the synthetic gate. Actor threads write into fixed slices of
one contiguous wave buffer; a barrier action performs exactly one vector-batch forward.

Reference environments:

- `CartPole-v1`: target mean return `475`;
- `Pendulum-v1`: target mean return `-200`.

GAE boundary rules:

```text
terminated -> no value bootstrap and stop recursion
truncated  -> bootstrap from final observation but stop recursion
runtime fault -> invalid row; never enter target, advantage or loss
```

Formal command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmark_m1_learning.py \
  --seeds 1,2,3,4,5 \
  --environments CartPole-v1,Pendulum-v1 \
  --device cpu \
  --require-learning-gate \
  --output benchmarks/results/m1-learning-formal.json
```

For each environment:

```text
paired seeds >= 5
all required v1 seeds reach the target
all required v2 seeds reach the target
v2 median time-to-target <= 1.05 * v1 median time-to-target
v2 median normalized AUC >= 0.95 * v1 median normalized AUC
all transition and policy-version audits pass
all v2 runtime records identify vector-batch-learning
```

A censored seed is not reported as a successful time-to-target observation.

## Pinned formal workflow

`.github/workflows/forge-rl-m1-formal.yml` is a manual workflow restricted to:

```text
self-hosted, linux, x64, forgerl-benchmark
```

The runner must define `FORGERL_CPUSET` and contain the pinned Python, PyTorch, NumPy and Gymnasium
environment. Optional expected-version variables make drift fatal.

The workflow executes on the same runner and CPU affinity:

1. five-seed CartPole/Pendulum vector-batch learning gate;
2. three-repeat controlled vector-batch stability gate;
3. final vector-batch synthetic gate with the learning report attached;
4. upload of `m1-learning-formal.json`, `m1-vector-batch-controlled.json` and `m1-final.json`.

The final command uses:

```bash
python scripts/benchmark_m1_acceptance.py \
  --v2-runtime vector-batch \
  --throughput-gate 2.0 \
  --learning-report benchmarks/results/m1-learning-formal.json \
  --require-go \
  --output benchmarks/results/m1-final.json
```

## Final status values

| Status | Meaning |
|---|---|
| `FAIL_CORRECTNESS` | loss/duplication/invalid-row, policy-version or runtime error |
| `FAIL_NUMERICAL_EQUIVALENCE` | v1/v2 model outputs differ beyond tolerance |
| `FAIL_SYNTHETIC_THROUGHPUT` | valid-row speedup is below the configured gate |
| `PENDING_LEARNING_GATE` | runtime gates pass but formal learning evidence is absent |
| `FAIL_LEARNING_GATE` | target reach, time-to-target or normalized AUC gate failed |
| `GO` | all documented M1 gates passed on the pinned runner |

M1 is not performance accepted until a persisted `m1-final.json` has status `GO`. M2 remains
frozen unless the user explicitly changes the milestone order.
