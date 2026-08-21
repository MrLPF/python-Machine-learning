# M1 acceptance procedure

M1 is accepted only after **correctness**, **synthetic data-plane throughput**, and **learning
quality** pass on the same controlled hardware. A fast microbenchmark, a normal hosted CI run, or
the existence of an acceptance script cannot produce a `GO` decision.

## Frozen v1 reference

The user-provided source archive is identified by `benchmarks/legacy_v1_manifest.json`. The
repository publishes only a benchmark-oriented behavioral reference of the v1 central predictor
data path:

```text
actor NumPy payload
  -> multiprocessing.Queue / pickle
  -> central predictor process
  -> per-actor multiprocessing.Queue / pickle
```

Simulator-specific and domain adapters are intentionally excluded. The manifest records the
archive and relevant source-file SHA-256 digests so the baseline cannot be silently changed.

For the learning gate, the frozen data plane accepts an atomic policy-update control message. The
update is acknowledged only after a complete state dict has loaded between rollout rounds. This is
required to train v1 and v2 with the same PPO core; it does not change the frozen Queue/pickle
inference transport being measured.

## Current v2 local inference subject

The default v2 subject now routes compact request descriptors through one node-level event-driven
queue. Request and response tensors remain in generation-protected shared-memory slots. The first
notification blocks the collector; subsequent ready descriptors are drained in a bounded burst.
This replaces the previous endpoint-by-endpoint empty-poll loop while retaining exactly one
collector and one inference worker.

The prior implementations remain available only for regression comparison:

```text
PollingNodeLocalInferenceService
ThreadedNodeLocalInferenceService
```

The event-driven queue is validated from normal Actor threads and from a spawned Actor process.
Stale descriptors are counted and ignored; an unknown Actor or Actor-metadata mismatch is treated
as an explicit runtime protocol error rather than a valid transition.

## Synthetic acceptance benchmark

`benchmark_m1_acceptance.py` executes the same deterministic PyTorch model, observations, request
identities, Actor count, device and precision through:

1. `LegacyV1InferenceRuntime`, using the frozen Queue/pickle path;
2. the default event-driven `NodeLocalInferenceService`, using shared-memory tensor channels and
   cross-Actor batching.

It reports valid rows/s, v2/v1 speedup, transition identity errors, output equivalence,
p50/p95/p99 latency, batch fill, data-movement counters and the software/hardware fingerprint.

A normal-CI correctness smoke deliberately disables the speed gate:

```bash
python scripts/benchmark_m1_acceptance.py \
  --actors 2 --requests-per-actor 8 --items-per-request 2 \
  --width 16 --max-batch-items 8 \
  --v2-min-batch-items 4 --legacy-min-batch-items 1 \
  --throughput-gate 0 \
  --output benchmarks/results/m1-acceptance-smoke.json
```

The controlled synthetic gate uses the documented M1 threshold:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmark_m1_acceptance.py \
  --actors 8 --requests-per-actor 2000 --items-per-request 8 \
  --width 64 --max-batch-items 128 \
  --v2-min-batch-items 32 --legacy-min-batch-items 1 \
  --throughput-gate 2.0 \
  --output benchmarks/results/m1-acceptance-synthetic.json
```

Hosted-runner measurements are regression signals only. The formal result must be repeated on a
pinned image with fixed CPU affinity, thread counts, PyTorch/Gymnasium versions and power settings.
Both subjects use the same device and precision.

## Controlled tuning before C++ escalation

The event-driven path must be measured across the documented workload matrix before implementing a
custom C++ queue. A CI smoke validates the matrix runner and JSON schema but is never formal:

```bash
python scripts/benchmark_m1_tuning_matrix.py \
  --preset smoke --repeats 1 --warmup-repeats 0 \
  --requests-per-actor 8 \
  --output benchmarks/results/m1-tuning-matrix-smoke.json
```

The pinned screening command is:

```bash
taskset -c "$FORGERL_CPUSET" \
  python scripts/benchmark_m1_tuning_matrix.py \
    --preset controlled --controlled \
    --repeats 3 --warmup-repeats 1 \
    --requests-per-actor 1000 \
    --output benchmarks/results/m1-tuning-matrix-controlled.json
```

Each case runs the complete v1/v2 correctness and numerical-equivalence checks. A stable
`>=2.0x` case with speedup coefficient of variation `<=0.10` becomes a candidate for the fixed
formal gate; it does not itself produce `GO`.

A C++ atomic descriptor ring is the next M1 experiment only when all of these conditions hold:

```text
controlled pinned run
at least three measured repeats per case
all transition and numerical gates pass
all cases with actors >= 8 have median v2/v1 speedup < 1.0
matrix status == CPP_DESCRIPTOR_RING_RECOMMENDED
```

Otherwise the project continues tuning the Python event-driven path. This stop/go rule prevents a
C++ rewrite based on a noisy hosted-runner result or one undersized workload.

## Learning-quality gate

`benchmark_m1_learning.py` supplies the executable evidence generator. It uses one
`ReferencePPOPolicy`, one PPO loss implementation, identical initial weights, identical GAE,
identical minibatch order, identical rollout budget and deterministic per-Actor action RNGs for
both subjects. The intentional difference is only the inference data plane.

Supported reference environments are:

- `CartPole-v1`: categorical policy, target mean return `475`;
- `Pendulum-v1`: tanh-squashed Gaussian policy, target mean return `-200`.

The GAE boundary rules are:

```text
terminated -> no value bootstrap and stop recursion
truncated  -> bootstrap from final observation but stop recursion
runtime fault -> invalid row; never enter target, advantage or loss
```

Each run records unique transition identities, policy versions, valid collected/trained rows,
inference metrics, evaluation curves, first time-to-target and normalized learning AUC. Training
continues for the full fixed budget after first reaching the target so AUC comparisons use equal
budgets.

### Normal-CI plumbing smoke

```bash
python scripts/benchmark_m1_learning.py \
  --smoke \
  --seeds 7 \
  --environments CartPole-v1,Pendulum-v1 \
  --device cpu \
  --output benchmarks/results/m1-learning-smoke.json
```

Smoke mode uses one seed and a tiny update budget. It can return only `SMOKE_PASS` or a failure;
its report sets `formal_eligible=false` and cannot satisfy the final M1 gate.

### Formal five-seed run

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmark_m1_learning.py \
  --seeds 1,2,3,4,5 \
  --environments CartPole-v1,Pendulum-v1 \
  --device cpu \
  --require-learning-gate \
  --output benchmarks/results/m1-learning-formal.json
```

For each environment, formal acceptance requires:

```text
paired seeds >= 5
all required v1 seeds reach the target
all required v2 seeds reach the target
v2 median time-to-target <= 1.05 * v1 median time-to-target
v2 median normalized AUC >= 0.95 * v1 median normalized AUC
transition and policy-version audits pass for every run
```

The JSON keeps the compatibility fields consumed by `LearningGate`, plus detailed target-reach
counts and per-seed evidence. A failed or censored seed is not presented as a successful
time-to-target observation.

## Pinned formal workflow

`.github/workflows/forge-rl-m1-formal.yml` is a manual workflow restricted to a self-hosted runner
with labels:

```text
self-hosted, linux, x64, forgerl-benchmark
```

The runner must already contain the pinned Python/PyTorch/NumPy/Gymnasium environment and define
`FORGERL_CPUSET`. Optional `FORGERL_EXPECTED_TORCH` and `FORGERL_EXPECTED_GYMNASIUM` values make
version drift fatal. The workflow does not replace the runner's Python installation.

It executes, in order:

1. the formal five-seed learning gate;
2. the controlled synthetic `>=2x` gate;
3. `benchmark_m1_acceptance.py --learning-report ... --require-go`;
4. upload of `m1-learning-formal.json` and `m1-final.json`, including on failure.

The final command is equivalent to:

```bash
python scripts/benchmark_m1_acceptance.py \
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

M2 may be developed behind isolated interfaces, but the project must not describe M1 as
performance accepted until a persisted `m1-final.json` has status `GO`.
