# Benchmark and evaluation specification

## Reference environments

| Tier | Environment | Purpose | Normal CI |
|---|---|---|---|
| 0 | deterministic synthetic env | protocol, duplicates, latency and fault injection | yes |
| 1 | Gymnasium `CartPole-v1` | discrete PPO correctness and time-to-target | smoke only |
| 1 | Gymnasium `Pendulum-v1` | continuous PPO correctness and time-to-target | smoke only |
| 2 | PettingZoo MPE `simple_spread_v3` | multi-agent mapping and MAPPO | scheduled |
| 3 | Gymnasium MuJoCo `Ant-v5` | larger continuous-control/cloud scaling | opt-in |

## Compared systems

1. single-process mathematical reference where applicable;
2. source-fingerprinted ForgeRL v1 Queue/pickle predictor path;
3. ForgeRL v2 shared-memory/node-local inference path;
4. Sample Factory on matching PPO/APPO settings;
5. RLlib or TorchRL collector/LearnerGroup on matching resources.

The model architecture, initialized weights, rollout horizon, valid transitions, seeds, optimizer,
CPU thread count, precision, device and evaluation episodes must be held constant wherever
possible. The M1 v1/v2 learning harness uses one shared PPO implementation so the runtime is the
independent variable.

## Correctness gates

```text
missing_valid_step_ids == 0
duplicate_trained_step_ids == 0
unexpected_trained_step_ids == 0
invalid_fault_rows_in_loss == 0
collected_valid_rows == trained_valid_rows
policy_versions are monotonic and match rollout requests
terminated/truncated bootstrap tests pass
GAE and V-trace match hand-computed references within tolerance
checkpoint resume preserves policy/optimizer/step versions
```

## Learning-quality protocol

The formal M1 report contains paired v1/v2 runs for at least five identical seeds on both
`CartPole-v1` and `Pendulum-v1`. Subject order alternates by seed to reduce systematic order bias.
Each subject receives the same fixed environment-step budget even after first reaching the target.

Report at least:

- target-reached seed count;
- median time-to-target;
- normalized learning-curve AUC;
- environment steps and valid trained rows;
- useful-sample ratio;
- per-seed evaluation curves and policy versions;
- p50/p95/p99 inference latency and batch fill;
- complete software/hardware fingerprint.

A run that never reaches the target is explicitly censored and fails the formal all-seeds target
criterion. Normal hosted CI runs a tiny plumbing smoke only and is never formal evidence.

## Performance metrics

- environment steps/s and valid training rows/s;
- useful sample ratio = unique valid trained rows / valid collected rows;
- inference items/s, batch-fill ratio and p50/p95/p99 latency;
- policy-lag and queue-age percentiles;
- learner rows/s, GPU utilization, CPU utilization and memory peaks;
- serialization bytes, shared-memory bytes and network bytes;
- collector thread count, polls and empty-poll ratio;
- input-assembly allocations, copied bytes, reused-buffer batches and zero-copy batches;
- strong/weak scaling efficiency;
- time-to-target and learning-curve AUC over at least five seeds;
- total resource cost to target.

## M1 local inference implementation controls

The default v2 inference service uses one fair round-robin collector for all actor endpoints.
`max_drain_per_endpoint` bounds how many descriptors one endpoint may contribute before the
collector advances, preventing a hot actor from starving the others. The previous one-thread-per-
endpoint service remains exported as `ThreadedNodeLocalInferenceService` only for controlled
regression comparison.

One-request batches are passed directly from the request shared-memory slot to the PyTorch CPU
tensor view. Multi-request batches use schema-keyed, preallocated contiguous buffers that are
reused until the input schema changes. This removes repeated `np.concatenate` allocations while
preserving the copy required to combine physically separate slots.

`benchmark_m1_inference.py` reports both the standard service metrics and the optimization
counters. These counters explain where time and memory traffic go; they do not replace the formal
v1/v2 valid-row throughput gate.

## Formal M1 fairness controls

The self-hosted benchmark runner must use a pinned image and record:

```text
commit SHA
CPU model and affinity
GPU model/driver when applicable
Python, PyTorch, NumPy and Gymnasium versions
OMP/MKL/OpenBLAS/NumExpr thread counts
power/performance settings
```

The formal workflow fixes CPU affinity with `taskset`, verifies expected package versions when
configured, runs the learning gate first, then the synthetic `>=2x` gate, and persists both JSON
reports. One unsuccessful gate keeps M1 unaccepted.

## Fault matrix

Kill or stall one environment, actor, inference replica and learner rank. Saturate the experience
queue, delay the network, duplicate a descriptor and interrupt checkpoint writing. The runtime
must either recover or fail explicitly; it must never convert an infrastructure fault into a valid
zero-reward terminal transition.
