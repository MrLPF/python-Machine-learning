# Benchmark and evaluation specification

## Reference environments

| Tier | Environment | Purpose | Normal CI |
|---|---|---|---|
| 0 | deterministic synthetic env | protocol, duplicates, latency and fault injection | yes |
| 1 | Gymnasium `CartPole-v1` | discrete PPO correctness and time-to-target | yes |
| 1 | Gymnasium `Pendulum-v1` | continuous action schema and throughput | yes |
| 2 | PettingZoo MPE `simple_spread_v3` | multi-agent mapping and MAPPO | scheduled |
| 3 | Gymnasium MuJoCo `Ant-v5` | larger continuous-control/cloud scaling | opt-in |

## Compared systems

1. single-process reference implementation;
2. frozen uploaded ForgeRL v1;
3. ForgeRL v2;
4. Sample Factory on matching PPO/APPO settings;
5. RLlib or TorchRL collector/LearnerGroup on matching resources.

The model architecture, rollout horizon, total valid transitions, seeds, CPU thread count,
precision, device type and evaluation episodes must be held constant.

The uploaded v1 source fingerprint is stored in `benchmarks/legacy_v1_manifest.json`. The public
frozen runtime is deliberately limited to the Queue/pickle predictor data path and does not expose
simulator-specific code. See `docs/M1_ACCEPTANCE.md` for the exact comparison procedure.

## Correctness gates

```text
missing_valid_step_ids == 0
duplicate_trained_step_ids == 0
unexpected_trained_step_ids == 0
invalid_fault_rows_in_loss == 0
terminated/truncated bootstrap tests pass
GAE and V-trace match hand-computed references within tolerance
checkpoint resume preserves policy/optimizer/step versions
```

Every benchmark result must include a source commit, environment fingerprint, configuration and
transition audit. Raw environment steps/s without a useful-sample ratio is not an accepted result.

## Performance metrics

- environment steps/s and valid training rows/s;
- useful sample ratio = unique valid trained rows / valid collected rows;
- inference items/s, batch-fill ratio and p50/p95/p99 latency;
- policy-lag and queue-age percentiles;
- learner rows/s, GPU utilization, CPU utilization and memory peaks;
- serialization bytes, shared-memory bytes and network bytes;
- strong/weak scaling efficiency;
- time-to-target and learning-curve AUC over at least five seeds.

## M1 gate

M1 requires all of the following:

```text
transition audit passes
v1/v2 inference checksum relative error <= 1e-6
v2 valid-row throughput / v1 valid-row throughput >= 2.0
CartPole and Pendulum each use at least five seeds
v2 median time-to-target <= 1.05 * v1
v2 normalized learning-curve AUC >= 0.95 * v1
```

The CI acceptance smoke test validates contracts and emits a JSON artifact, but it uses a disabled
throughput threshold because shared hosted runners are not stable performance hardware. It must
not be interpreted as an M1 `GO` result.

## Fault matrix

Kill or stall one environment, actor, inference replica and learner rank. Saturate the experience
queue, delay the network, duplicate a descriptor and interrupt checkpoint writing. The runtime
must either recover or fail explicitly; it must never convert an infrastructure fault into a valid
zero-reward terminal transition.
