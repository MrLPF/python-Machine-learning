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
precision, device type and evaluation episodes must be held constant wherever possible.

## Correctness gates

```text
missing_valid_step_ids == 0
duplicate_trained_step_ids == 0
invalid_fault_rows_in_loss == 0
terminated/truncated bootstrap tests pass
GAE and V-trace match hand-computed references within tolerance
checkpoint resume preserves policy/optimizer/step versions
```

## Performance metrics

- environment steps/s and valid training rows/s;
- useful sample ratio = unique valid trained rows / valid collected rows;
- inference items/s, batch-fill ratio and p50/p95/p99 latency;
- policy-lag and queue-age percentiles;
- learner rows/s, GPU utilization, CPU utilization and memory peaks;
- serialization bytes, shared-memory bytes and network bytes;
- strong/weak scaling efficiency;
- time-to-target and learning-curve AUC over at least five seeds.

## Fault matrix

Kill or stall one environment, actor, inference replica and learner rank. Saturate the experience
queue, delay the network, duplicate a descriptor and interrupt checkpoint writing. The runtime
must either recover or fail explicitly; it must never convert an infrastructure fault into a valid
zero-reward terminal transition.
