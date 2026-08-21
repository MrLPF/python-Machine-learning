# M1 acceptance procedure

M1 is accepted only after **correctness**, **synthetic data-plane throughput**, and **learning
quality** pass on the same controlled hardware. A fast microbenchmark alone cannot produce a
`GO` decision.

## Frozen v1 reference

The user-provided source archive is identified by
`benchmarks/legacy_v1_manifest.json`. The repository publishes only a benchmark-oriented
behavioral reference of the v1 central predictor data path:

```text
actor NumPy payload
  -> multiprocessing.Queue / pickle
  -> central predictor process
  -> per-actor multiprocessing.Queue / pickle
```

Simulator-specific and domain adapters are intentionally excluded. The manifest records the
archive and relevant source-file SHA-256 digests so the baseline can be audited without silently
changing it.

## Synthetic acceptance benchmark

`benchmark_m1_acceptance.py` executes the same deterministic PyTorch model, observations,
request identities, actor count and item count through:

1. `LegacyV1InferenceRuntime`, using the frozen Queue/pickle path;
2. `NodeLocalInferenceService`, using shared-memory tensor channels and cross-actor batching.

It reports:

- valid rows/s and v2/v1 speedup;
- missing, duplicate, unexpected and invalid training rows;
- output checksum equivalence;
- p50/p95/p99 latency and batch fill;
- serialized Queue bytes for v1 and tensor payload bytes for v2;
- environment and GitHub commit fingerprint.

A CI correctness smoke run deliberately disables the speed gate:

```bash
python scripts/benchmark_m1_acceptance.py \
  --actors 2 --requests-per-actor 8 --items-per-request 2 \
  --width 16 --max-batch-items 8 \
  --v2-min-batch-items 4 --legacy-min-batch-items 1 \
  --throughput-gate 0 \
  --output benchmarks/results/m1-acceptance-smoke.json
```

A controlled synthetic screening run uses the documented M1 threshold:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/benchmark_m1_acceptance.py \
  --actors 8 --requests-per-actor 2000 --items-per-request 8 \
  --width 64 --max-batch-items 128 \
  --v2-min-batch-items 32 --legacy-min-batch-items 1 \
  --throughput-gate 2.0 \
  --output benchmarks/results/m1-acceptance-synthetic.json
```

Hosted CI measurements are regression signals. The formal result must be repeated on a pinned
machine image with fixed CPU affinity, thread counts, PyTorch version and power/performance
settings. Both subjects use the same `--device` and precision. CUDA screening is available with
`--device cuda`; optional autocast must be enabled symmetrically with `--amp-dtype float16` or
`--amp-dtype bfloat16`, and the checksum tolerance must be documented explicitly.

## Learning gate

An overall `GO` also requires at least five seeds for both `CartPole-v1` and `Pendulum-v1`. The
learning report uses normalized AUC values where larger is better:

```json
{
  "environments": [
    {
      "environment": "CartPole-v1",
      "seeds": 5,
      "v1_time_to_target_median_seconds": 120.0,
      "v2_time_to_target_median_seconds": 118.0,
      "v1_normalized_auc": 0.82,
      "v2_normalized_auc": 0.83
    },
    {
      "environment": "Pendulum-v1",
      "seeds": 5,
      "v1_time_to_target_median_seconds": 300.0,
      "v2_time_to_target_median_seconds": 305.0,
      "v1_normalized_auc": 0.74,
      "v2_normalized_auc": 0.73
    }
  ]
}
```

The final command is:

```bash
python scripts/benchmark_m1_acceptance.py \
  --throughput-gate 2.0 \
  --learning-report benchmarks/results/m1-learning.json \
  --require-go \
  --output benchmarks/results/m1-final.json
```

Each environment passes only when:

```text
seeds >= 5
v2 median time-to-target <= 1.05 * v1 median time-to-target
v2 normalized AUC >= 0.95 * v1 normalized AUC
```

## Status values

| Status | Meaning |
|---|---|
| `FAIL_CORRECTNESS` | loss/duplication/invalid-row or runtime error |
| `FAIL_NUMERICAL_EQUIVALENCE` | v1/v2 model outputs differ beyond tolerance |
| `FAIL_SYNTHETIC_THROUGHPUT` | valid-row speedup is below the configured gate |
| `PENDING_LEARNING_GATE` | runtime gates pass but five-seed learning evidence is absent |
| `FAIL_LEARNING_GATE` | time-to-target or normalized AUC regressed beyond tolerance |
| `GO` | all documented M1 gates pass |

M2 implementation may be developed behind isolated interfaces, but the project must not describe
M1 as performance-accepted until a persisted report has status `GO`.
