# Migration from ForgeRL v1

The migration separates the new runtime kernel from the existing v1 trainer. The uploaded v1
implementation remains the behavioral reference until each component passes its correctness and
performance gates.

## Source freeze

`benchmarks/legacy_v1_manifest.json` records the SHA-256 digest of the uploaded archive and the
v1 predictor/protocol/trajectory/queue sources used to derive the public benchmark reference. The
complete archive is not copied into the repository because it includes simulator-specific code
that is outside the public runtime scope.

`LegacyV1InferenceRuntime` preserves the relevant M1 behavior: NumPy tensor payloads are copied
through multiprocessing Queue/pickle into a central predictor process and returned through
per-actor queues. It is a data-plane baseline, not a claim that the full v1 trainer has been
published or accepted as mathematically correct.

## P0 fixes already represented by v2 contracts

- simulator-specific imports are lazy rather than required for generic Gym usage;
- `terminated`, `truncated`, invalid infrastructure transitions and bootstrap masks are distinct;
- trajectory fragments no longer discard one real transition at every boundary;
- infrastructure faults are excluded from policy/value losses;
- predictor overflow is deferred instead of dropped or rejected;
- every valid row has a unique `(actor, env, episode, step)` identity.

## Migration sequence

1. [x] Adapt trajectory output to `TransitionBatch` and enforce unique row identities.
2. [x] Replace predictor tensor payload copies with shared-memory slot descriptors.
3. [x] Move predictor requests to deadline batching and atomically versioned policies.
4. [x] Route rollout contracts through a bounded on-policy queue with lag/age limits.
5. [x] Run learner modules through the PyTorch DDP context in local distributed tests.
6. [x] Freeze and fingerprint the v1 predictor data plane for same-model M1 comparison.
7. [ ] Complete the controlled throughput and five-seed learning acceptance gates.
8. [ ] Replace the in-memory coordinator with a networked service without changing runtime-facing
   interfaces.
9. [ ] Add direct cross-node experience and policy channels, distributed checkpointing and fault
   recovery.
10. [ ] Replace Python hot paths with C++/CUDA only when profiling demonstrates a material
    bottleneck.

The next implementation step after this benchmark commit is the five-seed learning harness. M2
network code must stay isolated from the M1 comparison so it cannot change the baseline workload.
