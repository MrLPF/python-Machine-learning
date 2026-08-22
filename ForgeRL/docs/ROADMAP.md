# ForgeRL v2 implementation roadmap

## M0 — baseline and correctness gates

- [x] Package the uploaded source without bytecode artifacts.
- [x] Remove import-time dependency on the proprietary simulator.
- [x] Preserve termination/truncation/fault distinctions at the environment boundary.
- [x] Stop recurring loss of the last real GAE transition in each fragment.
- [x] Apply adaptive KL penalty in the standard PPO path.
- [x] Add unit tests for trajectory identity and runtime contracts.
- [x] Add a two-process CPU DDP smoke test.

**Go gate:** zero missing/duplicate step IDs in deterministic collector tests; all CPU CI jobs pass.

## M1 — local high-throughput runtime

- [x] Shared-memory tensor arena.
- [x] Deadline/minimum-size inference batching contract.
- [x] Bounded on-policy queue and policy registry.
- [x] Replace inference tensor payload copies with shared-memory slot descriptors.
- [x] Add double-buffered node-local inference replicas.
- [x] Add vectorized C++ environment ABI and one reference implementation.
- [x] Add pinned-memory/non-blocking H2D pipeline and AMP inference.
- [x] Freeze the uploaded v1 predictor data-plane reference with source hashes.
- [x] Add same-model v1/v2 transition audit and synthetic acceptance report.
- [x] Implement a shared PPO learning core for the v1/v2 CartPole and Pendulum comparison.
- [x] Add deterministic transition, policy-version, time-to-target and normalized-AUC evidence.
- [x] Add a normal-CI learning plumbing smoke that cannot produce a formal `GO`.
- [x] Add a pinned self-hosted workflow for the formal five-seed learning and final M1 gates.
- [x] Preserve event-driven, polling and process-mailbox paths as regression/fallback subjects.
- [x] Add controlled Actor/request/batch tuning and transport matrices.
- [x] Identify timing-based cross-Actor aggregation as the target-topology bottleneck.
- [x] Add `VectorBatchInferenceRuntime` with producer-defined contiguous batch membership.
- [x] Align the formal synthetic throughput subject with `vector-batch`.
- [x] Align the PPO learning-quality subject with the same `vector-batch` runtime.
- [x] Close the reference C++ VectorEnv -> vector-batch -> TransitionBatch -> ExperienceQueue -> PPO update loop.
- [x] Use explicit FP32 tolerances for batched-versus-sliced numerical checks on Python 3.11/3.12.
- [ ] Run the controlled stable >=2x vector-batch throughput gate on the pinned runner.
- [ ] Run the formal five-seed CartPole/Pendulum time-to-target and normalized-AUC gates on the same runner.
- [ ] Persist an `m1-final.json` report whose status is `GO`.

**Go gate:** at least 2x valid-transition throughput over v1 on the controlled vector-batch
benchmark, zero missing/duplicate/invalid rows, numerically equivalent outputs, every required
learning seed reaching its environment target, and no more than 5% degradation in time-to-target
or normalized learning AUC on CartPole/Pendulum. CI smoke success and hosted-runner screening do
not constitute M1 acceptance. The persisted final report must have status `GO`.

The formal M1 subject is now the production local topology:

```text
C++/vectorized EnvRunner
    -> one contiguous observation batch
    -> VectorBatchInferenceRuntime
    -> per-Actor/per-environment output views
```

Timing-based request aggregation remains supported for Python-thread and process-isolated fallback
paths, but it is not used to prove the vectorized production topology. A C++ atomic descriptor ring
is conditional fallback work only for process-isolated Actors after a controlled matrix proves that
Python transport remains below v1; it is not required for the producer-defined vector-batch path.

## M2 — multi-node learner and control plane

- [x] DDP context/wrapping utilities.
- [x] Coordinator lease model.
- [ ] Networked coordinator service.
- [ ] Direct EnvRunner-to-Experience and Learner-to-Policy channels.
- [ ] Distributed checkpoint with topology-independent restore.
- [ ] Failure injection: actor, inference replica and learner-rank restart.

**Go gate:** two nodes, at least two learner ranks, no deadlock, no invalid transition entering a
loss, and >=70% weak-scaling efficiency at the target topology.

M2 implementation is frozen until M1 produces a persisted `m1-final.json` with status `GO`, unless
the user explicitly changes that ordering.

## M3 — algorithms and replay

- [ ] Stable PPO/APPO/V-trace plugins on the v2 batch contract.
- [ ] MAPPO and policy-to-agent mapping.
- [ ] Replay table for SAC/R2D2/offline RL.
- [ ] Burn-in and true truncated BPTT for recurrent policies.
- [ ] League/self-play policy registry.

The reference PPO used by the M1 acceptance harness is intentionally benchmark-only. It does not
mark the M3 algorithm-plugin architecture as complete.

## Stop/reconsider conditions

Pause a custom runtime layer when a maintained external component meets all requirements with less
integration risk. Do not rewrite tensor/autograd, NCCL, CUDA allocation or FSDP. A rewrite must
demonstrate a material improvement in valid rows/s, time-to-target or total cost—not only a higher
raw environment-step counter.
