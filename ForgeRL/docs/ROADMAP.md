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
- [ ] Run the controlled >=2x valid-row throughput gate on pinned hardware.
- [ ] Run the formal five-seed CartPole/Pendulum time-to-target and normalized-AUC gates.
- [ ] Persist an `m1-final.json` report whose status is `GO`.

**Go gate:** at least 2x valid-transition throughput over v1 on the controlled synthetic benchmark,
zero missing/duplicate/invalid rows, numerically equivalent outputs, every required learning seed
reaching its environment target, and no more than 5% degradation in time-to-target or normalized
learning AUC on CartPole/Pendulum. CI smoke success alone is not M1 acceptance. The persisted final
report must have status `GO`.

The implementation of a gate is not evidence that the gate passed. Until the two unchecked runs
above complete on the documented pinned runner, M1 remains **not performance accepted**.

## M2 — multi-node learner and control plane

- [x] DDP context/wrapping utilities.
- [x] Coordinator lease model.
- [ ] Networked coordinator service.
- [ ] Direct EnvRunner-to-Experience and Learner-to-Policy channels.
- [ ] Distributed checkpoint with topology-independent restore.
- [ ] Failure injection: actor, inference replica and learner-rank restart.

**Go gate:** two nodes, at least two learner ranks, no deadlock, no invalid transition entering a
loss, and >=70% weak-scaling efficiency at the target topology.

M2 code may be developed behind isolated interfaces while M1 evidence is collected, but M1 must
not be marked performance-accepted or merged as such before `docs/M1_ACCEPTANCE.md` produces a
`GO` report.

## M3 — algorithms and replay

- [ ] Stable PPO/APPO/V-trace plugins on the v2 batch contract.
- [ ] MAPPO and policy-to-agent mapping.
- [ ] Replay table for SAC/R2D2/offline RL.
- [ ] Burn-in and true truncated BPTT for recurrent policies.
- [ ] League/self-play policy registry.

The reference PPO used by the M1 acceptance harness is intentionally benchmark-only. It does not
mark the M3 algorithm-plugin architecture as complete.

## Stop/reconsider conditions

Pause a custom runtime layer when a maintained external component meets all requirements with
less integration risk. Do not rewrite tensor/autograd, NCCL, CUDA allocation or FSDP. A rewrite
must demonstrate a material improvement in valid rows/s, time-to-target or total cost—not only a
higher raw environment-step counter.
