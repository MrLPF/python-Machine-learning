# ForgeRL v2 implementation roadmap

## M0 — baseline and correctness gates (this PR)

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
- [ ] Replace pickle payloads with shared-memory slot descriptors end-to-end.
- [ ] Add double-buffered node-local inference replicas.
- [ ] Add vectorized C++ environment ABI and one reference implementation.
- [ ] Add pinned-memory/non-blocking H2D pipeline and AMP inference.

**Go gate:** at least 2x valid-transition throughput over v1 on the synthetic benchmark without
more than 5% degradation in time-to-target on CartPole/Pendulum.

## M2 — multi-node learner and control plane

- [x] DDP context/wrapping utilities.
- [x] Coordinator lease model.
- [ ] Networked coordinator service.
- [ ] Direct EnvRunner-to-Experience and Learner-to-Policy channels.
- [ ] Distributed checkpoint with topology-independent restore.
- [ ] Failure injection: actor, inference replica and learner-rank restart.

**Go gate:** two nodes, at least two learner ranks, no deadlock, no invalid transition entering a
loss, and >=70% weak-scaling efficiency at the target topology.

## M3 — algorithms and replay

- [ ] Stable PPO/APPO/V-trace plugins on the v2 batch contract.
- [ ] MAPPO and policy-to-agent mapping.
- [ ] Replay table for SAC/R2D2/offline RL.
- [ ] Burn-in and true truncated BPTT for recurrent policies.
- [ ] League/self-play policy registry.

## Stop/reconsider conditions

Pause a custom runtime layer when a maintained external component meets all requirements with
less integration risk. Do not rewrite tensor/autograd, NCCL, CUDA allocation or FSDP. A rewrite
must demonstrate a material improvement in valid rows/s, time-to-target or total cost—not only a
higher raw environment-step counter.
