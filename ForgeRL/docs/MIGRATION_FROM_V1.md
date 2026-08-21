# Migration from ForgeRL v1

This bootstrap deliberately separates the new runtime kernel from the existing v1 trainer.
The v1 implementation remains the behavioural reference until each component passes the
correctness and performance gates below.

## P0 fixes already validated locally

- simulator-specific imports are lazy rather than required for generic Gym usage;
- `terminated`, `truncated`, invalid infrastructure transitions and bootstrap masks are distinct;
- GAE fragments no longer discard one real transition at every boundary;
- infrastructure faults are excluded from policy/value losses;
- the adaptive KL coefficient contributes to the standard PPO loss;
- predictor overflow is deferred instead of dropped or rejected.

These compatibility patches will be migrated in a dedicated follow-up commit together with the
legacy trainer, after the public runtime contracts in this milestone are accepted.

## Migration sequence

1. Adapt v1 trajectories to `TransitionBatch` and enforce unique `(actor, env, episode, step)` IDs.
2. Replace `multiprocessing.Queue` payload copies with `SharedMemoryArena` descriptors.
3. Move predictor requests to `DeadlineBatcher` and publish weights through `PolicyRegistry`.
4. Route rollouts through `OnPolicyExperienceQueue` with policy-lag and age limits.
5. Run one learner per rank through the PyTorch DDP `DistributedContext`.
6. Replace the in-memory coordinator with a networked control-plane service without changing
   the runtime-facing interfaces.
7. Add a C++ vector-environment and ring-buffer extension only after profiling demonstrates that
   Python orchestration remains a material bottleneck.
