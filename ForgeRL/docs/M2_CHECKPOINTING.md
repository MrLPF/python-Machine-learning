# M2 distributed checkpointing

## Scope

`DistributedCheckpointManager` saves and restores the learner model and optimizer with PyTorch
Distributed Checkpoint (DCP). Every learner rank participates in the operation. DCP writes local
shards in parallel and performs load-time resharding when the restore world size or PyTorch
parallelism differs from the saved topology.

The checkpoint directory must reside on storage visible at the same path to all participating
learner ranks.

## Saved state

Each checkpoint contains:

```text
model state with canonical parameter names
optimizer state with canonical parameter names
ForgeRL completion manifest
```

The JSON manifest records:

```text
schema version
checkpoint ID
global training step
policy version
saved world size
creation time
model class
optimizer classes
JSON-safe application metadata
```

Rank-local RNG streams are intentionally not persisted as topology-independent state. A restore
with a different rank count should derive rank-local seeds from the saved global step/base seed in
application metadata rather than replaying the old rank assignment.

## Atomic commit

The coordinator rank creates a unique incomplete directory. DCP writes all rank shards there. Only
after the distributed save succeeds does the coordinator write `forgerl_manifest.json` and rename
the directory to the final checkpoint ID.

A directory without the completion manifest is rejected during load. Existing completed checkpoint
IDs are never overwritten.

## API

```python
from forge_rl.distributed import DistributedCheckpointManager

manager = DistributedCheckpointManager("/shared/checkpoints")

manager.save(
    "step-1000",
    model=model,
    optimizers=optimizer,
    global_step=1000,
    policy_version=42,
    metadata={"base_seed": 17},
)

manifest = manager.load(
    "step-1000",
    model=new_model,
    optimizers=new_optimizer,
)
```

The model may be an unwrapped module, DDP module or another PyTorch parallel module supported by
the canonical distributed state-dict APIs.

## Topology validation

The distributed test suite covers both directions:

```text
2-rank DDP save -> 1-rank unwrapped restore
1-rank unwrapped save -> 2-rank DDP restore
```

Both model parameters and Adam optimizer state must match exactly after restore. The saved world
size remains audit metadata only; it does not constrain the restore topology.

## M2 boundary

This completes only the topology-independent distributed checkpoint item. Actor, inference replica
and learner-rank failure injection remain the next M2 task.
