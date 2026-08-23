# M2 failure injection and component restart

## Scope

This M2 step verifies recovery from three failures without admitting invalid transitions or
silently reusing stale component identity:

```text
Actor process crash
Inference replica process crash
Learner rank crash
```

The implementation uses `RestartableProcess` for Actor and inference component supervision.
Learner recovery restarts the complete DDP learner group from the latest completed distributed
checkpoint; in-place elastic rank replacement is not claimed by this milestone.

## Actor restart

An Actor replacement registers with the same logical `node_id` and receives a strictly newer
Coordinator generation. Heartbeats from the old generation are rejected. The replacement resumes
with a new transition identity and sends directly to `NetworkExperienceService`.

The injection test verifies:

```text
old process is terminated
replacement generation increments
stale heartbeat returns stale_generation
both valid rows enter the Experience queue exactly once
no duplicate transition identity is produced
```

Infrastructure failure itself never becomes an environment terminal transition.

## Inference replica restart

An inference replica owns a local `PolicyRegistry` and `NetworkPolicyService`. After the process is
terminated, a replacement registers a new generation and endpoint. The Learner republishes the
latest complete policy snapshot to that endpoint before the replica is considered recovered.

The injection test verifies:

```text
replacement generation increments
endpoint changes
latest policy version is restored
restored tensor checksum matches the Learner snapshot
Coordinator discovery returns only the replacement
```

A partial state dict is never activated; `NetworkPolicyService` validates and publishes the full
snapshot atomically through `PolicyRegistry`.

## Learner rank restart

A two-rank Gloo learner group performs an optimizer update and commits a distributed checkpoint.
One rank is then terminated with a non-zero exit code. The failed group is discarded and a new
two-rank group restores the last completed checkpoint before continuing training.

The injection test verifies:

```text
rank failure is observed by the process supervisor
completed checkpoint remains readable
new group restores global_step and policy_version
model and Adam state restore successfully
post-restart DDP parameters remain identical across ranks
no collective deadlock occurs
```

This recovery model deliberately restarts the entire learner group because ordinary DDP process
groups cannot safely continue after a member disappears. Topology-independent restore remains
provided by `DistributedCheckpointManager`.

## M2 boundary

This completes the roadmap failure-injection item. The remaining M2 work is execution of the
formal two-node validation and weak-scaling gate. M3 algorithm and replay work is not part of this
step.
