# ForgeRL v2 architecture

## Scope boundary

ForgeRL is an RL runtime, not a new tensor framework. PyTorch remains responsible for tensor
kernels, autograd, CUDA allocation, collectives, AMP, DDP and FSDP2. C++/CUDA is reserved for
measured hot paths: vectorized simulation, shared-memory slot management, preprocessing and
custom fused operators.

## Planes

```text
Control plane
  Coordinator -> membership, leases, placement, policy version, checkpoint metadata

Data plane
  EnvRunner -> InferenceService -> EnvRunner
  EnvRunner -> ExperienceService -> LearnerGroup
  LearnerGroup -> PolicyRegistry -> InferenceService

Compute plane
  C++/Python vector environments | PyTorch inference replicas | DDP/FSDP2 learners
```

Large tensors never travel through the coordinator. Local processes exchange slot descriptors
backed by shared memory. Cross-node transports implement the same descriptor/message contracts;
gRPC is intended for control messages, while tensor payloads can use torch.distributed, UCX or a
future dedicated transport according to topology.

## Components

### Coordinator

Tracks node role, endpoint, resources, heartbeat deadline and the current committed policy
version. The milestone-1 implementation is in-memory; the interface is deliberately small so a
Raft/etcd-backed implementation can replace it later.

### EnvRunner

Owns vectorized environment instances and episode identity. It emits transitions with explicit
`terminated`, `truncated`, `valid_mask`, `episode_id` and `step_id`. A simulator crash is a
runtime fault, not an artificial terminal transition.

### InferenceService

Uses node-local replicas by default. Requests are batched by item count, policy version and a
latency deadline. Oversized requests are deferred instead of rejected. Active/staging policy
buffers permit atomic version switches.

### ExperienceService

The first table is a bounded on-policy queue. It enforces row-based capacity, backpressure,
policy-lag limits, age limits and stale-data accounting. A later replay table adds sampler,
remover, priority updater and rate limiter interfaces for SAC/R2D2/offline RL.

### LearnerGroup

Uses one process per GPU and PyTorch DDP initially. FSDP2 is enabled only when the policy or
optimizer state no longer fits a device. Rank 0 publishes a committed policy version after an
optimizer/checkpoint boundary.

### PolicyRegistry

Maintains monotonic policy versions and a bounded snapshot history. Inference replicas can ask
for latest or an exact version, which is required by strict-round collection and lag diagnosis.

## OneFlow ideas retained

- distribution/layout is explicit rather than hidden in ad-hoc calls;
- communication is a planned conversion between states, analogous to Boxing;
- runtime dependencies and bounded resources are first-class;
- computation, communication and control are separated.

ForgeRL does not copy OneFlow's tensor runtime. DTensor/DeviceMesh and DDP/FSDP2 provide the
compute-layer equivalent while ForgeRL concentrates on RL dataflow.
