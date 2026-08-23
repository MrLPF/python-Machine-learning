# M2 network coordinator control plane

## Scope

The network coordinator carries control metadata only. It never forwards observations,
trajectories, gradients, checkpoints or model tensors.

The implementation is dependency-free HTTP/JSON over the thread-safe `InMemoryCoordinator` state.
TLS termination and durable replicated state are deployment concerns outside this first M2 step.

## Endpoints

```text
GET  /healthz
GET  /v1/state
GET  /v1/nodes?role=<role>
POST /v1/nodes/register
POST /v1/nodes/<node_id>/heartbeat
POST /v1/policy/commit
```

Registration accepts:

```json
{
  "node_id": "learner-0",
  "role": "learner",
  "endpoint": "tcp://10.0.0.10:9100",
  "resources": {"cpu": 16, "gpu": 1},
  "metadata": {"hostname": "node-a"}
}
```

The server returns a monotonically increasing generation for each logical `node_id`. Generation
history survives lease expiry, so a delayed process cannot regain membership after a replacement
process registers with the same ID.

Heartbeats must include the exact current generation. Stale generations return HTTP `409` and are
never interpreted as valid membership.

Policy publication is also strictly monotonic. Duplicate or decreasing versions return HTTP `409`.

## Authentication

Set `FORGERL_COORDINATOR_TOKEN` before starting the service to require:

```text
Authorization: Bearer <token>
```

The built-in bearer check is suitable for isolated benchmark networks. Production deployments
should additionally use TLS or a trusted service mesh.

## Launcher

```bash
FORGERL_COORDINATOR_TOKEN=secret \
python scripts/run_network_coordinator.py \
  --host 0.0.0.0 \
  --port 8080 \
  --lease-seconds 10
```

## M2 boundary

This completes only the networked coordinator item. Direct experience/policy data channels,
distributed checkpointing, fault injection and two-node weak-scaling validation remain separate
M2 work items.
