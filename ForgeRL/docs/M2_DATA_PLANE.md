# M2 direct experience and policy data channels

## Scope

This M2 step implements the two direct point-to-point channels required by the roadmap:

```text
EnvRunner -> Experience Service
Learner   -> Policy Service
```

The network coordinator remains a control-plane directory. It advertises service endpoints and
policy versions, but it does not relay trajectories or model tensors.

## Wire format

Both channels use a persistent TCP connection and a versioned binary frame:

```text
magic | protocol version | JSON metadata length | binary payload length
JSON metadata
binary payload
```

The initial protocol version is `FRL2/1`. Frame sizes are bounded before allocation. Every binary
payload includes a SHA-256 digest in the metadata.

No Python pickle is accepted on either channel:

- transitions are encoded as contiguous NumPy byte ranges with explicit dtype, shape and path;
- policy state dictionaries are encoded as contiguous raw Tensor byte ranges with explicit
  PyTorch dtype, shape and parameter name.

The receiver reconstructs owned CPU arrays/tensors and then runs the existing ForgeRL validation
contracts.

## EnvRunner to Experience

`NetworkExperienceClient.put()` sends one canonical `TransitionBatch` directly to a
`NetworkExperienceService`.

The receiver performs, in order:

1. bearer-token validation when configured;
2. frame and SHA-256 validation;
3. safe array reconstruction without executable deserialization;
4. `TransitionBatch` validation, including identity and terminal/truncation rules;
5. valid-row counting and policy-version range extraction;
6. bounded insertion into `OnPolicyExperienceQueue`.

If the queue has no capacity, the sender receives a retryable `backpressure` error. Infrastructure
or protocol errors never become artificial terminal transitions.

Each request carries a caller-supplied or generated `request_id`. Accepted acknowledgements are
cached, so retrying the same ID after an uncertain connection result does not enqueue the same
trajectory twice.

## Learner to Policy

`NetworkPolicyClient.publish()` sends a complete CPU state dictionary and an explicit monotonically
increasing version to `NetworkPolicyService`.

The receiver validates the payload and publishes it into `PolicyRegistry`. Duplicate or decreasing
versions return `policy_version_conflict`. Retrying an already accepted request with the same
`request_id` returns the original acknowledgement without publishing again.

An inference process colocated with the policy service can use `PolicyRegistry.wait_for_newer()` and
refresh its active/staging replica only after a complete snapshot has been accepted. The learner
should commit the corresponding control-plane policy version only after the direct policy-channel
acknowledgement succeeds.

## Coordinator registration

Services register their direct endpoints through the existing network coordinator:

```text
role=experience endpoint=tcp://host:port
role=inference  endpoint=tcp://host:port metadata.protocol=forge-rl-policy-v1
```

EnvRunners and Learners discover the appropriate endpoint through the coordinator, then connect
directly. Large payloads never pass through HTTP/JSON coordinator routes.

## Security boundary

The built-in bearer token is intended for isolated benchmark or trusted cluster networks. A
production deployment should additionally use TLS termination, a service mesh or an encrypted
private network. The binary codecs do not execute serialized Python code.

## Current boundary

This completes the direct experience and policy channel roadmap item. Distributed checkpointing,
failure injection and the two-node weak-scaling gate remain separate M2 work items.
