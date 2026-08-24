# M2 formal multi-node acceptance

## Scope

M2 formal acceptance validates the implemented multi-node runtime without entering M3. The required
topology is:

```text
physical node 0: learner rank 0 + direct Experience/Policy receivers
physical node 1: learner rank 1 + direct Experience/Policy receivers
                         │
                         └── Gloo or NCCL DDP process group
```

Each rank sends its generated `TransitionBatch` directly to the peer rank's Experience service.
Each rank trains only on the peer batch and publishes complete policy snapshots to the peer Policy
service. The Coordinator is not placed in the tensor path.

## Weak-scaling workload

The one-rank baseline and two-rank target use identical per-rank settings:

```text
warmup steps
measured steps
valid rows per rank per step
invalid infrastructure rows per batch
observation width
model width
optimizer and learning rate
policy publication interval
```

Global target work doubles when the learner rank count doubles. Weak-scaling efficiency is:

```text
target valid rows/s
-----------------------------------------------
baseline valid rows/s * target world size
```

The M2 gate requires efficiency greater than or equal to `0.70`.

## Correctness gates

Every measured run records global transition identities and requires:

```text
missing valid transitions = 0
duplicate valid transitions = 0
unexpected trained transitions = 0
invalid infrastructure rows entering loss = 0
runtime errors = 0
no collective or data-channel deadlock
```

The benchmark intentionally places one or more `valid_mask=0` rows in every batch. These rows must
cross the network for diagnosis but must never be selected for the learner loss.

## Reports

`benchmark_m2_worker.py` writes one report for each topology:

```text
m2-baseline.json
m2-target.json
```

`benchmark_m2_acceptance.py` combines both reports into:

```text
m2-final.json
```

Possible final states are:

```text
GO
SCREENING_ONLY
FAIL_CORRECTNESS
FAIL_TOPOLOGY
FAIL_CONFIGURATION
FAIL_WEAK_SCALING
```

Only `GO` constitutes formal M2 acceptance.

## Formal execution

The manual GitHub workflow `ForgeRL M2 formal acceptance` runs on a controller labeled:

```text
self-hosted
linux
x64
forgerl-m2-controller
```

The controller must have passwordless SSH and `rsync` access to two distinct learner hosts. Both
hosts must already provide the pinned Python, PyTorch and NumPy environment. The workflow installs
ForgeRL with `--no-deps`, runs a one-rank baseline on node 0, then launches one learner rank on each
physical node with the same per-rank workload.

A 900-second command timeout turns a collective or network deadlock into an explicit workflow
failure. Hosted runners, two containers on one host, and uncontrolled local runs are screening only
and cannot produce `GO`.

## Manual commands

A single-rank baseline may be launched with:

```bash
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
python scripts/benchmark_m2_worker.py \
  --controlled \
  --advertise-host NODE0 \
  --physical-node-id NODE0 \
  --output /shared/m2-baseline.json
```

The two target ranks use the same arguments plus:

```text
WORLD_SIZE=2
RANK=0 or 1
MASTER_ADDR=NODE0
MASTER_PORT=<free port>
```

After both ranks finish:

```bash
python scripts/benchmark_m2_acceptance.py \
  --baseline /shared/m2-baseline.json \
  --target /shared/m2-target.json \
  --efficiency-gate 0.70 \
  --require-go \
  --output /shared/m2-final.json
```

## Milestone boundary

M3 does not begin until the user accepts the M2 result. M1 remains `formal validation pending`
until the user's local M1 report is produced.
