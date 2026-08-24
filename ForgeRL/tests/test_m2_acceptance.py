from __future__ import annotations

from pathlib import Path

import json
import pytest
import torch.multiprocessing as mp

from forge_rl.benchmarks import (
    M2RunReport,
    M2WorkerConfig,
    evaluate_m2_acceptance,
    run_m2_worker,
)
from forge_rl.distributed import DistributedContext


def _report(
    *,
    world_size: int,
    throughput: float,
    controlled: bool,
    node_ids: tuple[str, ...],
    invalid_rows_in_loss: int = 0,
) -> M2RunReport:
    configuration = {
        "measured_steps": 10,
        "valid_rows_per_rank": 32,
        "invalid_rows_per_batch": 1,
        "observation_width": 8,
        "hidden_size": 16,
        "learning_rate": 1e-3,
        "policy_publish_interval": 2,
    }
    return M2RunReport(
        schema_version=1,
        status="COMPLETED" if invalid_rows_in_loss == 0 else "FAIL_CORRECTNESS",
        controlled=controlled,
        world_size=world_size,
        learner_ranks=world_size,
        physical_node_ids=node_ids,
        unique_physical_nodes=len(set(node_ids)),
        elapsed_seconds=1.0,
        valid_rows=320 * world_size,
        valid_rows_per_second=throughput,
        invalid_rows_received=10 * world_size,
        invalid_rows_in_loss=invalid_rows_in_loss,
        runtime_errors=0,
        deadlock_free=True,
        audit={"passed": invalid_rows_in_loss == 0},
        experience_requests=10 * world_size,
        experience_rows=320 * world_size,
        policy_publications=5 * world_size,
        latest_policy_version=4,
        configuration=configuration,
        environment={"python": "test"},
    )


def test_m2_acceptance_requires_controlled_two_node_seventy_percent_efficiency() -> None:
    baseline = _report(
        world_size=1,
        throughput=100.0,
        controlled=True,
        node_ids=("node-a",),
    )
    target = _report(
        world_size=2,
        throughput=150.0,
        controlled=True,
        node_ids=("node-a", "node-b"),
    )
    accepted = evaluate_m2_acceptance(baseline, target)
    assert accepted.status == "GO"
    assert accepted.go
    assert accepted.weak_scaling_efficiency == pytest.approx(0.75)

    slow = evaluate_m2_acceptance(
        baseline,
        _report(
            world_size=2,
            throughput=120.0,
            controlled=True,
            node_ids=("node-a", "node-b"),
        ),
    )
    assert slow.status == "FAIL_WEAK_SCALING"

    screening = evaluate_m2_acceptance(
        _report(
            world_size=1,
            throughput=100.0,
            controlled=False,
            node_ids=("container-a",),
        ),
        _report(
            world_size=2,
            throughput=180.0,
            controlled=False,
            node_ids=("container-a", "container-b"),
        ),
    )
    assert screening.status == "SCREENING_ONLY"
    assert not screening.go


def _distributed_worker(
    rank: int,
    world_size: int,
    init_method: str,
    output_path: str,
) -> None:
    context = DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        backend="gloo",
    )
    context.initialize(init_method=init_method, timeout_seconds=60.0)
    try:
        report = run_m2_worker(
            context,
            M2WorkerConfig(
                warmup_steps=1,
                measured_steps=4,
                valid_rows_per_rank=8,
                invalid_rows_per_batch=1,
                observation_width=4,
                hidden_size=8,
                policy_publish_interval=2,
                advertise_host="127.0.0.1",
                physical_node_id=f"screening-node-{rank}",
                controlled=False,
            ),
        )
        if report is not None:
            Path(output_path).write_text(
                json.dumps(report.to_dict(), sort_keys=True),
                encoding="utf-8",
            )
    finally:
        context.close()


@pytest.mark.distributed
def test_two_rank_m2_worker_has_no_invalid_loss_rows_or_deadlock(tmp_path: Path) -> None:
    output = tmp_path / "m2-target.json"
    init_method = f"file://{tmp_path / 'rendezvous'}"
    mp.spawn(
        _distributed_worker,
        args=(2, init_method, str(output)),
        nprocs=2,
        join=True,
    )
    report = M2RunReport.load(output)
    assert report.status == "COMPLETED"
    assert report.deadlock_free
    assert report.world_size == 2
    assert report.learner_ranks == 2
    assert report.audit["passed"]
    assert report.invalid_rows_received == 10
    assert report.invalid_rows_in_loss == 0
    assert report.runtime_errors == 0
    assert report.experience_requests == 10
    assert report.experience_rows == 80
    assert report.policy_publications == 4
    assert report.latest_policy_version == 1
