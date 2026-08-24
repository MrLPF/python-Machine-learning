from __future__ import annotations

import pytest

from forge_rl.benchmarks import M1AcceptanceConfig, run_m1_acceptance


@pytest.mark.integration
def test_m1_acceptance_smoke_is_correct_but_waits_for_learning_gate() -> None:
    report = run_m1_acceptance(
        M1AcceptanceConfig(
            actors=2,
            requests_per_actor=4,
            items_per_request=2,
            width=8,
            output_size=3,
            max_batch_items=8,
            v2_min_batch_items=4,
            legacy_min_batch_items=1,
            request_slots=4,
            throughput_gate=0.0,
        )
    )
    assert report.legacy.audit.passed
    assert report.v2.audit.passed
    assert report.legacy.errors == 0
    assert report.v2.errors == 0
    assert report.checksum_relative_error <= report.config.checksum_tolerance
    assert report.status == "PENDING_LEARNING_GATE"
    assert not report.go
