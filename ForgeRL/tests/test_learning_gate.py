from __future__ import annotations

import json
from pathlib import Path

from forge_rl.benchmarks import LearningGate


def _payload(*, pendulum_time: float = 101.0, seeds: int = 5) -> dict:
    return {
        "environments": [
            {
                "environment": "CartPole-v1",
                "seeds": seeds,
                "v1_time_to_target_median_seconds": 100.0,
                "v2_time_to_target_median_seconds": 104.0,
                "v1_normalized_auc": 0.8,
                "v2_normalized_auc": 0.78,
            },
            {
                "environment": "Pendulum-v1",
                "seeds": seeds,
                "v1_time_to_target_median_seconds": 100.0,
                "v2_time_to_target_median_seconds": pendulum_time,
                "v1_normalized_auc": 0.7,
                "v2_normalized_auc": 0.69,
            },
        ]
    }


def _load(tmp_path: Path, payload: dict) -> LearningGate:
    path = tmp_path / "learning-report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return LearningGate.load(path)


def test_learning_gate_requires_five_seeds_and_bounded_regression(tmp_path: Path) -> None:
    assert _load(tmp_path, _payload()).passed
    assert not _load(tmp_path, _payload(seeds=4)).passed
    assert not _load(tmp_path, _payload(pendulum_time=106.0)).passed
