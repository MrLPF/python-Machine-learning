from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_m1_tuning_matrix.py"
    spec = importlib.util.spec_from_file_location("forge_rl_m1_tuning_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _summary(*, actors: int, speedup: float, cv: float = 0.02) -> dict:
    return {
        "case": {"actors": actors, "name": f"a{actors}"},
        "correctness_passed": True,
        "median_speedup": speedup,
        "speedup_coefficient_of_variation": cv,
    }


def test_tuning_matrix_never_treats_hosted_smoke_as_formal() -> None:
    module = _load_module()
    status, _ = module._decision(
        [_summary(actors=8, speedup=2.5)], controlled=False, repeats=3
    )
    assert status == "SCREENING_ONLY"


def test_tuning_matrix_recommends_cpp_only_after_controlled_large_actor_failure() -> None:
    module = _load_module()
    status, _ = module._decision(
        [
            _summary(actors=4, speedup=1.2),
            _summary(actors=8, speedup=0.8),
            _summary(actors=16, speedup=0.9),
        ],
        controlled=True,
        repeats=3,
    )
    assert status == "CPP_DESCRIPTOR_RING_RECOMMENDED"


def test_tuning_matrix_requires_stable_two_x_candidate() -> None:
    module = _load_module()
    status, _ = module._decision(
        [_summary(actors=8, speedup=2.1, cv=0.05)], controlled=True, repeats=3
    )
    assert status == "CANDIDATE_FOR_FORMAL_M1"
    unstable, _ = module._decision(
        [_summary(actors=8, speedup=2.1, cv=0.2)], controlled=True, repeats=3
    )
    assert unstable == "CONTINUE_PYTHON_TUNING"
