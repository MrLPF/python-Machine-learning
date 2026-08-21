#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from typing import Any, Iterable

from forge_rl.benchmarks import M1AcceptanceConfig, run_m1_acceptance


@dataclass(frozen=True, slots=True)
class TuningCase:
    name: str
    actors: int
    items_per_request: int
    max_batch_items: int
    min_batch_items: int
    max_wait_ms: float


SMOKE_CASES = (
    TuningCase("a2_i2_b8_m4_w0.5", 2, 2, 8, 4, 0.5),
    TuningCase("a4_i4_b32_m16_w0.5", 4, 4, 32, 16, 0.5),
)

CONTROLLED_CASES = (
    TuningCase("a4_i4_b64_m16_w0.25", 4, 4, 64, 16, 0.25),
    TuningCase("a4_i8_b64_m32_w0.5", 4, 8, 64, 32, 0.5),
    TuningCase("a8_i4_b128_m32_w0.25", 8, 4, 128, 32, 0.25),
    TuningCase("a8_i8_b128_m32_w0.5", 8, 8, 128, 32, 0.5),
    TuningCase("a8_i8_b128_m64_w1.0", 8, 8, 128, 64, 1.0),
    TuningCase("a8_i16_b128_m64_w0.5", 8, 16, 128, 64, 0.5),
    TuningCase("a16_i8_b256_m64_w0.5", 16, 8, 256, 64, 0.5),
    TuningCase("a16_i16_b256_m128_w1.0", 16, 16, 256, 128, 1.0),
)


def _coefficient_of_variation(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if len(rows) < 2:
        return 0.0
    mean = statistics.fmean(rows)
    return 0.0 if mean == 0 else statistics.pstdev(rows) / abs(mean)


def _trial(case: TuningCase, args: argparse.Namespace) -> dict[str, Any]:
    config = M1AcceptanceConfig(
        actors=case.actors,
        requests_per_actor=args.requests_per_actor,
        items_per_request=case.items_per_request,
        width=args.width,
        output_size=args.output_size,
        max_batch_items=case.max_batch_items,
        v2_min_batch_items=case.min_batch_items,
        legacy_min_batch_items=1,
        max_wait_ms=case.max_wait_ms,
        request_slots=args.request_slots,
        model_seed=args.model_seed,
        throughput_gate=0.0,
        checksum_tolerance=args.checksum_tolerance,
        device=args.device,
        amp_dtype=args.amp_dtype,
    )
    report = run_m1_acceptance(config)
    correctness = (
        report.legacy.audit.passed
        and report.v2.audit.passed
        and report.legacy.errors == 0
        and report.v2.errors == 0
        and report.checksum_relative_error <= config.checksum_tolerance
    )
    return {
        "correctness_passed": correctness,
        "throughput_speedup": report.throughput_speedup,
        "checksum_relative_error": report.checksum_relative_error,
        "legacy_valid_rows_per_second": report.legacy.valid_rows_per_second,
        "v2_valid_rows_per_second": report.v2.valid_rows_per_second,
        "legacy_latency_p95_ms": report.legacy.latency_p95_ms,
        "v2_latency_p95_ms": report.v2.latency_p95_ms,
        "legacy_batch_fill_ratio": report.legacy.batch_fill_ratio,
        "v2_batch_fill_ratio": report.v2.batch_fill_ratio,
        "legacy_audit": report.legacy.audit.to_dict(),
        "v2_audit": report.v2.audit.to_dict(),
    }


def _summarize(case: TuningCase, trials: list[dict[str, Any]]) -> dict[str, Any]:
    speedups = [float(row["throughput_speedup"]) for row in trials]
    v2_latency = [float(row["v2_latency_p95_ms"]) for row in trials]
    return {
        "case": asdict(case),
        "correctness_passed": all(bool(row["correctness_passed"]) for row in trials),
        "trials": trials,
        "median_speedup": statistics.median(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
        "speedup_coefficient_of_variation": _coefficient_of_variation(speedups),
        "median_v2_latency_p95_ms": statistics.median(v2_latency),
        "median_v2_batch_fill_ratio": statistics.median(
            float(row["v2_batch_fill_ratio"]) for row in trials
        ),
    }


def _decision(
    summaries: list[dict[str, Any]], *, controlled: bool, repeats: int
) -> tuple[str, list[str]]:
    if not summaries or any(not row["correctness_passed"] for row in summaries):
        return "FAIL_CORRECTNESS", ["at least one tuning case failed correctness"]
    target_cases = [
        row for row in summaries if int(row["case"]["actors"]) >= 8
    ]
    if not controlled:
        return "SCREENING_ONLY", [
            "hosted or unpinned matrix results are regression signals only"
        ]
    if repeats < 3:
        return "INSUFFICIENT_REPEATS", ["controlled decisions require at least three repeats"]
    if not target_cases:
        return "INSUFFICIENT_TARGET_CASES", [
            "controlled decisions require at least one case with eight or more actors"
        ]
    best_target = max(target_cases, key=lambda row: float(row["median_speedup"]))
    best_speedup = float(best_target["median_speedup"])
    best_cv = float(best_target["speedup_coefficient_of_variation"])
    if best_speedup >= 2.0 and best_cv <= 0.10:
        return "CANDIDATE_FOR_FORMAL_M1", [
            "a stable target-scale configuration reached the documented 2x threshold"
        ]
    if max(float(row["median_speedup"]) for row in target_cases) < 1.0:
        return "CPP_DESCRIPTOR_RING_RECOMMENDED", [
            "all controlled eight-or-more-actor cases remained slower than v1"
        ]
    return "CONTINUE_PYTHON_TUNING", [
        "no stable 2x candidate yet; evidence is insufficient to justify the C++ ring"
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the documented M1 node-local inference tuning matrix"
    )
    parser.add_argument("--preset", choices=("smoke", "controlled"), default="smoke")
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmup-repeats", type=int)
    parser.add_argument("--requests-per-actor", type=int)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--output-size", type=int, default=4)
    parser.add_argument("--request-slots", type=int, default=8)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--checksum-tolerance", type=float, default=1e-6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"))
    parser.add_argument("--require-candidate", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/m1-tuning-matrix.json"),
    )
    args = parser.parse_args()
    if args.repeats is None:
        args.repeats = 1 if args.preset == "smoke" else 3
    if args.warmup_repeats is None:
        args.warmup_repeats = 0 if args.preset == "smoke" else 1
    if args.requests_per_actor is None:
        args.requests_per_actor = 8 if args.preset == "smoke" else 1000
    if args.repeats <= 0 or args.warmup_repeats < 0 or args.requests_per_actor <= 0:
        parser.error("repeats and requests-per-actor must be positive; warmups cannot be negative")
    if args.controlled and args.preset != "controlled":
        parser.error("--controlled requires --preset controlled")
    if args.controlled and not os.environ.get("FORGERL_CPUSET"):
        parser.error("--controlled requires FORGERL_CPUSET on the pinned runner")
    if args.require_candidate and not args.controlled:
        parser.error("--require-candidate is valid only with --controlled")
    return args


def main() -> None:
    args = parse_args()
    cases = SMOKE_CASES if args.preset == "smoke" else CONTROLLED_CASES
    summaries: list[dict[str, Any]] = []
    for case in cases:
        for _ in range(args.warmup_repeats):
            warmup = _trial(case, args)
            if not warmup["correctness_passed"]:
                raise RuntimeError(f"warmup correctness failed for {case.name}")
        trials = [_trial(case, args) for _ in range(args.repeats)]
        summaries.append(_summarize(case, trials))
    status, reasons = _decision(
        summaries,
        controlled=bool(args.controlled),
        repeats=int(args.repeats),
    )
    best = max(summaries, key=lambda row: float(row["median_speedup"]))
    payload = {
        "schema_version": 1,
        "status": status,
        "formal_eligible": bool(args.controlled and args.repeats >= 3),
        "reasons": reasons,
        "preset": args.preset,
        "repeats": args.repeats,
        "warmup_repeats": args.warmup_repeats,
        "requests_per_actor": args.requests_per_actor,
        "v2_runtime": "event-driven-shared-ready-descriptor-queue",
        "device": args.device,
        "amp_dtype": args.amp_dtype,
        "best_case": best,
        "cases": summaries,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "github_sha": os.environ.get("GITHUB_SHA", ""),
            "cpuset": os.environ.get("FORGERL_CPUSET", ""),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "best_case": best["case"]["name"],
                "best_median_speedup": best["median_speedup"],
                "best_speedup_cv": best["speedup_coefficient_of_variation"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_candidate and status != "CANDIDATE_FOR_FORMAL_M1":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
