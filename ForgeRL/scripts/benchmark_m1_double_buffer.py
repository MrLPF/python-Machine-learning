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
class BufferCase:
    name: str
    splits_per_actor: int


def _coefficient_of_variation(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if len(rows) < 2:
        return 0.0
    mean = statistics.fmean(rows)
    return 0.0 if mean == 0 else statistics.pstdev(rows) / abs(mean)


def _trial(case: BufferCase, args: argparse.Namespace) -> dict[str, Any]:
    if args.requests_per_actor % case.splits_per_actor:
        raise ValueError(
            "requests-per-actor must be divisible by every split count so total work stays fixed"
        )
    logical_clients = args.actors * case.splits_per_actor
    requests_per_client = args.requests_per_actor // case.splits_per_actor
    config = M1AcceptanceConfig(
        actors=logical_clients,
        requests_per_actor=requests_per_client,
        items_per_request=args.items_per_request,
        width=args.width,
        output_size=args.output_size,
        max_batch_items=args.max_batch_items,
        v2_min_batch_items=args.min_batch_items,
        legacy_min_batch_items=1,
        max_wait_ms=args.max_wait_ms,
        request_slots=1,
        model_seed=args.model_seed,
        throughput_gate=0.0,
        checksum_tolerance=args.checksum_tolerance,
        device="cpu",
        amp_dtype=None,
    )
    report = run_m1_acceptance(
        config,
        runtime_mode="in-process-batch",
    )
    correctness = (
        report.legacy.audit.passed
        and report.v2.audit.passed
        and report.legacy.errors == 0
        and report.v2.errors == 0
        and report.checksum_relative_error <= config.checksum_tolerance
    )
    return {
        "correctness_passed": correctness,
        "physical_actors": args.actors,
        "splits_per_actor": case.splits_per_actor,
        "logical_inference_clients": logical_clients,
        "requests_per_logical_client": requests_per_client,
        "total_requests": logical_clients * requests_per_client,
        "total_valid_rows": (
            logical_clients * requests_per_client * args.items_per_request
        ),
        "throughput_speedup": report.throughput_speedup,
        "checksum_relative_error": report.checksum_relative_error,
        "legacy_valid_rows_per_second": report.legacy.valid_rows_per_second,
        "v2_valid_rows_per_second": report.v2.valid_rows_per_second,
        "legacy_latency_p95_ms": report.legacy.latency_p95_ms,
        "v2_latency_p95_ms": report.v2.latency_p95_ms,
        "legacy_mean_batch_items": report.legacy.mean_batch_items,
        "v2_mean_batch_items": report.v2.mean_batch_items,
        "legacy_batch_fill_ratio": report.legacy.batch_fill_ratio,
        "v2_batch_fill_ratio": report.v2.batch_fill_ratio,
        "legacy_audit": report.legacy.audit.to_dict(),
        "v2_audit": report.v2.audit.to_dict(),
    }


def _summary(case: BufferCase, trials: list[dict[str, Any]]) -> dict[str, Any]:
    speedups = [float(row["throughput_speedup"]) for row in trials]
    return {
        "case": asdict(case),
        "correctness_passed": all(bool(row["correctness_passed"]) for row in trials),
        "trials": trials,
        "median_speedup": statistics.median(speedups),
        "speedup_coefficient_of_variation": _coefficient_of_variation(speedups),
        "median_legacy_rows_per_second": statistics.median(
            float(row["legacy_valid_rows_per_second"]) for row in trials
        ),
        "median_v2_rows_per_second": statistics.median(
            float(row["v2_valid_rows_per_second"]) for row in trials
        ),
        "median_v2_latency_p95_ms": statistics.median(
            float(row["v2_latency_p95_ms"]) for row in trials
        ),
        "median_v2_mean_batch_items": statistics.median(
            float(row["v2_mean_batch_items"]) for row in trials
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one and two inference lanes per physical Actor while keeping total valid "
            "rows identical for the frozen v1 and ForgeRL v2 subjects."
        )
    )
    parser.add_argument("--actors", type=int, default=8)
    parser.add_argument("--requests-per-actor", type=int, default=1000)
    parser.add_argument("--items-per-request", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--output-size", type=int, default=4)
    parser.add_argument("--max-batch-items", type=int, default=128)
    parser.add_argument("--min-batch-items", type=int, default=64)
    parser.add_argument("--max-wait-ms", type=float, default=0.5)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--checksum-tolerance", type=float, default=1e-6)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-repeats", type=int, default=0)
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/m1-double-buffer.json"),
    )
    args = parser.parse_args()
    positive = (
        args.actors,
        args.requests_per_actor,
        args.items_per_request,
        args.width,
        args.output_size,
        args.max_batch_items,
        args.min_batch_items,
        args.repeats,
    )
    if any(value <= 0 for value in positive):
        parser.error("all sizes and repeats must be positive")
    if args.warmup_repeats < 0 or args.max_wait_ms < 0:
        parser.error("warmup repeats and max wait cannot be negative")
    if args.requests_per_actor % 2:
        parser.error("requests-per-actor must be divisible by two")
    if args.min_batch_items > args.max_batch_items:
        parser.error("min-batch-items cannot exceed max-batch-items")
    if args.controlled and not os.environ.get("FORGERL_CPUSET"):
        parser.error("--controlled requires FORGERL_CPUSET on the pinned runner")
    return args


def main() -> None:
    args = parse_args()
    cases = (
        BufferCase("single-buffer", 1),
        BufferCase("double-buffer", 2),
    )
    rows: dict[str, list[dict[str, Any]]] = {case.name: [] for case in cases}

    for case in cases:
        for _ in range(args.warmup_repeats):
            result = _trial(case, args)
            if not result["correctness_passed"]:
                raise RuntimeError(f"warmup correctness failed for {case.name}")

    # Alternate execution order to reduce systematic thermal/frequency ordering bias.
    for repeat in range(args.repeats):
        order = cases if repeat % 2 == 0 else tuple(reversed(cases))
        for case in order:
            rows[case.name].append(_trial(case, args))

    summaries = {
        case.name: _summary(case, rows[case.name])
        for case in cases
    }
    single = summaries["single-buffer"]
    double = summaries["double-buffer"]
    correctness = bool(single["correctness_passed"] and double["correctness_passed"])
    v2_gain = float(double["median_v2_rows_per_second"]) / max(
        float(single["median_v2_rows_per_second"]), 1e-12
    )
    v1_gain = float(double["median_legacy_rows_per_second"]) / max(
        float(single["median_legacy_rows_per_second"]), 1e-12
    )
    stable_two_x = (
        float(double["median_speedup"]) >= 2.0
        and float(double["speedup_coefficient_of_variation"]) <= 0.10
    )

    if not correctness:
        status = "FAIL_CORRECTNESS"
        reasons = ["single- or double-buffer trial failed correctness"]
    elif not args.controlled:
        status = "SCREENING_ONLY"
        reasons = ["hosted or unpinned double-buffer evidence is screening-only"]
    elif args.repeats < 3:
        status = "INSUFFICIENT_REPEATS"
        reasons = ["controlled evidence requires at least three repeats"]
    elif stable_two_x and v2_gain > 1.0:
        status = "DOUBLE_BUFFER_CANDIDATE"
        reasons = [
            "two lanes per Actor produced a stable >=2x v2/v1 result and improved v2 throughput"
        ]
    else:
        status = "CONTINUE_M1_TUNING"
        reasons = [
            "double buffering did not produce a stable >=2x target-scale candidate"
        ]

    payload = {
        "schema_version": 1,
        "status": status,
        "formal_eligible": bool(args.controlled and args.repeats >= 3),
        "reasons": reasons,
        "configuration": {
            "physical_actors": args.actors,
            "requests_per_physical_actor": args.requests_per_actor,
            "items_per_request": args.items_per_request,
            "width": args.width,
            "output_size": args.output_size,
            "max_batch_items": args.max_batch_items,
            "min_batch_items": args.min_batch_items,
            "max_wait_ms": args.max_wait_ms,
            "repeats": args.repeats,
            "warmup_repeats": args.warmup_repeats,
        },
        "work_conservation": {
            "single_total_valid_rows": single["trials"][0]["total_valid_rows"],
            "double_total_valid_rows": double["trials"][0]["total_valid_rows"],
            "identical": (
                single["trials"][0]["total_valid_rows"]
                == double["trials"][0]["total_valid_rows"]
            ),
        },
        "single_buffer": single,
        "double_buffer": double,
        "double_buffer_v2_throughput_gain": v2_gain,
        "double_buffer_v1_throughput_gain": v1_gain,
        "double_buffer_stable_two_x": stable_two_x,
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
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "single_speedup": single["median_speedup"],
                "double_speedup": double["median_speedup"],
                "double_v2_gain": v2_gain,
                "double_mean_batch": double["median_v2_mean_batch_items"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
