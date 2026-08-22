from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from forge_rl.benchmarks import LearningGate, M1AcceptanceConfig, run_m1_acceptance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the frozen ForgeRL v1 queue data plane with a ForgeRL v2 runtime."
    )
    parser.add_argument("--actors", type=int, default=4)
    parser.add_argument("--requests-per-actor", type=int, default=250)
    parser.add_argument("--items-per-request", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--output-size", type=int, default=4)
    parser.add_argument("--max-batch-items", type=int, default=128)
    parser.add_argument("--v2-min-batch-items", type=int, default=32)
    parser.add_argument("--legacy-min-batch-items", type=int, default=1)
    parser.add_argument("--max-wait-ms", type=float, default=2.0)
    parser.add_argument("--request-slots", type=int, default=8)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--v2-runtime",
        choices=("auto", "actor-local", "in-process-batch", "process-mailbox"),
        default="auto",
        help=(
            "auto selects zero-serialization in-process batching for CPU and the "
            "process mailbox for CUDA. Explicit modes are retained for controlled A/B."
        ),
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        help="Optional CUDA autocast dtype; omit for full precision.",
    )
    parser.add_argument("--checksum-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--throughput-gate",
        type=float,
        default=2.0,
        help="Required v2/v1 valid-row throughput ratio. Use 0 for CI screening only.",
    )
    parser.add_argument(
        "--learning-report",
        type=Path,
        help="Optional five-seed CartPole/Pendulum JSON report required for an overall GO.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-go",
        action="store_true",
        help="Exit non-zero unless every synthetic and learning gate passes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = M1AcceptanceConfig(
        actors=args.actors,
        requests_per_actor=args.requests_per_actor,
        items_per_request=args.items_per_request,
        width=args.width,
        output_size=args.output_size,
        max_batch_items=args.max_batch_items,
        v2_min_batch_items=args.v2_min_batch_items,
        legacy_min_batch_items=args.legacy_min_batch_items,
        max_wait_ms=args.max_wait_ms,
        request_slots=args.request_slots,
        model_seed=args.model_seed,
        throughput_gate=args.throughput_gate,
        checksum_tolerance=args.checksum_tolerance,
        device=args.device,
        amp_dtype=args.amp_dtype,
    )
    learning_gate = LearningGate.load(args.learning_report) if args.learning_report else None
    report = run_m1_acceptance(
        config,
        learning_gate=learning_gate,
        runtime_mode=args.v2_runtime,
    )
    payload = report.to_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.require_go and not report.go:
        sys.exit(2)
    if report.status.startswith("FAIL_"):
        sys.exit(1)


if __name__ == "__main__":
    main()
