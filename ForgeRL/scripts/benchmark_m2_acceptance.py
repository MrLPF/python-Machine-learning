#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from forge_rl.benchmarks import M2RunReport, evaluate_m2_acceptance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ForgeRL M2 two-node weak-scaling acceptance."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--efficiency-gate", type=float, default=0.70)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-go", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_m2_acceptance(
        M2RunReport.load(args.baseline),
        M2RunReport.load(args.target),
        efficiency_gate=args.efficiency_gate,
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.require_go and not report.go:
        sys.exit(2)
    if report.status.startswith("FAIL_"):
        sys.exit(1)


if __name__ == "__main__":
    main()
