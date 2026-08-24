from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import time

import numpy as np
import torch
from torch import nn

from forge_rl.runtime import (
    DoubleBufferedPolicyReplica,
    FastMailboxInferenceClient,
    FastMailboxNodeLocalInferenceService,
    MailboxInferenceClient,
    MailboxInferenceEndpoint,
    MailboxNodeLocalInferenceService,
    NodeLocalInferenceService,
    PolicyRegistry,
    PollingNodeLocalInferenceService,
    SharedInferenceClient,
    SharedInferenceEndpoint,
    ThreadedNodeLocalInferenceService,
)


class BenchmarkPolicy(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 4),
        )
        self.value = nn.Linear(width, 1)


def run_policy(module: BenchmarkPolicy, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "action": module.network(batch["obs"]),
        "value": module.value(batch["obs"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actors", type=int, default=4)
    parser.add_argument("--requests-per-actor", type=int, default=250)
    parser.add_argument("--items-per-request", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--max-batch-items", type=int, default=128)
    parser.add_argument("--min-batch-items", type=int, default=32)
    parser.add_argument("--max-wait-ms", type=float, default=2.0)
    parser.add_argument(
        "--service-mode",
        choices=("mailbox-fast", "mailbox", "event", "polling", "threaded"),
        default="mailbox-fast",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mailbox = args.service_mode in {"mailbox-fast", "mailbox"}
    if mailbox:
        endpoints = [
            MailboxInferenceEndpoint.create(
                actor_id=actor_id,
                max_items=args.items_per_request,
                request_fields={"obs": ((args.width,), np.float32)},
                response_fields={
                    "action": ((4,), np.float32),
                    "value": ((1,), np.float32),
                },
            )
            for actor_id in range(args.actors)
        ]
    else:
        endpoints = [
            SharedInferenceEndpoint.create(
                actor_id=actor_id,
                slot_count=8,
                max_items=args.items_per_request,
                request_fields={"obs": ((args.width,), np.float32)},
                response_fields={
                    "action": ((4,), np.float32),
                    "value": ((1,), np.float32),
                },
            )
            for actor_id in range(args.actors)
        ]

    factory = lambda: BenchmarkPolicy(args.width)
    source = factory()
    registry = PolicyRegistry(history_size=2)
    snapshot = registry.publish(source.state_dict(), version=0)
    service_class = {
        "mailbox-fast": FastMailboxNodeLocalInferenceService,
        "mailbox": MailboxNodeLocalInferenceService,
        "event": NodeLocalInferenceService,
        "polling": PollingNodeLocalInferenceService,
        "threaded": ThreadedNodeLocalInferenceService,
    }[args.service_mode]
    service = service_class(
        endpoints=endpoints,
        replica=DoubleBufferedPolicyReplica(factory, device="cpu"),
        infer_fn=run_policy,
        max_batch_items=args.max_batch_items,
        min_batch_items=args.min_batch_items,
        max_wait_ms=args.max_wait_ms,
    )
    service.refresh_policy(snapshot)
    service.start()
    if args.service_mode == "mailbox-fast":
        clients = [
            FastMailboxInferenceClient(endpoint, copy_outputs=False)
            for endpoint in endpoints
        ]
    elif args.service_mode == "mailbox":
        clients = [
            MailboxInferenceClient(endpoint, copy_outputs=False)
            for endpoint in endpoints
        ]
    else:
        clients = [SharedInferenceClient(endpoint) for endpoint in endpoints]
    inputs = [
        np.random.default_rng(actor_id).standard_normal(
            (args.items_per_request, args.width), dtype=np.float32
        )
        for actor_id in range(args.actors)
    ]

    def actor_loop(actor_id: int) -> None:
        for _ in range(args.requests_per_actor):
            clients[actor_id].infer(
                {"obs": inputs[actor_id]},
                min_policy_version=0,
                timeout=10.0,
            )

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=args.actors) as executor:
            list(executor.map(actor_loop, range(args.actors)))
        elapsed = time.perf_counter() - started
    finally:
        service.stop()
    metrics = service.metrics()
    total_items = args.actors * args.requests_per_actor * args.items_per_request
    report = {
        "service_mode": args.service_mode,
        "actors": args.actors,
        "requests": args.actors * args.requests_per_actor,
        "items": total_items,
        "elapsed_seconds": elapsed,
        "items_per_second": total_items / elapsed,
        "service": asdict(metrics),
    }
    if hasattr(service, "optimization_metrics"):
        report["optimization"] = asdict(service.optimization_metrics())
    if hasattr(service, "event_metrics"):
        report["event_driven"] = asdict(service.event_metrics())
    if hasattr(service, "transport_metrics"):
        report["mailbox"] = asdict(service.transport_metrics())
    print(json.dumps(report, indent=2, sort_keys=True))
    for client in clients:
        close = getattr(client, "close", None)
        if close is not None:
            close()
    for endpoint in endpoints:
        endpoint.shutdown()
        endpoint.close()
        endpoint.unlink()


if __name__ == "__main__":
    main()
