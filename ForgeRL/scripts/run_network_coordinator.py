from __future__ import annotations

import argparse
import json
import os

from forge_rl.runtime import NetworkCoordinatorService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ForgeRL M2 network coordinator.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument(
        "--token-env",
        default="FORGERL_COORDINATOR_TOKEN",
        help="environment variable containing the optional bearer token",
    )
    args = parser.parse_args()
    if args.port < 0 or args.port > 65535:
        parser.error("port must be in [0, 65535]")
    if args.lease_seconds <= 0:
        parser.error("lease-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env) or None
    service = NetworkCoordinatorService(
        host=args.host,
        port=args.port,
        lease_seconds=args.lease_seconds,
        bearer_token=token,
    )
    print(
        json.dumps(
            {
                "endpoint": service.endpoint,
                "lease_seconds": args.lease_seconds,
                "bearer_auth": token is not None,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.close()


if __name__ == "__main__":
    main()
