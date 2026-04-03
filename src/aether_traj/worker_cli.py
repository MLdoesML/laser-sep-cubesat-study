from __future__ import annotations

import argparse
import json
from pathlib import Path

from aether_traj.campaign_manager import run_next_job


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run managed Aether campaign jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-job", help="Run the next pending job or a specific managed job.")
    run_parser.add_argument("--root", default=".")
    run_parser.add_argument("--campaign-id")
    run_parser.add_argument("--job-id")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    root = Path(args.root)

    if args.command == "run-job":
        payload = run_next_job(root, campaign_id=args.campaign_id, job_id=args.job_id)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
