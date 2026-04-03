from __future__ import annotations

import argparse
import json
from pathlib import Path

from aether_traj.campaign_manager import (
    DEFAULT_CAMPAIGN_SPEC_PATH,
    create_campaign_from_spec_path,
    list_campaigns,
    load_campaign_spec,
    plan_campaign,
    summarize_campaign,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan, launch, and summarize Aether laser-SEP experiment campaigns.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Preview the job matrix for a campaign spec.")
    plan_parser.add_argument("--spec", default=str(DEFAULT_CAMPAIGN_SPEC_PATH))
    plan_parser.add_argument("--root", default=".")

    launch_parser = subparsers.add_parser("launch", help="Materialize a campaign from a TOML spec.")
    launch_parser.add_argument("--spec", default=str(DEFAULT_CAMPAIGN_SPEC_PATH))
    launch_parser.add_argument("--root", default=".")

    summarize_parser = subparsers.add_parser("summarize", help="Build leaderboard and observation outputs for a campaign.")
    summarize_parser.add_argument("campaign_id")
    summarize_parser.add_argument("--root", default=".")

    list_parser = subparsers.add_parser("list", help="List known campaigns.")
    list_parser.add_argument("--root", default=".")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    root = Path(args.root)

    if args.command == "plan":
        spec = load_campaign_spec(root / args.spec)
        print(json.dumps(plan_campaign(spec), indent=2, sort_keys=True))
        return

    if args.command == "launch":
        payload = create_campaign_from_spec_path(root, Path(args.spec))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "summarize":
        payload = summarize_campaign(root, args.campaign_id)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "list":
        print(json.dumps(list_campaigns(root), indent=2, sort_keys=True))
        return

    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
