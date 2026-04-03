from __future__ import annotations

import argparse
import csv
from pathlib import Path

from aether_traj.experiments import (
    WORKFLOW_SPECS,
    aggregate_run_manifests,
    default_run_output_dir,
    load_target,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or aggregate Aether laser-SEP study workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a workflow/profile pair into an isolated output directory.")
    run_parser.add_argument("workflow", choices=sorted(WORKFLOW_SPECS))
    run_parser.add_argument("profile")
    run_parser.add_argument("--output-dir")
    run_parser.add_argument("--show-progress", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate run manifests into a CSV.")
    aggregate_parser.add_argument("--root", default="outputs/runs")
    aggregate_parser.add_argument("--output", default="outputs/runs/aggregate.csv")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        spec = WORKFLOW_SPECS[args.workflow]
        if args.profile not in spec.profiles:
            parser.error(f"{args.workflow} does not support profile {args.profile}")
        runner = load_target(spec.runner_target)
        output_dir = Path(args.output_dir) if args.output_dir else default_run_output_dir(args.workflow, args.profile)
        show_progress = True if args.show_progress else None
        runner(profile=args.profile, output_dir=output_dir, show_progress=show_progress)
        print(output_dir)
        return

    rows = aggregate_run_manifests(Path(args.root))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        output_path.write_text("", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
