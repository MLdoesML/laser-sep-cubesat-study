from __future__ import annotations

import argparse
from pathlib import Path

from aether_traj.experiments import default_run_output_dir
from aether_traj.sep_lbfgs_workflow import run_sep_lbfgs_workflow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Aether differentiable SEP L-BFGS-B workflow.")
    parser.add_argument("--profile", default="sep_baseline_direct_capture")
    parser.add_argument("--output-dir")
    parser.add_argument("--show-progress", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_run_output_dir("sep_lbfgs", args.profile)
    run_sep_lbfgs_workflow(profile=args.profile, output_dir=output_dir, show_progress=args.show_progress)
    print(output_dir)


if __name__ == "__main__":
    main()
