from __future__ import annotations

from pathlib import Path

from aether_traj.optimizer_workflow import run_scalar_optimizer_workflow


def run_sep_jax_workflow(profile: str, output_dir: Path, show_progress: bool | None = None) -> None:
    run_scalar_optimizer_workflow(profile=profile, output_dir=output_dir, workflow_id="sep_jax", show_progress=show_progress)
