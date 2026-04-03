from __future__ import annotations

from datetime import datetime
from importlib import import_module
import json
import math
from pathlib import Path
import subprocess
from typing import Any

from aether_traj.models import WorkflowSpec


WORKFLOW_SPECS: dict[str, WorkflowSpec] = {
    "sep_jax": WorkflowSpec(
        workflow_id="sep_jax",
        kind="scalar",
        runner_target="aether_traj.sep_jax_workflow:run_sep_jax_workflow",
        profile_check_target="aether_traj.config:build_sep_run_profile",
        profiles=(
            "sep_baseline_direct_capture",
            "laser_perigee_boost",
            "laser_lunar_brake",
            "laser_dual_window_fixed",
        ),
        scripts={
            "sep_baseline_direct_capture": "aether-study-jax",
            "laser_perigee_boost": "aether-study-jax",
            "laser_lunar_brake": "aether-study-jax",
            "laser_dual_window_fixed": "aether-study-jax",
        },
    ),
    "sep_lbfgs": WorkflowSpec(
        workflow_id="sep_lbfgs",
        kind="scalar",
        runner_target="aether_traj.sep_lbfgs_workflow:run_sep_lbfgs_workflow",
        profile_check_target="aether_traj.config:build_sep_run_profile",
        profiles=(
            "sep_baseline_direct_capture",
            "laser_perigee_boost",
            "laser_lunar_brake",
            "laser_dual_window_fixed",
        ),
        scripts={
            "sep_baseline_direct_capture": "aether-study-lbfgs",
            "laser_perigee_boost": "aether-study-lbfgs",
            "laser_lunar_brake": "aether-study-lbfgs",
            "laser_dual_window_fixed": "aether-study-lbfgs",
        },
    ),
    "sep_de": WorkflowSpec(
        workflow_id="sep_de",
        kind="scalar",
        runner_target="aether_traj.sep_de_workflow:run_sep_de_workflow",
        profile_check_target="aether_traj.config:build_sep_run_profile",
        profiles=(
            "sep_baseline_direct_capture",
            "laser_perigee_boost",
            "laser_lunar_brake",
            "laser_dual_window_fixed",
        ),
        scripts={
            "sep_baseline_direct_capture": "aether-study-de",
            "laser_perigee_boost": "aether-study-de",
            "laser_lunar_brake": "aether-study-de",
            "laser_dual_window_fixed": "aether-study-de",
        },
    ),
    "sep_pso": WorkflowSpec(
        workflow_id="sep_pso",
        kind="scalar",
        runner_target="aether_traj.sep_pso_workflow:run_sep_pso_workflow",
        profile_check_target="aether_traj.config:build_sep_run_profile",
        profiles=(
            "sep_baseline_direct_capture",
            "laser_perigee_boost",
            "laser_lunar_brake",
            "laser_dual_window_fixed",
        ),
        scripts={
            "sep_baseline_direct_capture": "aether-study-pso",
            "laser_perigee_boost": "aether-study-pso",
            "laser_lunar_brake": "aether-study-pso",
            "laser_dual_window_fixed": "aether-study-pso",
        },
    ),
}


def load_target(target: str) -> Any:
    module_name, attr_name = target.split(":")
    module = import_module(module_name)
    return getattr(module, attr_name)


def expected_script_targets() -> dict[str, str]:
    targets: dict[str, str] = {
        "aether-campaign": "aether_traj.campaign_cli:main",
        "aether-experiment": "aether_traj.experiment_cli:main",
        "aether-worker": "aether_traj.worker_cli:main",
        "aether-study-jax": "aether_traj.sep_jax_cli:main",
        "aether-study-lbfgs": "aether_traj.sep_lbfgs_cli:main",
        "aether-study-de": "aether_traj.sep_de_cli:main",
        "aether-study-pso": "aether_traj.sep_pso_cli:main",
    }
    return targets


def _sanitize(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def current_git_sha() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return completed.stdout.strip() or None
    except Exception:
        return None


def build_run_id(workflow_id: str, profile: str) -> str:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{workflow_id.replace('_', '-')}-{profile.replace('_', '-')}"


def default_run_output_dir(workflow_id: str, profile: str, root: Path | None = None) -> Path:
    base = root or Path("outputs/runs")
    return base / build_run_id(workflow_id, profile)


def write_run_manifest(output_dir: Path, workflow_id: str, profile: str, summary_filename: str, extra: dict[str, object] | None = None) -> None:
    spec = WORKFLOW_SPECS[workflow_id]
    manifest = {
        "schema_version": 1,
        "run_id": output_dir.name,
        "workflow_id": workflow_id,
        "workflow_kind": spec.kind,
        "profile": profile,
        "script_name": spec.scripts[profile],
        "command_hint": f"uv run {spec.scripts[profile]} {profile}",
        "git_sha": current_git_sha(),
        "output_dir": str(output_dir),
        "summary_filename": summary_filename,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "completed",
    }
    if extra:
        manifest.update(extra)
    write_json_file(output_dir / "run_manifest.json", manifest)


def aggregate_run_manifests(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for manifest_path in sorted(root.glob("**/run_manifest.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary_path = manifest_path.parent / str(payload.get("summary_filename", ""))
        summary: dict[str, object] = {}
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}
        selected_validation = summary.get("selected_validation")
        if isinstance(selected_validation, dict):
            classification = selected_validation.get("classification")
            perilune = selected_validation.get("perilune_altitude_km")
            capture = selected_validation.get("capture_duration_days")
        else:
            classification = payload.get("classification")
            perilune = payload.get("perilune_altitude_km")
            capture = payload.get("capture_duration_days")
        rows.append(
            {
                "run_id": payload.get("run_id"),
                "workflow_id": payload.get("workflow_id"),
                "profile": payload.get("profile"),
                "script_name": payload.get("script_name"),
                "git_sha": payload.get("git_sha"),
                "summary_filename": payload.get("summary_filename"),
                "classification": classification,
                "perilune_altitude_km": perilune,
                "capture_duration_days": capture,
            }
        )
    return rows
