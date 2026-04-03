from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import re
import tomllib
from typing import Any
from unittest.mock import patch

from aether_traj.config import DEFAULT_CAMPAIGN_SPEC_PATH, DEFAULT_MANAGER_PROGRAM_PATH
from aether_traj.experiments import WORKFLOW_SPECS, current_git_sha, load_target
from aether_traj.models import SepRunProfile, SimulationConfig
from aether_traj.run_catalog import SCALAR_WORKFLOW_IDS, load_run, load_runs


MANAGER_ROOT = Path("outputs/manager")
CAMPAIGNS_ROOT = MANAGER_ROOT / "campaigns"
RUNS_ROOT = Path("outputs/runs")
DEFAULT_MUTATION_KNOBS = (
    "bounds",
    "seed_priors",
    "optimizer_hyperparameters",
    "profile_allocation",
    "promotion_thresholds",
)
PROTECTED_MUTATION_PATTERNS = (
    "src/aether_traj/dynamics.py",
    "src/aether_traj/gravity.py",
    "src/aether_traj/ephemeris.py",
    "src/aether_traj/sep_jax_workflow.py",
)
SAFE_CLASSIFICATIONS = {"direct_capture", "capture_like"}
UNVALIDATED_CLASSIFICATIONS = {None, "", "unvalidated"}


@dataclass(frozen=True)
class CampaignSpec:
    campaign_id: str
    git_sha: str
    physics_model_id: str
    design_space_id: str
    objective_set_id: str
    workflows: tuple[str, ...]
    profiles: tuple[str, ...]
    replicates: int
    truth_budget: int
    max_parallel: int
    parent_campaign_id: str | None = None
    manager_program_path: str = str(DEFAULT_MANAGER_PROGRAM_PATH)
    mutation_knobs: tuple[str, ...] = DEFAULT_MUTATION_KNOBS
    max_attempts: int = 2
    follow_on_children: int = 1
    base_config_overrides: dict[str, Any] = field(default_factory=dict)
    profile_base_config_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    workflow_run_profile_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    workflow_profile_run_profile_overrides: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    promotion_thresholds: dict[str, Any] = field(default_factory=lambda: {"top_k": 3, "max_constraint_violation": 6.0})


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _slugify(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[^a-z0-9._-]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered)
    return lowered.strip("-") or "campaign"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def configured_metadata_backend(root: Path) -> str:
    return os.getenv("AETHER_MANAGER_METADATA_URL") or f"file://{(root / MANAGER_ROOT).resolve()}"


def configured_artifact_backend(root: Path) -> str:
    return os.getenv("AETHER_MANAGER_ARTIFACT_URL") or f"file://{(root / RUNS_ROOT).resolve()}"


def validate_mutation_paths(paths: list[str]) -> None:
    blocked: list[str] = []
    for path in paths:
        normalized = path.replace("\\", "/")
        if normalized.endswith(".py") and any(normalized.endswith(pattern) for pattern in PROTECTED_MUTATION_PATTERNS):
            blocked.append(path)
        elif any(normalized == pattern or normalized.endswith(pattern) for pattern in PROTECTED_MUTATION_PATTERNS):
            blocked.append(path)
    if blocked:
        raise ValueError(f"manager mutation guardrails block edits to protected paths: {', '.join(sorted(blocked))}")


def _coerce_like(template: Any, value: Any) -> Any:
    if isinstance(template, tuple) and isinstance(value, (list, tuple)):
        return tuple(value)
    if isinstance(template, float):
        return float(value)
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, bool):
        return bool(value)
    return value


def _apply_dataclass_overrides(instance: Any, overrides: dict[str, Any]) -> Any:
    if not overrides:
        return instance
    if not is_dataclass(instance):
        raise TypeError(f"cannot override non-dataclass instance: {type(instance)!r}")
    updates: dict[str, Any] = {}
    for key, value in overrides.items():
        if not hasattr(instance, key):
            raise ValueError(f"unknown override field {key!r} for {type(instance).__name__}")
        updates[key] = _coerce_like(getattr(instance, key), value)
    return replace(instance, **updates)


def _ensure_supported_pairs(spec: CampaignSpec) -> None:
    for workflow_id in spec.workflows:
        if workflow_id not in WORKFLOW_SPECS:
            raise ValueError(f"unknown workflow_id: {workflow_id}")
        supported = set(WORKFLOW_SPECS[workflow_id].profiles)
        unsupported = sorted(profile for profile in spec.profiles if profile not in supported)
        if unsupported:
            raise ValueError(f"workflow {workflow_id} does not support profiles: {', '.join(unsupported)}")


def _normalize_nested_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result[str(key)] = dict(item)
    return result


def validate_campaign_spec(spec: CampaignSpec) -> CampaignSpec:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", spec.campaign_id):
        raise ValueError("campaign_id must contain only letters, digits, dot, underscore, or hyphen")
    if spec.replicates < 1:
        raise ValueError("replicates must be >= 1")
    if spec.truth_budget < 1:
        raise ValueError("truth_budget must be >= 1")
    if spec.max_parallel < 1:
        raise ValueError("max_parallel must be >= 1")
    if spec.max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    unknown_knobs = sorted(set(spec.mutation_knobs) - set(DEFAULT_MUTATION_KNOBS))
    if unknown_knobs:
        raise ValueError(f"unknown mutation knobs: {', '.join(unknown_knobs)}")
    _ensure_supported_pairs(spec)
    validate_mutation_paths([spec.manager_program_path])
    return spec


def campaign_spec_from_dict(payload: dict[str, Any]) -> CampaignSpec:
    spec = CampaignSpec(
        campaign_id=_slugify(str(payload["campaign_id"])),
        git_sha=str(payload.get("git_sha") or "HEAD"),
        physics_model_id=str(payload["physics_model_id"]),
        design_space_id=str(payload["design_space_id"]),
        objective_set_id=str(payload["objective_set_id"]),
        workflows=tuple(str(item) for item in payload["workflows"]),
        profiles=tuple(str(item) for item in payload["profiles"]),
        replicates=int(payload.get("replicates", 1)),
        truth_budget=int(payload.get("truth_budget", 3)),
        max_parallel=int(payload.get("max_parallel", 1)),
        parent_campaign_id=(str(payload["parent_campaign_id"]) if payload.get("parent_campaign_id") else None),
        manager_program_path=str(payload.get("manager_program_path", DEFAULT_MANAGER_PROGRAM_PATH)),
        mutation_knobs=tuple(str(item) for item in payload.get("mutation_knobs", DEFAULT_MUTATION_KNOBS)),
        max_attempts=int(payload.get("max_attempts", 2)),
        follow_on_children=int(payload.get("follow_on_children", 1)),
        base_config_overrides=dict(payload.get("base_config_overrides", {})),
        profile_base_config_overrides=_normalize_nested_dict(payload.get("profile_base_config_overrides", {})),
        workflow_run_profile_overrides=_normalize_nested_dict(payload.get("workflow_run_profile_overrides", {})),
        workflow_profile_run_profile_overrides={
            str(workflow): _normalize_nested_dict(mapping)
            for workflow, mapping in dict(payload.get("workflow_profile_run_profile_overrides", {})).items()
            if isinstance(mapping, dict)
        },
        promotion_thresholds=dict(payload.get("promotion_thresholds", {"top_k": 3, "max_constraint_violation": 6.0})),
    )
    return validate_campaign_spec(spec)


def load_campaign_spec(path: Path) -> CampaignSpec:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    return campaign_spec_from_dict(payload)


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    return json.dumps(str(value))


def _toml_append_table(lines: list[str], prefix: str, mapping: dict[str, Any]) -> None:
    scalars = {key: value for key, value in mapping.items() if not isinstance(value, dict)}
    if scalars:
        lines.append(f"[{prefix}]")
        for key, value in scalars.items():
            lines.append(f"{key} = {_toml_literal(value)}")
        lines.append("")
    for key, value in mapping.items():
        if isinstance(value, dict):
            _toml_append_table(lines, f"{prefix}.{key}", value)


def campaign_spec_to_toml(spec: CampaignSpec) -> str:
    lines = [
        f"campaign_id = {_toml_literal(spec.campaign_id)}",
        f"git_sha = {_toml_literal(spec.git_sha)}",
        f"physics_model_id = {_toml_literal(spec.physics_model_id)}",
        f"design_space_id = {_toml_literal(spec.design_space_id)}",
        f"objective_set_id = {_toml_literal(spec.objective_set_id)}",
        f"workflows = {_toml_literal(spec.workflows)}",
        f"profiles = {_toml_literal(spec.profiles)}",
        f"replicates = {spec.replicates}",
        f"truth_budget = {spec.truth_budget}",
        f"max_parallel = {spec.max_parallel}",
        f"manager_program_path = {_toml_literal(spec.manager_program_path)}",
        f"mutation_knobs = {_toml_literal(spec.mutation_knobs)}",
        f"max_attempts = {spec.max_attempts}",
        f"follow_on_children = {spec.follow_on_children}",
    ]
    if spec.parent_campaign_id:
        lines.append(f"parent_campaign_id = {_toml_literal(spec.parent_campaign_id)}")
    lines.append("")
    if spec.base_config_overrides:
        _toml_append_table(lines, "base_config_overrides", spec.base_config_overrides)
    if spec.profile_base_config_overrides:
        _toml_append_table(lines, "profile_base_config_overrides", spec.profile_base_config_overrides)
    if spec.workflow_run_profile_overrides:
        _toml_append_table(lines, "workflow_run_profile_overrides", spec.workflow_run_profile_overrides)
    if spec.workflow_profile_run_profile_overrides:
        _toml_append_table(lines, "workflow_profile_run_profile_overrides", spec.workflow_profile_run_profile_overrides)
    if spec.promotion_thresholds:
        _toml_append_table(lines, "promotion_thresholds", spec.promotion_thresholds)
    return "\n".join(lines).rstrip() + "\n"


def _campaign_dir(root: Path, campaign_id: str) -> Path:
    return root / CAMPAIGNS_ROOT / campaign_id


def _campaign_state_path(root: Path, campaign_id: str) -> Path:
    return _campaign_dir(root, campaign_id) / "campaign_state.json"


def _campaign_spec_path(root: Path, campaign_id: str) -> Path:
    return _campaign_dir(root, campaign_id) / "campaign.toml"


def _campaign_jobs_dir(root: Path, campaign_id: str) -> Path:
    return _campaign_dir(root, campaign_id) / "jobs"


def _campaign_summary_path(root: Path, campaign_id: str) -> Path:
    return _campaign_dir(root, campaign_id) / "campaign_summary.json"


def _campaign_leaderboard_path(root: Path, campaign_id: str) -> Path:
    return _campaign_dir(root, campaign_id) / "leaderboard.json"


def _campaign_leaderboard_csv_path(root: Path, campaign_id: str) -> Path:
    return _campaign_dir(root, campaign_id) / "leaderboard.csv"


def _campaign_observations_path(root: Path, campaign_id: str) -> Path:
    return _campaign_dir(root, campaign_id) / "observations.json"


def _campaign_suggested_spec_path(root: Path, campaign_id: str) -> Path:
    return _campaign_dir(root, campaign_id) / "suggested_next_campaign.toml"


def _job_path(root: Path, campaign_id: str, job_id: str) -> Path:
    return _campaign_jobs_dir(root, campaign_id) / f"{job_id}.json"


def _load_campaign_state(root: Path, campaign_id: str) -> dict[str, Any]:
    payload = _read_json(_campaign_state_path(root, campaign_id))
    if not payload:
        raise KeyError(f"unknown campaign id: {campaign_id}")
    return payload


def _load_campaign_spec_for_id(root: Path, campaign_id: str) -> CampaignSpec:
    spec_path = _campaign_spec_path(root, campaign_id)
    if not spec_path.is_file():
        raise KeyError(f"unknown campaign id: {campaign_id}")
    return load_campaign_spec(spec_path)


def _load_jobs(root: Path, campaign_id: str) -> list[dict[str, Any]]:
    jobs_dir = _campaign_jobs_dir(root, campaign_id)
    if not jobs_dir.is_dir():
        if _campaign_dir(root, campaign_id).exists():
            return []
        raise KeyError(f"unknown campaign id: {campaign_id}")
    return [_read_json(path) for path in sorted(jobs_dir.glob("*.json")) if _read_json(path)]


def _job_output_dir(root: Path, campaign_id: str, workflow_id: str, profile: str, replicate: int, launch_stamp: str, index: int, design_space_id: str) -> Path:
    run_name = f"{launch_stamp}-{index:03d}-{workflow_id.replace('_', '-')}-{profile.replace('_', '-')}-{_slugify(design_space_id)}-r{replicate}"
    return (root / RUNS_ROOT / campaign_id / run_name).resolve()


def _initial_campaign_state(root: Path, spec: CampaignSpec, *, launched_from: str) -> dict[str, Any]:
    return {
        "campaign_id": spec.campaign_id,
        "status": "queued",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "launched_from": launched_from,
        "manager_program_path": spec.manager_program_path,
        "metadata_backend": configured_metadata_backend(root),
        "artifact_backend": configured_artifact_backend(root),
        "git_sha": spec.git_sha,
        "physics_model_id": spec.physics_model_id,
        "design_space_id": spec.design_space_id,
        "objective_set_id": spec.objective_set_id,
        "parent_campaign_id": spec.parent_campaign_id,
        "max_parallel": spec.max_parallel,
        "truth_budget": spec.truth_budget,
        "planned_jobs": 0,
        "pending_jobs": 0,
        "running_jobs": 0,
        "completed_jobs": 0,
        "failed_jobs": 0,
        "cancelled_jobs": 0,
    }


def plan_campaign(spec: CampaignSpec) -> dict[str, Any]:
    matrix: list[dict[str, Any]] = []
    for workflow_id in spec.workflows:
        for profile in spec.profiles:
            for replicate in range(1, spec.replicates + 1):
                matrix.append({"workflow_id": workflow_id, "profile": profile, "replicate": replicate})
    return {
        "campaign_id": spec.campaign_id,
        "job_count": len(matrix),
        "truth_budget": spec.truth_budget,
        "max_parallel": spec.max_parallel,
        "matrix": matrix,
    }


def materialize_campaign(root: Path, spec: CampaignSpec, *, launched_from: str) -> dict[str, Any]:
    root = root.resolve()
    campaign_id = spec.campaign_id
    campaign_dir = _campaign_dir(root, campaign_id)
    if campaign_dir.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        spec = replace(spec, campaign_id=f"{campaign_id}-{stamp}")
        campaign_id = spec.campaign_id
        campaign_dir = _campaign_dir(root, campaign_id)
    campaign_dir.mkdir(parents=True, exist_ok=False)
    _campaign_jobs_dir(root, campaign_id).mkdir(parents=True, exist_ok=True)

    launch_stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_records: list[dict[str, Any]] = []
    index = 0
    for workflow_id in spec.workflows:
        for profile in spec.profiles:
            for replicate in range(1, spec.replicates + 1):
                index += 1
                job_id = f"job-{index:03d}"
                output_dir = _job_output_dir(root, campaign_id, workflow_id, profile, replicate, launch_stamp, index, spec.design_space_id)
                record = {
                    "job_id": job_id,
                    "campaign_id": campaign_id,
                    "workflow_id": workflow_id,
                    "profile": profile,
                    "replicate_id": replicate,
                    "status": "pending",
                    "attempts": 0,
                    "max_attempts": spec.max_attempts,
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                    "output_dir": str(output_dir),
                    "run_id": output_dir.name,
                    "design_space_id": spec.design_space_id,
                    "physics_model_id": spec.physics_model_id,
                    "objective_set_id": spec.objective_set_id,
                    "git_sha": spec.git_sha,
                    "last_error": None,
                    "started_at": None,
                    "finished_at": None,
                    "cancel_requested": False,
                }
                job_records.append(record)
                _write_json(_job_path(root, campaign_id, job_id), record)

    spec_path = _campaign_spec_path(root, campaign_id)
    spec_path.write_text(campaign_spec_to_toml(replace(spec, campaign_id=campaign_id)), encoding="utf-8")
    state = _initial_campaign_state(root, replace(spec, campaign_id=campaign_id), launched_from=launched_from)
    state["planned_jobs"] = len(job_records)
    state["pending_jobs"] = len(job_records)
    _write_json(_campaign_state_path(root, campaign_id), state)
    return load_campaign(root, campaign_id)


def create_campaign_from_payload(root: Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    workspace_root = Path(".") if root is None else root
    spec = campaign_spec_from_dict(payload)
    return materialize_campaign(workspace_root, spec, launched_from="payload")


def create_campaign_from_spec_path(root: Path | None, spec_path: Path) -> dict[str, Any]:
    workspace_root = Path(".") if root is None else root
    spec = load_campaign_spec((workspace_root / spec_path) if not spec_path.is_absolute() else spec_path)
    return materialize_campaign(workspace_root, spec, launched_from=str(spec_path))


def list_campaigns(root: Path | None = None) -> list[dict[str, Any]]:
    workspace_root = Path(".") if root is None else root
    rows: list[dict[str, Any]] = []
    base = workspace_root / CAMPAIGNS_ROOT
    if not base.is_dir():
        return []
    for state_path in sorted(base.glob("*/campaign_state.json")):
        rows.append(load_campaign(workspace_root, state_path.parent.name))
    return sorted(rows, key=lambda row: row.get("created_at") or "", reverse=True)


def _refresh_campaign_counts(root: Path, campaign_id: str) -> dict[str, Any]:
    state = _load_campaign_state(root, campaign_id)
    jobs = _load_jobs(root, campaign_id)
    pending = sum(1 for job in jobs if job.get("status") == "pending")
    running = sum(1 for job in jobs if job.get("status") == "running")
    completed = sum(1 for job in jobs if job.get("status") == "completed")
    failed = sum(1 for job in jobs if job.get("status") == "failed")
    cancelled = sum(1 for job in jobs if job.get("status") == "cancelled")
    if state.get("status") == "paused":
        status = "paused"
    elif running > 0:
        status = "running"
    elif pending > 0:
        status = "queued"
    elif completed > 0 and failed == 0:
        status = "completed"
    elif failed > 0 and completed == 0:
        status = "failed"
    elif cancelled == len(jobs) and jobs:
        status = "cancelled"
    else:
        status = state.get("status", "queued")
    state.update(
        {
            "status": status,
            "updated_at": _now_iso(),
            "planned_jobs": len(jobs),
            "pending_jobs": pending,
            "running_jobs": running,
            "completed_jobs": completed,
            "failed_jobs": failed,
            "cancelled_jobs": cancelled,
        }
    )
    _write_json(_campaign_state_path(root, campaign_id), state)
    return state


def campaign_design_space_diff(root: Path, campaign_id: str) -> dict[str, Any]:
    spec = _load_campaign_spec_for_id(root, campaign_id)
    if not spec.parent_campaign_id:
        return {
            "parent_campaign_id": None,
            "profiles_added": list(spec.profiles),
            "profiles_removed": [],
            "workflows_added": list(spec.workflows),
            "workflows_removed": [],
            "truth_budget_delta": spec.truth_budget,
            "max_parallel_delta": spec.max_parallel,
            "base_config_override_keys_changed": sorted(spec.base_config_overrides),
        }
    parent = _load_campaign_spec_for_id(root, spec.parent_campaign_id)
    return {
        "parent_campaign_id": parent.campaign_id,
        "profiles_added": sorted(set(spec.profiles) - set(parent.profiles)),
        "profiles_removed": sorted(set(parent.profiles) - set(spec.profiles)),
        "workflows_added": sorted(set(spec.workflows) - set(parent.workflows)),
        "workflows_removed": sorted(set(parent.workflows) - set(spec.workflows)),
        "truth_budget_delta": spec.truth_budget - parent.truth_budget,
        "max_parallel_delta": spec.max_parallel - parent.max_parallel,
        "base_config_override_keys_changed": sorted(
            set(spec.base_config_overrides) ^ set(parent.base_config_overrides)
            | {key for key in spec.base_config_overrides if parent.base_config_overrides.get(key) != spec.base_config_overrides.get(key)}
        ),
    }


def load_campaign(root: Path | None, campaign_id: str) -> dict[str, Any]:
    workspace_root = Path(".") if root is None else root
    state = _refresh_campaign_counts(workspace_root, campaign_id)
    spec = _load_campaign_spec_for_id(workspace_root, campaign_id)
    jobs = _load_jobs(workspace_root, campaign_id)
    leaderboard = _read_json(_campaign_leaderboard_path(workspace_root, campaign_id)).get("rows", [])
    observations = _read_json(_campaign_observations_path(workspace_root, campaign_id)).get("observations", [])
    summary = _read_json(_campaign_summary_path(workspace_root, campaign_id))
    return {
        **state,
        "spec": asdict(spec),
        "jobs": jobs,
        "leaderboard_path": str(_campaign_leaderboard_path(workspace_root, campaign_id)),
        "observations_path": str(_campaign_observations_path(workspace_root, campaign_id)),
        "suggested_next_campaign_path": str(_campaign_suggested_spec_path(workspace_root, campaign_id)),
        "leaderboard_count": len(leaderboard),
        "observation_count": len(observations),
        "top_observation": observations[0] if observations else None,
        "design_space_diff": campaign_design_space_diff(workspace_root, campaign_id),
        "summary": summary,
    }


def _builder_patch_target(workflow_id: str) -> str:
    module_name, attr_name = WORKFLOW_SPECS[workflow_id].profile_check_target.split(":")
    return f"{module_name}.{attr_name}"


def _build_overridden_profile(spec: CampaignSpec, workflow_id: str, profile: str) -> Any:
    builder = load_target(WORKFLOW_SPECS[workflow_id].profile_check_target)
    profile_obj = builder(profile)
    base_overrides = dict(spec.base_config_overrides)
    base_overrides.update(spec.profile_base_config_overrides.get(profile, {}))
    if isinstance(profile_obj, SimulationConfig):
        return _apply_dataclass_overrides(profile_obj, base_overrides)
    if isinstance(profile_obj, SepRunProfile):
        run_overrides = dict(spec.workflow_run_profile_overrides.get(workflow_id, {}))
        run_overrides.update(spec.workflow_profile_run_profile_overrides.get(workflow_id, {}).get(profile, {}))
        if "truth_candidate_count" not in run_overrides:
            run_overrides["truth_candidate_count"] = spec.truth_budget
        profile_obj = replace(profile_obj, base_config=_apply_dataclass_overrides(profile_obj.base_config, base_overrides))
        if run_overrides:
            profile_obj = _apply_dataclass_overrides(profile_obj, run_overrides)
        return profile_obj
    if is_dataclass(profile_obj):
        return _apply_dataclass_overrides(profile_obj, base_overrides)
    return profile_obj


def _classification_priority(classification: Any, perilune_altitude_km: Any, capture_duration_days: Any) -> float:
    cls = str(classification or "").lower()
    perilune = _safe_float(perilune_altitude_km)
    capture = _safe_float(capture_duration_days) or 0.0
    mapping = {
        "direct_capture": 8.0,
        "capture_like": 5.0,
        "flyby": 2.0,
        "escape": 0.5,
        "unsafe_perilune": -4.0,
        "unvalidated": 1.0,
    }
    score = mapping.get(cls, 1.0)
    if perilune is not None:
        if perilune >= 100.0:
            score += 0.5
        elif perilune < 30.0:
            score -= 2.5
    score += min(max(capture, 0.0), 3.0)
    return score


def _truth_status_for_run(run: dict[str, Any]) -> str:
    candidates = run.get("candidates") or []
    if any(candidate.get("classification") not in UNVALIDATED_CLASSIFICATIONS for candidate in candidates):
        return "validated"
    return "unvalidated"


def _normalized_summary_from_run(run: dict[str, Any], spec: CampaignSpec, job: dict[str, Any]) -> dict[str, Any]:
    top = (run.get("candidates") or [None])[0]
    summary = run.get("summary") or {}
    return {
        "run_id": run["run_id"],
        "campaign_id": spec.campaign_id,
        "workflow_id": run["workflow_id"],
        "workflow_kind": run["workflow_kind"],
        "profile": run["profile"],
        "design_space_id": spec.design_space_id,
        "physics_model_id": spec.physics_model_id,
        "objective_set_id": spec.objective_set_id,
        "replicate_id": job["replicate_id"],
        "job_id": job["job_id"],
        "git_sha": run.get("git_sha"),
        "status": job.get("status"),
        "truth_status": _truth_status_for_run(run),
        "candidate_count": run.get("candidate_count", 0),
        "replayable_candidate_count": run.get("replayable_candidate_count", 0),
        "top_candidate_id": None if top is None else top.get("candidate_id"),
        "top_classification": None if top is None else top.get("classification"),
        "top_perilune_altitude_km": None if top is None else top.get("perilune_altitude_km"),
        "top_capture_duration_days": None if top is None else top.get("capture_duration_days"),
        "top_capture_apoapsis_km": None if top is None else top.get("capture_apoapsis_km"),
        "top_final_mass_kg": None if top is None else top.get("final_mass_kg"),
        "top_total_beam_energy_mj": None if top is None else top.get("total_beam_energy_mj"),
        "top_surrogate_objective": None if top is None else top.get("surrogate_objective"),
        "top_physical_constraint_violation": None if top is None else top.get("physical_constraint_violation"),
        "top_physics_priority_score": _classification_priority(
            None if top is None else top.get("classification"),
            None if top is None else top.get("perilune_altitude_km"),
            None if top is None else top.get("capture_duration_days"),
        ),
        "best_objective": summary.get("best_objective"),
        "created_at": run.get("created_at"),
        "output_dir": run.get("output_dir"),
    }


def _extract_design_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    raw = candidate.get("raw")
    if isinstance(raw, dict):
        for key in (
            "departure_apogee_km",
            "departure_perigee_altitude_km",
            "epoch_shift_days",
            "departure_wait_days",
            "speed_scale",
            "boosted_perigee_passes",
            "perigee_boost_power_scale",
            "lunar_brake_power_scale",
        ):
            if key in raw:
                extracted[key] = raw[key]
    return extracted


def _candidate_catalog_rows(run: dict[str, Any], normalized_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in run.get("_internal_candidates", []):
        row = {
            "campaign_id": normalized_summary["campaign_id"],
            "run_id": normalized_summary["run_id"],
            "workflow_id": normalized_summary["workflow_id"],
            "profile": normalized_summary["profile"],
            "design_space_id": normalized_summary["design_space_id"],
            "physics_model_id": normalized_summary["physics_model_id"],
            "objective_set_id": normalized_summary["objective_set_id"],
            "truth_status": normalized_summary["truth_status"],
            "replicate_id": normalized_summary["replicate_id"],
            "candidate_id": candidate.get("candidate_id"),
            "rank": candidate.get("rank"),
            "source": candidate.get("source"),
            "classification": candidate.get("classification"),
            "capture_duration_days": candidate.get("capture_duration_days"),
            "perilune_altitude_km": candidate.get("perilune_altitude_km"),
            "capture_apoapsis_km": candidate.get("capture_apoapsis_km"),
            "final_mass_kg": candidate.get("final_mass_kg"),
            "propellant_used_kg": candidate.get("propellant_used_kg"),
            "total_beam_energy_mj": candidate.get("total_beam_energy_mj"),
            "perigee_boost_dwell_hours": candidate.get("perigee_boost_dwell_hours"),
            "lunar_brake_dwell_hours": candidate.get("lunar_brake_dwell_hours"),
            "surrogate_objective": candidate.get("surrogate_objective"),
            "physical_constraint_violation": candidate.get("physical_constraint_violation"),
            "replay_available": candidate.get("replay_available"),
            "physics_priority_score": _classification_priority(
                candidate.get("classification"),
                candidate.get("perilune_altitude_km"),
                candidate.get("capture_duration_days"),
            ),
        }
        row.update(_extract_design_fields(candidate))
        rows.append(row)
    return rows


def _observation_signals(normalized_summary: dict[str, Any], candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    safe_candidates = [row for row in candidate_rows if str(row.get("classification") or "").lower() in SAFE_CLASSIFICATIONS]
    beam_enabled = normalized_summary["profile"] != "sep_baseline_direct_capture"
    return {
        "run_id": normalized_summary["run_id"],
        "campaign_id": normalized_summary["campaign_id"],
        "workflow_id": normalized_summary["workflow_id"],
        "profile": normalized_summary["profile"],
        "design_space_id": normalized_summary["design_space_id"],
        "truth_status": normalized_summary["truth_status"],
        "top_physics_priority_score": normalized_summary["top_physics_priority_score"],
        "safe_candidate_count": len(safe_candidates),
        "validated_candidate_count": sum(1 for row in candidate_rows if row.get("classification") not in UNVALIDATED_CLASSIFICATIONS),
        "best_surrogate_objective": normalized_summary.get("top_surrogate_objective"),
        "top_classification": normalized_summary.get("top_classification"),
        "beam_enabled": beam_enabled,
    }


def normalize_managed_run(root: Path, output_dir: Path, spec: CampaignSpec, job: dict[str, Any]) -> dict[str, Any]:
    run = load_run(root, output_dir.name, include_internal=True)
    normalized_summary = _normalized_summary_from_run(run, spec, job)
    candidate_rows = _candidate_catalog_rows(run, normalized_summary)
    signals = _observation_signals(normalized_summary, candidate_rows)
    _write_json(output_dir / "normalized_summary.json", normalized_summary)
    _write_csv(output_dir / "candidate_catalog.csv", candidate_rows)
    _write_json(output_dir / "observation_signals.json", signals)
    return normalized_summary


def _augment_run_manifest(output_dir: Path, spec: CampaignSpec, job: dict[str, Any], root: Path) -> None:
    manifest_path = output_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest.update(
        {
            "campaign_id": spec.campaign_id,
            "design_space_id": spec.design_space_id,
            "physics_model_id": spec.physics_model_id,
            "objective_set_id": spec.objective_set_id,
            "replicate_id": job["replicate_id"],
            "job_id": job["job_id"],
            "managed": True,
            "metadata_backend": configured_metadata_backend(root),
            "artifact_backend": configured_artifact_backend(root),
        }
    )
    _write_json(manifest_path, manifest)


def _ensure_requested_git_sha(spec: CampaignSpec) -> None:
    requested = spec.git_sha
    if requested in {"", "HEAD"}:
        return
    current = current_git_sha()
    if current == requested:
        return
    raise RuntimeError(
        f"worker git SHA mismatch: requested {requested}, current {current}. "
        "Managed execution currently requires the requested SHA to already be checked out."
    )


def _update_job(root: Path, campaign_id: str, job: dict[str, Any]) -> None:
    job["updated_at"] = _now_iso()
    _write_json(_job_path(root, campaign_id, str(job["job_id"])), job)


def _claim_next_job(root: Path, campaign_id: str | None = None) -> dict[str, Any] | None:
    campaigns = [campaign_id] if campaign_id is not None else [row["campaign_id"] for row in list_campaigns(root)]
    for current_campaign_id in campaigns:
        if not current_campaign_id:
            continue
        state = _refresh_campaign_counts(root, current_campaign_id)
        if state.get("status") == "paused":
            continue
        if int(state.get("running_jobs", 0)) >= int(state.get("max_parallel", 1)):
            continue
        jobs = _load_jobs(root, current_campaign_id)
        pending = [job for job in jobs if job.get("status") == "pending" and not job.get("cancel_requested")]
        if not pending:
            continue
        job = pending[0]
        job["status"] = "running"
        job["started_at"] = _now_iso()
        job["attempts"] = int(job.get("attempts", 0)) + 1
        _update_job(root, current_campaign_id, job)
        _refresh_campaign_counts(root, current_campaign_id)
        return job
    return None


def cancel_job(root: Path | None, job_id: str) -> dict[str, Any]:
    workspace_root = Path(".") if root is None else root
    for campaign in list_campaigns(workspace_root):
        for job in _load_jobs(workspace_root, campaign["campaign_id"]):
            if job.get("job_id") != job_id:
                continue
            if job.get("status") == "pending":
                job["status"] = "cancelled"
                job["finished_at"] = _now_iso()
            else:
                job["cancel_requested"] = True
            _update_job(workspace_root, campaign["campaign_id"], job)
            _refresh_campaign_counts(workspace_root, campaign["campaign_id"])
            return job
    raise KeyError(f"unknown job id: {job_id}")


def pause_campaign(root: Path | None, campaign_id: str) -> dict[str, Any]:
    workspace_root = Path(".") if root is None else root
    state = _load_campaign_state(workspace_root, campaign_id)
    state["status"] = "paused"
    _write_json(_campaign_state_path(workspace_root, campaign_id), state)
    return load_campaign(workspace_root, campaign_id)


def resume_campaign(root: Path | None, campaign_id: str) -> dict[str, Any]:
    workspace_root = Path(".") if root is None else root
    state = _load_campaign_state(workspace_root, campaign_id)
    state["status"] = "queued"
    _write_json(_campaign_state_path(workspace_root, campaign_id), state)
    return load_campaign(workspace_root, campaign_id)


def _run_job(root: Path, job: dict[str, Any]) -> dict[str, Any]:
    spec = _load_campaign_spec_for_id(root, str(job["campaign_id"]))
    _ensure_requested_git_sha(spec)
    workflow_id = str(job["workflow_id"])
    profile = str(job["profile"])
    output_dir = Path(str(job["output_dir"]))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    overridden_profile = _build_overridden_profile(spec, workflow_id, profile)
    runner = load_target(WORKFLOW_SPECS[workflow_id].runner_target)
    with patch(_builder_patch_target(workflow_id), return_value=overridden_profile):
        runner(profile=profile, output_dir=output_dir, show_progress=False)
    _augment_run_manifest(output_dir, spec, job, root)
    normalize_managed_run(root, output_dir, spec, job)
    return job


def run_next_job(root: Path | None = None, *, campaign_id: str | None = None, job_id: str | None = None) -> dict[str, Any] | None:
    workspace_root = Path(".") if root is None else root
    if job_id is not None:
        candidate_job = None
        for campaign in list_campaigns(workspace_root):
            for existing_job in _load_jobs(workspace_root, campaign["campaign_id"]):
                if existing_job.get("job_id") == job_id:
                    candidate_job = existing_job
                    break
            if candidate_job is not None:
                break
        if candidate_job is None:
            raise KeyError(f"unknown job id: {job_id}")
        if candidate_job.get("status") != "pending":
            return candidate_job
        job = candidate_job
        job["status"] = "running"
        job["started_at"] = _now_iso()
        job["attempts"] = int(job.get("attempts", 0)) + 1
        _update_job(workspace_root, str(job["campaign_id"]), job)
    else:
        job = _claim_next_job(workspace_root, campaign_id=campaign_id)
        if job is None:
            return None

    current_campaign_id = str(job["campaign_id"])
    try:
        _run_job(workspace_root, job)
        job["status"] = "completed"
        job["finished_at"] = _now_iso()
        job["last_error"] = None
    except Exception as exc:
        if int(job.get("attempts", 0)) < int(job.get("max_attempts", 1)) and not job.get("cancel_requested"):
            job["status"] = "pending"
        else:
            job["status"] = "failed"
            job["finished_at"] = _now_iso()
        job["last_error"] = str(exc)
    _update_job(workspace_root, current_campaign_id, job)
    _refresh_campaign_counts(workspace_root, current_campaign_id)
    if job.get("status") == "completed":
        summarize_campaign(workspace_root, current_campaign_id)
    return job


def _load_campaign_run_records(root: Path, campaign_id: str) -> list[dict[str, Any]]:
    return [run for run in load_runs(root) if run.get("campaign_id") == campaign_id]


def load_campaign_runs(root: Path | None, campaign_id: str) -> list[dict[str, Any]]:
    workspace_root = Path(".") if root is None else root
    _load_campaign_spec_for_id(workspace_root, campaign_id)
    return _load_campaign_run_records(workspace_root, campaign_id)


def _candidate_rows_for_campaign(root: Path, campaign_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in _load_campaign_run_records(root, campaign_id):
        output_dir = Path(str(run["output_dir"]))
        rows.extend(_read_csv_rows(output_dir / "candidate_catalog.csv"))
    return rows


def _safe_float(value: Any) -> float | None:
    try:
        cast = float(value)
    except Exception:
        return None
    if not math.isfinite(cast):
        return None
    return cast


def _leaderboard_rows(root: Path, campaign_id: str) -> list[dict[str, Any]]:
    rows = _candidate_rows_for_campaign(root, campaign_id)
    leaderboard: list[dict[str, Any]] = []
    for row in rows:
        classification = row.get("classification")
        perilune = _safe_float(row.get("perilune_altitude_km"))
        capture = _safe_float(row.get("capture_duration_days")) or 0.0
        score = _classification_priority(classification, perilune, capture)
        beam_energy = _safe_float(row.get("total_beam_energy_mj")) or 0.0
        station_burden = beam_energy + 5.0 * ((_safe_float(row.get("perigee_boost_dwell_hours")) or 0.0) + (_safe_float(row.get("lunar_brake_dwell_hours")) or 0.0))
        leaderboard.append(
            {
                **row,
                "physics_priority_score": score,
                "safe_perilune": bool(perilune is not None and perilune >= 100.0),
                "station_infrastructure_burden": station_burden,
                "truth_status": row.get("truth_status") or ("validated" if classification not in UNVALIDATED_CLASSIFICATIONS else "unvalidated"),
            }
        )
    leaderboard.sort(
        key=lambda row: (
            -(_safe_float(row.get("physics_priority_score")) or 0.0),
            -(_safe_float(row.get("capture_duration_days")) or 0.0),
            _safe_float(row.get("station_infrastructure_burden")) or float("inf"),
        )
    )
    return leaderboard


def _best_rows_by_key(rows: list[dict[str, Any]], key_name: str) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(key_name) or "")
        if not key:
            continue
        if key not in best or float(row.get("physics_priority_score") or 0.0) > float(best[key].get("physics_priority_score") or 0.0):
            best[key] = row
    return best


def _consensus_observations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("design_space_id") or ""), str(row.get("profile") or ""))
        grouped.setdefault(key, []).append(row)
    observations: list[dict[str, Any]] = []
    for (design_space_id, profile), group_rows in grouped.items():
        safe_rows = [row for row in group_rows if str(row.get("classification") or "").lower() in SAFE_CLASSIFICATIONS and row.get("safe_perilune")]
        workflows = sorted({str(row.get("workflow_id")) for row in safe_rows})
        if len(workflows) < 2:
            continue
        observations.append(
            {
                "title": f"{profile} shows repeatable capture behaviour across {len(workflows)} workflows",
                "body": f"Design space {design_space_id} now has safe, truth-validated capture-like candidates in {', '.join(workflows)}.",
                "score": 9.0 + min(len(workflows), 4) * 0.25,
                "kind": "consensus",
                "workflow_ids": workflows,
                "profiles": [profile],
                "run_ids": sorted({str(row.get('run_id')) for row in safe_rows}),
            }
        )
    return observations


def _surrogate_truth_gap_observations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sortable = [row for row in rows if row.get("surrogate_objective") not in {None, ""}]
    if not sortable:
        return []
    ordered = sorted(sortable, key=lambda row: _safe_float(row.get("surrogate_objective")) or float("inf"))
    cutoff = max(1, len(ordered) // 3)
    observations: list[dict[str, Any]] = []
    for row in ordered[:cutoff]:
        if str(row.get("classification") or "").lower() in SAFE_CLASSIFICATIONS:
            continue
        observations.append(
            {
                "title": f"{row.get('workflow_id')} looks promising in surrogate but misses in truth on {row.get('profile')}",
                "body": (
                    f"Run {row.get('run_id')} has a strong surrogate ranking ({row.get('surrogate_objective')}) "
                    f"but truth classification remains {row.get('classification')}."
                ),
                "score": 7.5,
                "kind": "surrogate_truth_gap",
                "workflow_ids": [row.get("workflow_id")],
                "profiles": [row.get("profile")],
                "run_ids": [row.get("run_id")],
            }
        )
    return observations


def _beam_value_observations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_profile = _best_rows_by_key(rows, "profile")
    baseline = best_by_profile.get("sep_baseline_direct_capture")
    observations: list[dict[str, Any]] = []
    comparisons = {
        "laser_perigee_boost": "perigee_boost_marginal_value",
        "laser_lunar_brake": "lunar_brake_marginal_value",
    }
    for profile, kind in comparisons.items():
        row = best_by_profile.get(profile)
        if baseline is None or row is None:
            continue
        delta = float(row.get("physics_priority_score") or 0.0) - float(baseline.get("physics_priority_score") or 0.0)
        if delta <= 0.5:
            continue
        observations.append(
            {
                "title": f"{profile} improves on the SEP-only baseline",
                "body": (
                    f"Profile {profile} improves the best physics-first score by {delta:.2f} relative to "
                    f"the SEP-only baseline."
                ),
                "score": 8.0,
                "kind": kind,
                "workflow_ids": sorted({row.get("workflow_id"), baseline.get("workflow_id")}),
                "profiles": ["sep_baseline_direct_capture", profile],
                "run_ids": [baseline.get("run_id"), row.get("run_id")],
            }
        )
        if str(baseline.get("classification") or "").lower() not in SAFE_CLASSIFICATIONS and str(row.get("classification") or "").lower() in SAFE_CLASSIFICATIONS:
            observations.append(
                {
                    "title": f"{profile} enables capture that the baseline does not close",
                    "body": f"The best {profile} candidate reaches {row.get('classification')} while the SEP-only baseline remains {baseline.get('classification')}.",
                    "score": 8.5,
                    "kind": "capture_enabled_by_beam",
                    "workflow_ids": sorted({row.get("workflow_id"), baseline.get("workflow_id")}),
                    "profiles": ["sep_baseline_direct_capture", profile],
                    "run_ids": [baseline.get("run_id"), row.get("run_id")],
                }
            )
    return observations


def _reuse_and_burden_observations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for row in rows:
        beam_energy = _safe_float(row.get("total_beam_energy_mj")) or 0.0
        burden = _safe_float(row.get("station_infrastructure_burden")) or beam_energy
        classification = str(row.get("classification") or "").lower()
        if classification == "direct_capture" and burden <= 35.0:
            observations.append(
                {
                    "title": f"{row.get('profile')} looks reusable rather than one-off",
                    "body": (
                        f"Run {row.get('run_id')} reaches direct capture with a moderate station burden "
                        f"({burden:.2f} in the current proxy metric)."
                    ),
                    "score": 6.8,
                    "kind": "reuse_candidate",
                    "workflow_ids": [row.get("workflow_id")],
                    "profiles": [row.get("profile")],
                    "run_ids": [row.get("run_id")],
                }
            )
        if classification not in SAFE_CLASSIFICATIONS and burden >= 40.0:
            observations.append(
                {
                    "title": f"{row.get('profile')} is paying heavy station cost without capture payoff",
                    "body": f"Run {row.get('run_id')} carries a high station-burden proxy ({burden:.2f}) but truth classification remains {row.get('classification')}.",
                    "score": 6.2,
                    "kind": "station_burden_regression",
                    "workflow_ids": [row.get("workflow_id")],
                    "profiles": [row.get("profile")],
                    "run_ids": [row.get("run_id")],
                }
            )
    return observations


def _parent_improvement_observations(root: Path, campaign_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spec = _load_campaign_spec_for_id(root, campaign_id)
    if not spec.parent_campaign_id:
        return []
    parent_rows = _leaderboard_rows(root, spec.parent_campaign_id)
    if not parent_rows or not rows:
        return []
    current_best = rows[0]
    parent_best = parent_rows[0]
    if float(current_best.get("physics_priority_score") or 0.0) <= float(parent_best.get("physics_priority_score") or 0.0):
        return []
    return [
        {
            "title": f"{campaign_id} improves on parent campaign {spec.parent_campaign_id}",
            "body": (
                f"The best candidate in {campaign_id} outranks the parent campaign on the physics-first score "
                f"({current_best.get('physics_priority_score')} vs {parent_best.get('physics_priority_score')})."
            ),
            "score": 8.0,
            "kind": "campaign_improvement",
            "workflow_ids": [current_best.get("workflow_id")],
            "profiles": [current_best.get("profile")],
            "run_ids": [current_best.get("run_id"), parent_best.get("run_id")],
        }
    ]


def _observations_for_campaign(root: Path, campaign_id: str, leaderboard_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    observations.extend(_consensus_observations(leaderboard_rows))
    observations.extend(_surrogate_truth_gap_observations(leaderboard_rows))
    observations.extend(_beam_value_observations(leaderboard_rows))
    observations.extend(_reuse_and_burden_observations(leaderboard_rows))
    observations.extend(_parent_improvement_observations(root, campaign_id, leaderboard_rows))
    observations.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    return observations


def _suggest_follow_on_campaign(root: Path, campaign_id: str, leaderboard_rows: list[dict[str, Any]]) -> CampaignSpec | None:
    if not leaderboard_rows:
        return None
    spec = _load_campaign_spec_for_id(root, campaign_id)
    best_by_profile = _best_rows_by_key(leaderboard_rows, "profile")
    ordered_profiles = sorted(best_by_profile.values(), key=lambda row: float(row.get("physics_priority_score") or 0.0), reverse=True)
    selected_profiles = [str(row.get("profile")) for row in ordered_profiles[: max(1, min(spec.follow_on_children + 1, len(ordered_profiles)))]]
    updated_profile_overrides = dict(spec.profile_base_config_overrides)
    for row in ordered_profiles[: max(1, min(2, len(ordered_profiles)))]:
        profile = str(row.get("profile"))
        apogee = _safe_float(row.get("departure_apogee_km"))
        if apogee is None:
            continue
        current = dict(updated_profile_overrides.get(profile, {}))
        current["departure_apogee_km"] = apogee
        current["speed_scale"] = _safe_float(row.get("speed_scale")) or 1.0
        updated_profile_overrides[profile] = current
    updated_workflow_overrides = dict(spec.workflow_run_profile_overrides)
    top_workflow = str(leaderboard_rows[0].get("workflow_id"))
    workflow_overrides = dict(updated_workflow_overrides.get(top_workflow, {}))
    workflow_overrides["candidate_count"] = max(int(workflow_overrides.get("candidate_count", 12)), 18)
    workflow_overrides["truth_candidate_count"] = max(int(workflow_overrides.get("truth_candidate_count", spec.truth_budget)), spec.truth_budget + 1)
    updated_workflow_overrides[top_workflow] = workflow_overrides
    return replace(
        spec,
        campaign_id=f"{spec.campaign_id}-follow-on",
        parent_campaign_id=spec.campaign_id,
        profiles=tuple(selected_profiles),
        truth_budget=spec.truth_budget + 1,
        profile_base_config_overrides=updated_profile_overrides,
        workflow_run_profile_overrides=updated_workflow_overrides,
    )


def summarize_campaign(root: Path | None, campaign_id: str) -> dict[str, Any]:
    workspace_root = Path(".") if root is None else root
    leaderboard_rows = _leaderboard_rows(workspace_root, campaign_id)
    observations = _observations_for_campaign(workspace_root, campaign_id, leaderboard_rows)
    summary = {
        "campaign_id": campaign_id,
        "generated_at": _now_iso(),
        "leaderboard_count": len(leaderboard_rows),
        "observation_count": len(observations),
        "top_run_id": leaderboard_rows[0]["run_id"] if leaderboard_rows else None,
        "top_profile": leaderboard_rows[0]["profile"] if leaderboard_rows else None,
        "top_workflow_id": leaderboard_rows[0]["workflow_id"] if leaderboard_rows else None,
    }
    _write_json(_campaign_leaderboard_path(workspace_root, campaign_id), {"rows": leaderboard_rows})
    _write_csv(_campaign_leaderboard_csv_path(workspace_root, campaign_id), leaderboard_rows)
    _write_json(_campaign_observations_path(workspace_root, campaign_id), {"observations": observations})
    _write_json(_campaign_summary_path(workspace_root, campaign_id), summary)
    suggested = _suggest_follow_on_campaign(workspace_root, campaign_id, leaderboard_rows)
    if suggested is not None:
        _campaign_suggested_spec_path(workspace_root, campaign_id).write_text(campaign_spec_to_toml(suggested), encoding="utf-8")
    return load_campaign(workspace_root, campaign_id)


def load_campaign_leaderboard(root: Path | None, campaign_id: str) -> list[dict[str, Any]]:
    workspace_root = Path(".") if root is None else root
    payload = _read_json(_campaign_leaderboard_path(workspace_root, campaign_id))
    if not payload:
        summarize_campaign(workspace_root, campaign_id)
        payload = _read_json(_campaign_leaderboard_path(workspace_root, campaign_id))
    return list(payload.get("rows", []))


def load_campaign_observations(root: Path | None, campaign_id: str) -> list[dict[str, Any]]:
    workspace_root = Path(".") if root is None else root
    payload = _read_json(_campaign_observations_path(workspace_root, campaign_id))
    if not payload:
        summarize_campaign(workspace_root, campaign_id)
        payload = _read_json(_campaign_observations_path(workspace_root, campaign_id))
    return list(payload.get("observations", []))
