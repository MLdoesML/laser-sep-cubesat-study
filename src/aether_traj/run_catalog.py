from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


SCALAR_WORKFLOW_IDS = {"sep_jax", "sep_lbfgs", "sep_de", "sep_pso", "test_scalar", "test_scalar_alt"}


def _resolve_output_dir(workspace_root: Path, run_id: str) -> Path:
    direct = workspace_root / "outputs/runs" / run_id
    if (direct / "run_manifest.json").is_file():
        return direct
    matches = list((workspace_root / "outputs/runs").glob(f"**/{run_id}/run_manifest.json"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError(f"unknown run id: {run_id}")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_run(root: Path | None, run_id: str, *, include_internal: bool = False) -> dict[str, Any]:
    workspace_root = Path(".") if root is None else root
    output_dir = _resolve_output_dir(workspace_root, run_id)
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {}
    summary_filename = manifest.get("summary_filename")
    if summary_filename and (output_dir / str(summary_filename)).is_file():
        summary = json.loads((output_dir / str(summary_filename)).read_text(encoding="utf-8"))
    validation_rows = _read_csv_rows(output_dir / f"{manifest['workflow_id']}_validation_candidates.csv")
    candidates = []
    for rank, row in enumerate(validation_rows, start=1):
        candidate = {
            "candidate_id": row.get("candidate_id"),
            "rank": rank,
            "source": row.get("source"),
            "classification": row.get("classification"),
            "capture_duration_days": float(row.get("capture_duration_days", 0.0) or 0.0),
            "perilune_altitude_km": float(row.get("perilune_altitude_km", 0.0) or 0.0),
            "capture_apoapsis_km": float(row.get("capture_apoapsis_km", 0.0) or 0.0),
            "final_mass_kg": float(row.get("final_mass_kg", 0.0) or 0.0),
            "propellant_used_kg": float(row.get("propellant_used_kg", 0.0) or 0.0),
            "total_beam_energy_mj": float(row.get("total_beam_energy_mj", 0.0) or 0.0),
            "perigee_boost_dwell_hours": float(row.get("perigee_boost_dwell_hours", 0.0) or 0.0),
            "lunar_brake_dwell_hours": float(row.get("lunar_brake_dwell_hours", 0.0) or 0.0),
            "surrogate_objective": float(row.get("surrogate_objective", 0.0) or 0.0),
            "physical_constraint_violation": float(row.get("physical_constraint_violation", 0.0) or 0.0),
            "replay_available": False,
            "raw": dict(row),
        }
        candidates.append(candidate)
    payload = {
        **manifest,
        "summary": summary,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "replayable_candidate_count": 0,
        "managed": bool(manifest.get("managed")),
    }
    if include_internal:
        payload["_internal_candidates"] = candidates
    return payload


def load_runs(root: Path | None) -> list[dict[str, Any]]:
    workspace_root = Path(".") if root is None else root
    runs: list[dict[str, Any]] = []
    for manifest_path in sorted((workspace_root / "outputs/runs").glob("**/run_manifest.json")):
        try:
            payload = load_run(workspace_root, manifest_path.parent.name)
        except Exception:
            continue
        runs.append(payload)
    runs.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    return runs
