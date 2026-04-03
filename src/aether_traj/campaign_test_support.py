from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import csv
import json
from pathlib import Path

from aether_traj.config import build_sep_baseline_direct_capture_config
from aether_traj.models import SepRunProfile


def build_fake_profile(profile: str) -> SepRunProfile:
    base = build_sep_baseline_direct_capture_config()
    if profile in {"baseline", "sep_baseline_direct_capture"}:
        config = base
    elif profile in {"beam_help", "laser_perigee_boost"}:
        config = replace(base, name="laser_perigee_boost", perigee_boost_power_scale=1.2)
    elif profile == "laser_lunar_brake":
        config = replace(base, name="laser_lunar_brake", lunar_brake_power_scale=1.15)
    elif profile == "laser_dual_window_fixed":
        config = replace(base, name="laser_dual_window_fixed", perigee_boost_power_scale=1.2, lunar_brake_power_scale=1.15)
    else:
        raise ValueError(f"unsupported fake profile: {profile}")
    return SepRunProfile(
        name=profile,
        base_config=config,
        candidate_count=6,
        truth_candidate_count=2,
        random_seed=11,
        iterations=6,
    )


def _fake_validation_payload(workflow_id: str, profile: str, apogee_km: float) -> tuple[dict[str, object], dict[str, object]]:
    direct_capture = apogee_km >= 400_000.0
    capture_like = apogee_km >= 360_000.0
    classification = "direct_capture" if direct_capture else ("capture_like" if capture_like else "flyby")
    capture_duration_days = 3.6 if direct_capture else (0.8 if capture_like else 0.0)
    perilune_altitude_km = 180.0 if direct_capture else (650.0 if capture_like else 4_500.0)
    capture_apoapsis_km = 18_000.0 if direct_capture else (52_000.0 if capture_like else 0.0)
    final_mass_kg = 21.7 if direct_capture else (22.0 if capture_like else 22.4)
    total_beam_energy_mj = 18.0 if direct_capture else (5.0 if capture_like else 0.0)
    perigee_dwell = 1.2 if direct_capture else 0.4
    lunar_dwell = 0.9 if direct_capture else 0.0
    surrogate_objective = -35.0 if direct_capture else (-22.0 if capture_like else -18.0)

    summary = {
        "workflow_id": workflow_id,
        "profile": profile,
        "best_objective": surrogate_objective,
        "selected_validation": {
            "candidate_id": "candidate-01",
            "classification": classification,
            "capture_duration_days": capture_duration_days,
            "perilune_altitude_km": perilune_altitude_km,
            "capture_apoapsis_km": capture_apoapsis_km,
            "final_mass_kg": final_mass_kg,
            "propellant_used_kg": 3.1,
            "total_beam_energy_mj": total_beam_energy_mj,
            "perigee_boost_dwell_hours": perigee_dwell,
            "lunar_brake_dwell_hours": lunar_dwell,
            "surrogate_objective": surrogate_objective,
            "physical_constraint_violation": 0.0 if direct_capture else 2.4,
            "physics_priority_score": 12.0 if direct_capture else (5.5 if capture_like else 1.0),
            "departure_apogee_km": apogee_km,
            "departure_perigee_altitude_km": 300.0,
            "epoch_shift_days": 0.0,
            "departure_wait_days": 0.0,
            "speed_scale": 1.0,
            "boosted_perigee_passes": 2 if direct_capture else 1,
            "perigee_boost_power_scale": 1.0,
            "lunar_brake_power_scale": 1.0,
        },
        "validation_summaries": [],
        "design_space_id": "fake_design_v1",
        "physics_model_id": "fake_physics_v1",
        "objective_set_id": "direct_capture_first",
    }
    candidate = dict(summary["selected_validation"])
    candidate.update(
        {
            "physical_rank": 1,
            "candidate_index": 1,
            "source": f"{workflow_id}_best",
        }
    )
    return summary, candidate


def _write_scalar_bundle(output_dir: Path, workflow_id: str, profile: str, summary_filename: str) -> None:
    run_profile = build_fake_profile(profile)
    apogee = float(run_profile.base_config.departure_apogee_km)
    summary, candidate = _fake_validation_payload(workflow_id, profile, apogee)
    created_at = datetime.now().astimezone().isoformat()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / summary_filename).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with (output_dir / f"{workflow_id}_validation_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate.keys()))
        writer.writeheader()
        writer.writerow(candidate)

    with (output_dir / f"{workflow_id}_optimization_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "objective"])
        writer.writeheader()
        writer.writerow({"iteration": 0, "objective": -8.0})
        writer.writerow({"iteration": run_profile.iterations, "objective": summary["best_objective"]})

    manifest = {
        "schema_version": 1,
        "run_id": output_dir.name,
        "workflow_id": workflow_id,
        "workflow_kind": "scalar",
        "profile": profile,
        "script_name": f"aether-{workflow_id.replace('_', '-')}",
        "summary_filename": summary_filename,
        "output_dir": str(output_dir),
        "created_at": created_at,
        "status": "completed",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_fake_workflow(profile: str, output_dir: Path, show_progress: bool | None = None) -> None:
    del show_progress
    _write_scalar_bundle(Path(output_dir), "test_scalar", profile, "test_scalar_summary.json")


def run_fake_alt_workflow(profile: str, output_dir: Path, show_progress: bool | None = None) -> None:
    del show_progress
    _write_scalar_bundle(Path(output_dir), "test_scalar_alt", profile, "test_scalar_alt_summary.json")
