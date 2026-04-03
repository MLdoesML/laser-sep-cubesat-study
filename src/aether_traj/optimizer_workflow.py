from __future__ import annotations

import csv
from pathlib import Path

from aether_traj.config import build_sep_run_profile
from aether_traj.dynamics import build_truth_validation_config, physical_constraint_violation, physics_priority_score, propagate
from aether_traj.experiments import write_json_file, write_run_manifest
from aether_traj.jax_surrogate import build_jax_problem, config_from_unit
from aether_traj.optimizers import optimize_with_adam, optimize_with_de, optimize_with_lbfgs, optimize_with_pso


def _validation_row(rank: int, candidate, truth_result, surrogate_objective: float) -> dict[str, object]:
    config = truth_result.config
    return {
        "physical_rank": rank,
        "candidate_id": candidate.candidate_id,
        "candidate_index": rank,
        "source": candidate.source,
        "classification": truth_result.classification,
        "capture_duration_days": truth_result.stable_capture_duration_days,
        "perilune_altitude_km": truth_result.perilune_altitude_km,
        "capture_apoapsis_km": truth_result.capture_apoapsis_km,
        "final_mass_kg": truth_result.final_mass_kg,
        "propellant_used_kg": truth_result.propellant_used_kg,
        "total_beam_energy_mj": truth_result.total_beam_energy_mj,
        "perigee_boost_dwell_hours": truth_result.per_station_dwell_hours.get("perigee_boost", 0.0),
        "lunar_brake_dwell_hours": truth_result.per_station_dwell_hours.get("lunar_brake", 0.0),
        "surrogate_objective": surrogate_objective,
        "physical_constraint_violation": physical_constraint_violation(truth_result, config),
        "physics_priority_score": physics_priority_score(truth_result, config),
        "departure_apogee_km": config.departure_apogee_km,
        "departure_perigee_altitude_km": config.departure_perigee_altitude_km,
        "epoch_shift_days": config.epoch_shift_days,
        "departure_wait_days": config.departure_wait_days,
        "speed_scale": config.speed_scale,
        "boosted_perigee_passes": config.boosted_perigee_passes,
        "perigee_boost_power_scale": config.perigee_boost_power_scale,
        "lunar_brake_power_scale": config.lunar_brake_power_scale,
        "ephemeris_mode": config.ephemeris_mode,
        "earth_gravity_degree": config.earth_gravity_degree,
        "moon_harmonics_degree": config.moon_harmonics_degree,
        "moon_harmonics_order": config.moon_harmonics_order,
        "physical_rtol": config.physical_rtol,
        "physical_atol": config.physical_atol,
    }


def run_scalar_optimizer_workflow(profile: str, output_dir: Path, workflow_id: str, show_progress: bool | None = None) -> None:
    del show_progress
    run_profile = build_sep_run_profile(profile)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem = build_jax_problem(run_profile.base_config)

    if workflow_id == "sep_jax":
        result = optimize_with_adam(problem, workflow_id, iterations=run_profile.iterations, seed_count=run_profile.candidate_count, random_seed=run_profile.random_seed)
    elif workflow_id == "sep_lbfgs":
        result = optimize_with_lbfgs(problem, workflow_id, iterations=run_profile.iterations, seed_count=run_profile.candidate_count, random_seed=run_profile.random_seed)
    elif workflow_id == "sep_de":
        result = optimize_with_de(problem, workflow_id, generations=run_profile.iterations, population_size=run_profile.candidate_count, random_seed=run_profile.random_seed)
    elif workflow_id == "sep_pso":
        result = optimize_with_pso(problem, workflow_id, iterations=run_profile.iterations, swarm_size=run_profile.candidate_count, random_seed=run_profile.random_seed)
    else:
        raise ValueError(f"unsupported optimizer workflow: {workflow_id}")

    search_rows = []
    for candidate in result.candidates:
        row = {
            "candidate_id": candidate.candidate_id,
            "source": candidate.source,
            "objective": candidate.objective,
            "classification": candidate.metrics["classification"],
            "capture_duration_days": candidate.metrics["stable_capture_duration_days"],
            "perilune_altitude_km": candidate.metrics["perilune_altitude_km"],
            "capture_apoapsis_km": candidate.metrics["capture_apoapsis_km"],
            "final_mass_kg": candidate.metrics["final_mass_kg"],
            "propellant_used_kg": candidate.metrics["propellant_used_kg"],
            "total_beam_energy_mj": candidate.metrics["total_beam_energy_mj"],
        }
        row.update(candidate.design)
        search_rows.append(row)

    validation_rows: list[dict[str, object]] = []
    selected_validation: dict[str, object] | None = None
    selected_score = float("-inf")
    for rank, candidate in enumerate(result.candidates[: run_profile.truth_candidate_count], start=1):
        truth_config = build_truth_validation_config(
            config_from_unit(run_profile.base_config, problem.search_space, candidate.unit_vector)
        )
        truth_result = propagate(config=truth_config, label=f"{workflow_id}-truth-{rank:02d}", truth=True, store_stride=4)
        row = _validation_row(rank, candidate, truth_result, candidate.objective)
        validation_rows.append(row)
        score = float(row["physics_priority_score"])
        if score > selected_score:
            selected_score = score
            selected_validation = dict(row)

    search_space = {
        "variables": [
            {
                "name": variable.name,
                "lower": variable.lower,
                "upper": variable.upper,
                "default": variable.default,
                "unit": variable.unit,
                "description": variable.description,
            }
            for variable in problem.search_space.variables
        ]
    }
    write_json_file(output_dir / f"{workflow_id}_search_space.json", search_space)
    write_json_file(output_dir / f"{workflow_id}_grid_search.json", {"rows": search_rows})

    with (output_dir / f"{workflow_id}_optimization_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.history[0].keys()) if result.history else ["iteration", "objective", "best_objective"])
        writer.writeheader()
        if result.history:
            writer.writerows(result.history)

    with (output_dir / f"{workflow_id}_validation_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation_rows[0].keys()) if validation_rows else ["physical_rank"])
        writer.writeheader()
        if validation_rows:
            writer.writerows(validation_rows)

    summary = {
        "workflow_id": workflow_id,
        "profile": profile,
        "backend": result.backend,
        "best_objective": result.best_candidate.objective,
        "candidate_count": len(result.candidates),
        "selected_validation": selected_validation,
        "validation_summaries": validation_rows,
        "duration_days": run_profile.base_config.duration_days,
        "beam_architecture": run_profile.base_config.beam_architecture.mode_id,
        "design_space_id": f"{profile}_v1",
        "physics_model_id": "spice_dop853_sep_truth_v2",
        "objective_set_id": "direct_capture_first",
        "surrogate_model_id": "jax_diff_sep_v1",
        "best_design": result.best_candidate.design,
    }
    write_json_file(output_dir / f"{workflow_id}_summary.json", summary)
    write_run_manifest(
        output_dir,
        workflow_id=workflow_id,
        profile=profile,
        summary_filename=f"{workflow_id}_summary.json",
        extra={
            "design_space_id": summary["design_space_id"],
            "physics_model_id": summary["physics_model_id"],
            "objective_set_id": summary["objective_set_id"],
            "surrogate_model_id": summary["surrogate_model_id"],
        },
    )
