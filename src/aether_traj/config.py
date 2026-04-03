from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import tomllib

from aether_traj.ephemeris import DE440_HIACC_TABLE_ID
from aether_traj.models import (
    BeamArchitectureConfig,
    BeamStationFamilyConfig,
    LaunchLane,
    SepRunProfile,
    SimulationConfig,
    StudyProfile,
    SuccessCriteria,
    VehicleConfig,
)


DEFAULT_VEHICLE_CONFIG_PATH = Path("configs/vehicles/capstone_class_demo_v1.toml")
DEFAULT_CAMPAIGN_SPEC_PATH = Path("campaign.toml")
DEFAULT_MANAGER_PROGRAM_PATH = Path("manager_program.md")


def load_vehicle_config(path: Path | str = DEFAULT_VEHICLE_CONFIG_PATH) -> VehicleConfig:
    payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    mass = payload["mass"]
    propulsion = payload["propulsion"]
    power = payload["power"]
    return VehicleConfig(
        name=str(payload["name"]),
        wet_mass_kg=float(mass["wet_mass_kg"]),
        dry_mass_kg=float(mass["dry_mass_kg"]),
        usable_propellant_kg=float(mass["usable_propellant_kg"]),
        protected_reserve_kg=float(mass["protected_reserve_kg"]),
        isp_seconds=float(propulsion["isp_seconds"]),
        thruster_efficiency=float(propulsion["thruster_efficiency"]),
        onboard_solar_power_w=float(power["onboard_solar_power_w"]),
        beam_receiver_efficiency=float(power["beam_receiver_efficiency"]),
        propulsion_power_cap_w=float(power["propulsion_power_cap_w"]),
        receiver_area_m2=float(power.get("receiver_area_m2", 0.25)),
        thermal_power_cap_w=float(power.get("thermal_power_cap_w", power["propulsion_power_cap_w"])),
    )


def load_study_profile(path: Path | str) -> StudyProfile:
    payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    success = payload["success"]
    return StudyProfile(
        name=str(payload["name"]),
        description=str(payload["description"]),
        launch_lane=str(payload["launch_lane"]),
        beam_architecture=str(payload["beam_architecture"]),
        duration_days=float(payload["duration_days"]),
        success=SuccessCriteria(
            min_capture_duration_days=float(success["min_capture_duration_days"]),
            min_perilune_altitude_km=float(success["min_perilune_altitude_km"]),
            max_perilune_altitude_km=float(success["max_perilune_altitude_km"]),
            max_capture_apoapsis_km=float(success["max_capture_apoapsis_km"]),
            min_final_mass_kg=float(success["min_final_mass_kg"]),
        ),
    )


def default_launch_lanes() -> tuple[LaunchLane, ...]:
    return (
        LaunchLane(
            lane_id="gto_secondary",
            label="GTO secondary",
            insertion_proxy="GTO secondary deployment",
            rideshare_cost_musd=0.35,
            initial_perigee_altitude_km=300.0,
            initial_apogee_km=42_164.0,
            initial_inclination_deg=27.0,
        ),
    )


def launch_lane_for_id(lane_id: str) -> LaunchLane:
    for lane in default_launch_lanes():
        if lane.lane_id == lane_id:
            return lane
    raise ValueError(f"unknown launch lane: {lane_id}")


def beam_station_family(role: str, family: str) -> BeamStationFamilyConfig:
    normalized = family.lower()
    if role == "perigee_boost":
        if normalized == "geo":
            return BeamStationFamilyConfig(role, normalized, 80_000.0, 2.5, 60_000.0, 0.80, 35.0, 42_164.0, 1.0)
        if normalized == "high_earth_elliptical_relay":
            return BeamStationFamilyConfig(role, normalized, 120_000.0, 3.0, 120_000.0, 0.65, 40.0, 80_000.0, 2.0)
        if normalized == "supersynchronous_relay":
            return BeamStationFamilyConfig(role, normalized, 100_000.0, 2.8, 90_000.0, 0.75, 38.0, 60_000.0, 1.5)
    if role == "lunar_brake":
        if normalized == "eml1_halo":
            return BeamStationFamilyConfig(role, normalized, 60_000.0, 2.4, 110_000.0, 0.75, 45.0, 326_000.0, 14.0)
        if normalized == "nrho":
            return BeamStationFamilyConfig(role, normalized, 50_000.0, 2.0, 85_000.0, 0.70, 50.0, 70_000.0, 6.5)
        if normalized == "dro":
            return BeamStationFamilyConfig(role, normalized, 70_000.0, 2.6, 120_000.0, 0.80, 55.0, 90_000.0, 12.0)
    raise ValueError(f"unsupported station family: role={role}, family={family}")


def beam_architecture_for_mode(mode_id: str) -> BeamArchitectureConfig:
    normalized = mode_id.lower()
    if normalized == "none":
        return BeamArchitectureConfig(mode_id="none", description="SEP-only baseline with no external beam support.")
    if normalized == "perigee_boost_only":
        return BeamArchitectureConfig(
            mode_id=normalized,
            description="Earth-side beam support during repeated perigee passes.",
            perigee_boost_station=beam_station_family("perigee_boost", "geo"),
        )
    if normalized == "lunar_brake_only":
        return BeamArchitectureConfig(
            mode_id=normalized,
            description="Lunar-side beam support during approach and capture shaping.",
            lunar_brake_station=beam_station_family("lunar_brake", "nrho"),
        )
    if normalized in {"dual_window", "dual_window_fixed"}:
        return BeamArchitectureConfig(
            mode_id="dual_window_fixed",
            description="One Earth-side beam station and one lunar-side beam station with fixed reference placements.",
            perigee_boost_station=beam_station_family("perigee_boost", "geo"),
            lunar_brake_station=beam_station_family("lunar_brake", "nrho"),
        )
    raise ValueError(f"unsupported beam architecture mode: {mode_id}")


def _base_config(name: str, description: str, profile: StudyProfile, vehicle: VehicleConfig) -> SimulationConfig:
    launch_lane = launch_lane_for_id(profile.launch_lane)
    return SimulationConfig(
        name=name,
        description=description,
        duration_days=profile.duration_days,
        surrogate_step_seconds=1_800.0,
        truth_step_seconds=600.0,
        ephemeris_mode="cached_table",
        vehicle=vehicle,
        success=profile.success,
        launch_lane=launch_lane,
        beam_architecture=beam_architecture_for_mode(profile.beam_architecture),
        departure_apogee_km=max(launch_lane.initial_apogee_km, 420_000.0),
        departure_perigee_altitude_km=launch_lane.initial_perigee_altitude_km,
        departure_inclination_rad=math.radians(launch_lane.initial_inclination_deg),
        departure_raan_rad=0.0,
        departure_argument_of_latitude_rad=0.0,
        epoch_shift_days=0.0,
        departure_wait_days=0.0,
        speed_scale=1.0,
        flight_path_angle_rad=0.0,
        boosted_perigee_passes=2,
        sep_start_delay_days=0.5,
        ephemeris_table_id=DE440_HIACC_TABLE_ID,
        ephemeris_cache_step_seconds=600.0,
        earth_gravity_degree=4,
        moon_harmonics_degree=6,
        moon_harmonics_order=6,
        physical_integrator="dop853",
        physical_rtol=1.0e-10,
        physical_atol=1.0e-12,
    )


def build_sep_baseline_direct_capture_config(vehicle: VehicleConfig | None = None) -> SimulationConfig:
    vehicle = vehicle or load_vehicle_config()
    profile = load_study_profile(Path("configs/profiles/sep_baseline_direct_capture.toml"))
    return _base_config(profile.name, profile.description, profile, vehicle)


def build_laser_perigee_boost_config(vehicle: VehicleConfig | None = None) -> SimulationConfig:
    base = build_sep_baseline_direct_capture_config(vehicle)
    return replace(
        base,
        name="laser_perigee_boost",
        description="Perigee-boost-only beam support.",
        beam_architecture=beam_architecture_for_mode("perigee_boost_only"),
        boosted_perigee_passes=4,
        perigee_boost_power_scale=1.15,
    )


def build_laser_lunar_brake_config(vehicle: VehicleConfig | None = None) -> SimulationConfig:
    base = build_sep_baseline_direct_capture_config(vehicle)
    return replace(
        base,
        name="laser_lunar_brake",
        description="Lunar-brake-only beam support.",
        beam_architecture=beam_architecture_for_mode("lunar_brake_only"),
        lunar_brake_power_scale=1.15,
    )


def build_laser_dual_window_fixed_config(vehicle: VehicleConfig | None = None) -> SimulationConfig:
    vehicle = vehicle or load_vehicle_config()
    profile = load_study_profile(Path("configs/profiles/laser_dual_window_fixed.toml"))
    base = _base_config(profile.name, profile.description, profile, vehicle)
    return replace(
        base,
        beam_architecture=beam_architecture_for_mode("dual_window_fixed"),
        boosted_perigee_passes=4,
        perigee_boost_power_scale=1.20,
        lunar_brake_power_scale=1.15,
    )


def config_for_profile(profile: str) -> SimulationConfig:
    normalized = profile.lower()
    if normalized == "sep_baseline_direct_capture":
        return build_sep_baseline_direct_capture_config()
    if normalized == "laser_perigee_boost":
        return build_laser_perigee_boost_config()
    if normalized == "laser_lunar_brake":
        return build_laser_lunar_brake_config()
    if normalized == "laser_dual_window_fixed":
        return build_laser_dual_window_fixed_config()
    raise ValueError(f"unsupported profile: {profile}")


def build_sep_run_profile(profile: str) -> SepRunProfile:
    base = config_for_profile(profile)
    candidate_count = 12 if profile == "sep_baseline_direct_capture" else 16
    return SepRunProfile(
        name=profile,
        base_config=base,
        candidate_count=candidate_count,
        truth_candidate_count=4,
        random_seed=7,
        iterations=candidate_count,
    )
