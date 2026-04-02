from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleConfig:
    name: str
    wet_mass_kg: float
    dry_mass_kg: float
    usable_propellant_kg: float
    protected_reserve_kg: float
    isp_seconds: float
    thruster_efficiency: float
    onboard_solar_power_w: float
    beam_receiver_efficiency: float
    propulsion_power_cap_w: float


@dataclass(frozen=True)
class SuccessCriteria:
    min_capture_duration_days: float
    min_perilune_altitude_km: float
    max_perilune_altitude_km: float
    max_capture_apoapsis_km: float
    min_final_mass_kg: float


@dataclass(frozen=True)
class StudyProfile:
    name: str
    description: str
    launch_lane: str
    beam_architecture: str
    duration_days: float
    success: SuccessCriteria
