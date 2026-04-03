from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


SECONDS_PER_DAY = 86_400.0
EARTH_MU = 398_600.4418
MOON_MU = 4_902.8695
SUN_MU = 132_712_440_018.0
EARTH_RADIUS_KM = 6_378.1363
MOON_RADIUS_KM = 1_737.4
G0_MPS2 = 9.80665


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scale: float) -> "Vec3":
        return Vec3(self.x * scale, self.y * scale, self.z * scale)

    __rmul__ = __mul__

    def __truediv__(self, scale: float) -> "Vec3":
        return Vec3(self.x / scale, self.y / scale, self.z / scale)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self) -> float:
        return math.sqrt(max(0.0, self.dot(self)))

    def unit(self) -> "Vec3":
        magnitude = self.norm()
        if magnitude <= 0.0:
            return Vec3(0.0, 0.0, 0.0)
        return self / magnitude

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class State:
    position: Vec3
    velocity: Vec3
    mass_kg: float


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
    receiver_area_m2: float = 0.25
    thermal_power_cap_w: float = 700.0

    @property
    def minimum_final_mass_kg(self) -> float:
        return self.dry_mass_kg + self.protected_reserve_kg


@dataclass(frozen=True)
class SuccessCriteria:
    min_capture_duration_days: float
    min_perilune_altitude_km: float
    max_perilune_altitude_km: float
    max_capture_apoapsis_km: float
    min_final_mass_kg: float


@dataclass(frozen=True)
class BeamStationFamilyConfig:
    role: str
    family: str
    beam_power_w: float
    aperture_m: float
    useful_range_km: float
    duty_cycle: float
    allowed_pointing_half_angle_deg: float
    orbit_radius_km: float
    orbit_period_days: float
    phase_deg: float = 0.0
    minimum_dwell_seconds: float = 300.0


@dataclass(frozen=True)
class BeamArchitectureConfig:
    mode_id: str
    description: str
    perigee_boost_station: BeamStationFamilyConfig | None = None
    lunar_brake_station: BeamStationFamilyConfig | None = None

    def enabled_roles(self) -> tuple[str, ...]:
        roles: list[str] = []
        if self.perigee_boost_station is not None:
            roles.append("perigee_boost")
        if self.lunar_brake_station is not None:
            roles.append("lunar_brake")
        return tuple(roles)


@dataclass(frozen=True)
class LaunchLane:
    lane_id: str
    label: str
    insertion_proxy: str
    rideshare_cost_musd: float
    initial_perigee_altitude_km: float
    initial_apogee_km: float
    initial_inclination_deg: float


@dataclass(frozen=True)
class SimulationConfig:
    name: str
    description: str
    duration_days: float
    surrogate_step_seconds: float
    truth_step_seconds: float
    ephemeris_mode: str
    vehicle: VehicleConfig
    success: SuccessCriteria
    launch_lane: LaunchLane
    beam_architecture: BeamArchitectureConfig
    departure_apogee_km: float
    departure_perigee_altitude_km: float
    departure_inclination_rad: float
    departure_raan_rad: float
    departure_argument_of_latitude_rad: float
    epoch_shift_days: float
    departure_wait_days: float
    speed_scale: float
    flight_path_angle_rad: float
    boosted_perigee_passes: int
    sep_start_delay_days: float
    throttle_nodes: tuple[float, float, float, float] = (0.55, 0.65, 0.75, 0.70)
    azimuth_nodes_rad: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    elevation_nodes_rad: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    perigee_boost_power_scale: float = 1.0
    lunar_brake_power_scale: float = 1.0
    beam_wavelength_m: float = 1.064e-6
    beam_quality_factor: float = 1.2
    ephemeris_table_id: str = "de440_hiacc_v2_2026a"
    ephemeris_cache_step_seconds: float = 600.0
    earth_j2_enabled: bool = True
    earth_gravity_degree: int = 4
    moon_harmonics_degree: int = 6
    moon_harmonics_order: int = 6
    moon_harmonics_blend_start_altitude_km: float = 25_000.0
    moon_harmonics_blend_full_altitude_km: float = 10_000.0
    physical_integrator: str = "dop853"
    physical_rtol: float = 1.0e-10
    physical_atol: float = 1.0e-12
    max_distance_km: float = 1_500_000.0

    def total_steps(self, truth: bool = False) -> int:
        dt = self.truth_step_seconds if truth else self.surrogate_step_seconds
        return max(1, int(round(self.duration_days * SECONDS_PER_DAY / dt)))


@dataclass(frozen=True)
class StudyProfile:
    name: str
    description: str
    launch_lane: str
    beam_architecture: str
    duration_days: float
    success: SuccessCriteria


@dataclass(frozen=True)
class SearchVariable:
    name: str
    lower: float
    upper: float
    default: float
    unit: str
    description: str


@dataclass(frozen=True)
class SepRunProfile:
    name: str
    base_config: SimulationConfig
    candidate_count: int
    truth_candidate_count: int
    random_seed: int
    iterations: int


@dataclass
class SimulationResult:
    label: str
    config: SimulationConfig
    classification: str
    stable_capture_duration_days: float
    perilune_altitude_km: float
    capture_apoapsis_km: float | None
    final_mass_kg: float
    propellant_used_kg: float
    total_beam_energy_mj: float
    per_station_dwell_hours: dict[str, float]
    min_moon_distance_km: float
    max_earth_distance_km: float
    final_moon_specific_energy_km2s2: float
    escaped: bool
    unsafe_perilune: bool
    path: list[tuple[float, float, float, float, float, float, float, float]] = field(default_factory=list)

    def as_summary(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "capture_duration_days": self.stable_capture_duration_days,
            "perilune_altitude_km": self.perilune_altitude_km,
            "capture_apoapsis_km": self.capture_apoapsis_km,
            "final_mass_kg": self.final_mass_kg,
            "propellant_used_kg": self.propellant_used_kg,
            "total_beam_energy_mj": self.total_beam_energy_mj,
            "per_station_dwell_hours": dict(self.per_station_dwell_hours),
            "min_moon_distance_km": self.min_moon_distance_km,
            "max_earth_distance_km": self.max_earth_distance_km,
            "final_moon_specific_energy_km2s2": self.final_moon_specific_energy_km2s2,
            "escaped": self.escaped,
            "unsafe_perilune": self.unsafe_perilune,
        }


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_id: str
    kind: str
    runner_target: str
    profile_check_target: str
    profiles: tuple[str, ...]
    scripts: dict[str, str]


def dataclass_to_dict(instance: Any) -> dict[str, Any]:
    return asdict(instance)
