from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from aether_traj.ephemeris import AU_KM, ECLIPTIC_OBLIQUITY_RAD, REFERENCE_JULIAN_DAY
from aether_traj.models import (
    EARTH_MU,
    EARTH_RADIUS_KM,
    G0_MPS2,
    MOON_MU,
    MOON_RADIUS_KM,
    SECONDS_PER_DAY,
    SUN_MU,
    SearchVariable,
    SimulationConfig,
)

SUN_RADIUS_KM = 695_700.0


def search_variables_for_config(base: SimulationConfig) -> tuple[SearchVariable, ...]:
    return (
        SearchVariable("departure_apogee_km", max(380_000.0, base.departure_apogee_km * 0.9), base.departure_apogee_km * 1.25, base.departure_apogee_km, "km", "Earth-centered departure apogee."),
        SearchVariable("epoch_shift_days", -12.0, 12.0, base.epoch_shift_days, "days", "Global epoch shift."),
        SearchVariable("departure_wait_days", 0.0, 20.0, base.departure_wait_days, "days", "Departure wait at staging orbit."),
        SearchVariable("speed_scale", 0.92, 1.12, base.speed_scale, "ratio", "Initial departure speed scale."),
        SearchVariable("boosted_perigee_passes", 1.0, 5.0, float(base.boosted_perigee_passes), "count", "Number of perigee boost windows."),
        SearchVariable("perigee_boost_power_scale", 0.8, 1.3, base.perigee_boost_power_scale, "ratio", "Earth-side beam power scale."),
        SearchVariable("lunar_brake_power_scale", 0.8, 1.3, base.lunar_brake_power_scale, "ratio", "Lunar-side beam power scale."),
    )


@dataclass(frozen=True)
class JaxSearchSpace:
    variables: tuple[SearchVariable, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    default_unit: np.ndarray

    @property
    def dimension(self) -> int:
        return len(self.variables)


@dataclass(frozen=True)
class JaxSurrogateProblem:
    base_config: SimulationConfig
    search_space: JaxSearchSpace
    metrics_fn: Callable[[jnp.ndarray], dict[str, jnp.ndarray]]
    objective_fn: Callable[[jnp.ndarray], jnp.ndarray]
    value_and_grad_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
    batched_metrics_fn: Callable[[jnp.ndarray], dict[str, jnp.ndarray]]
    batched_objective_fn: Callable[[jnp.ndarray], jnp.ndarray]


def build_search_space(base: SimulationConfig) -> JaxSearchSpace:
    variables = search_variables_for_config(base)
    lower = np.array([variable.lower for variable in variables], dtype=np.float64)
    upper = np.array([variable.upper for variable in variables], dtype=np.float64)
    default = np.array([variable.default for variable in variables], dtype=np.float64)
    default_unit = np.clip((default - lower) / np.maximum(upper - lower, 1.0e-12), 0.0, 1.0)
    return JaxSearchSpace(variables=variables, lower_bounds=lower, upper_bounds=upper, default_unit=default_unit)


def design_dict_from_unit(search_space: JaxSearchSpace, unit_vector: np.ndarray | jnp.ndarray) -> dict[str, float]:
    unit = np.clip(np.asarray(unit_vector, dtype=np.float64), 0.0, 1.0)
    design_values = search_space.lower_bounds + unit * (search_space.upper_bounds - search_space.lower_bounds)
    payload: dict[str, float] = {}
    for variable, value in zip(search_space.variables, design_values):
        payload[variable.name] = int(round(float(value))) if variable.name == "boosted_perigee_passes" else float(value)
    return payload


def config_from_unit(base: SimulationConfig, search_space: JaxSearchSpace, unit_vector: np.ndarray | jnp.ndarray) -> SimulationConfig:
    from dataclasses import replace

    design = design_dict_from_unit(search_space, unit_vector)
    return replace(
        base,
        departure_apogee_km=design["departure_apogee_km"],
        epoch_shift_days=design["epoch_shift_days"],
        departure_wait_days=design["departure_wait_days"],
        speed_scale=design["speed_scale"],
        boosted_perigee_passes=int(design["boosted_perigee_passes"]),
        perigee_boost_power_scale=design["perigee_boost_power_scale"],
        lunar_brake_power_scale=design["lunar_brake_power_scale"],
    )


def metrics_dict_from_unit(problem: JaxSurrogateProblem, unit_vector: np.ndarray | jnp.ndarray) -> dict[str, float]:
    payload = problem.metrics_fn(jnp.asarray(unit_vector, dtype=jnp.float64))
    return {key: float(value) for key, value in payload.items()}


def classification_from_metrics(metrics: dict[str, float], config: SimulationConfig) -> str:
    if metrics["perilune_altitude_km"] < 0.0:
        return "unsafe_perilune"
    if (
        metrics["stable_capture_duration_days"] >= config.success.min_capture_duration_days
        and config.success.min_perilune_altitude_km <= metrics["perilune_altitude_km"] <= config.success.max_perilune_altitude_km
        and metrics["capture_apoapsis_km"] <= config.success.max_capture_apoapsis_km
        and metrics["final_mass_kg"] >= config.success.min_final_mass_kg
    ):
        return "direct_capture"
    if metrics["stable_capture_duration_days"] > 0.1:
        return "capture_like"
    if metrics["final_moon_specific_energy_km2s2"] > 0.0:
        return "escape"
    return "flyby"


def build_jax_problem(base: SimulationConfig) -> JaxSurrogateProblem:
    search_space = build_search_space(base)
    lower = jnp.asarray(search_space.lower_bounds, dtype=jnp.float64)
    upper = jnp.asarray(search_space.upper_bounds, dtype=jnp.float64)
    throttle_nodes = jnp.asarray(base.throttle_nodes, dtype=jnp.float64)
    perigee_station = base.beam_architecture.perigee_boost_station
    lunar_station = base.beam_architecture.lunar_brake_station
    dt = base.surrogate_step_seconds
    steps = base.total_steps(truth=False)
    mission_duration_seconds = base.duration_days * SECONDS_PER_DAY
    perigee_radius = EARTH_RADIUS_KM + base.departure_perigee_altitude_km
    capture_radius_limit_km = min(120_000.0, base.success.max_capture_apoapsis_km + MOON_RADIUS_KM + 50_000.0)

    def station_constants(station) -> tuple[float, float, float, float, float, float, float, float]:
        if station is None:
            return (0.0, 0.0, 1.0, 1.0, 0.0, math.pi, 0.0, 1.0)
        return (
            1.0,
            station.beam_power_w,
            station.aperture_m,
            station.useful_range_km,
            station.duty_cycle,
            math.radians(station.allowed_pointing_half_angle_deg),
            station.orbit_radius_km,
            station.orbit_period_days,
        )

    perigee_constants = station_constants(perigee_station)
    lunar_constants = station_constants(lunar_station)

    def _safe_norm(vector: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(jnp.maximum(jnp.dot(vector, vector), 1.0e-18))

    def _unit(vector: jnp.ndarray) -> jnp.ndarray:
        return vector / _safe_norm(vector)

    def _rotate_z(vector: jnp.ndarray, angle_rad: float) -> jnp.ndarray:
        cosine = jnp.cos(angle_rad)
        sine = jnp.sin(angle_rad)
        return jnp.asarray((cosine * vector[0] - sine * vector[1], sine * vector[0] + cosine * vector[1], vector[2]), dtype=jnp.float64)

    def _rotate_x(vector: jnp.ndarray, angle_rad: float) -> jnp.ndarray:
        cosine = jnp.cos(angle_rad)
        sine = jnp.sin(angle_rad)
        return jnp.asarray((vector[0], cosine * vector[1] - sine * vector[2], sine * vector[1] + cosine * vector[2]), dtype=jnp.float64)

    def _rotate_orbit(vector: jnp.ndarray) -> jnp.ndarray:
        rotated = _rotate_z(vector, base.departure_argument_of_latitude_rad)
        rotated = _rotate_x(rotated, base.departure_inclination_rad)
        return _rotate_z(rotated, base.departure_raan_rad)

    def _seconds_to_days_since_j2000(seconds_from_reference: jnp.ndarray) -> jnp.ndarray:
        return (REFERENCE_JULIAN_DAY - 2_451_545.0) + seconds_from_reference / SECONDS_PER_DAY

    def _rotate_ecliptic_to_equatorial(vector: jnp.ndarray) -> jnp.ndarray:
        cosine = jnp.cos(ECLIPTIC_OBLIQUITY_RAD)
        sine = jnp.sin(ECLIPTIC_OBLIQUITY_RAD)
        return jnp.asarray((vector[0], cosine * vector[1] - sine * vector[2], sine * vector[1] + cosine * vector[2]), dtype=jnp.float64)

    def _sun_position(days_since_j2000: jnp.ndarray) -> jnp.ndarray:
        mean_anomaly = jnp.deg2rad(jnp.mod(357.52911 + 0.98560028 * days_since_j2000, 360.0))
        mean_longitude = jnp.deg2rad(jnp.mod(280.46646 + 0.98564736 * days_since_j2000, 360.0))
        ecliptic_longitude = mean_longitude + jnp.deg2rad(
            1.914602 * jnp.sin(mean_anomaly)
            + 0.019993 * jnp.sin(2.0 * mean_anomaly)
            + 0.000289 * jnp.sin(3.0 * mean_anomaly)
        )
        radius_au = 1.000140612 - 0.016708617 * jnp.cos(mean_anomaly) - 0.000139589 * jnp.cos(2.0 * mean_anomaly)
        ecliptic = jnp.asarray((radius_au * AU_KM * jnp.cos(ecliptic_longitude), radius_au * AU_KM * jnp.sin(ecliptic_longitude), 0.0), dtype=jnp.float64)
        return _rotate_ecliptic_to_equatorial(ecliptic)

    def _moon_position(days_since_j2000: jnp.ndarray) -> jnp.ndarray:
        l0 = jnp.deg2rad(jnp.mod(218.316 + 13.176396 * days_since_j2000, 360.0))
        m_moon = jnp.deg2rad(jnp.mod(134.963 + 13.064993 * days_since_j2000, 360.0))
        d = jnp.deg2rad(jnp.mod(297.850 + 12.190749 * days_since_j2000, 360.0))
        f = jnp.deg2rad(jnp.mod(93.272 + 13.229350 * days_since_j2000, 360.0))
        longitude = l0 + jnp.deg2rad(
            6.289 * jnp.sin(m_moon)
            + 1.274 * jnp.sin(2.0 * d - m_moon)
            + 0.658 * jnp.sin(2.0 * d)
            + 0.214 * jnp.sin(2.0 * m_moon)
            + 0.110 * jnp.sin(d)
        )
        latitude = jnp.deg2rad(
            5.128 * jnp.sin(f)
            + 0.280 * jnp.sin(m_moon + f)
            + 0.277 * jnp.sin(m_moon - f)
            + 0.173 * jnp.sin(2.0 * d - f)
        )
        radius_km = 385_000.56 - 20_905.0 * jnp.cos(m_moon) - 3_699.0 * jnp.cos(2.0 * d - m_moon) - 2_956.0 * jnp.cos(2.0 * d)
        ecliptic = jnp.asarray(
            (
                radius_km * jnp.cos(latitude) * jnp.cos(longitude),
                radius_km * jnp.cos(latitude) * jnp.sin(longitude),
                radius_km * jnp.sin(latitude),
            ),
            dtype=jnp.float64,
        )
        return _rotate_ecliptic_to_equatorial(ecliptic)

    def _position_and_velocity(position_fn: Callable[[jnp.ndarray], jnp.ndarray], days_since_j2000: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        delta_seconds = 300.0
        delta_days = delta_seconds / SECONDS_PER_DAY
        position = position_fn(days_since_j2000)
        previous = position_fn(days_since_j2000 - delta_days)
        next_position = position_fn(days_since_j2000 + delta_days)
        velocity = (next_position - previous) * (0.5 / delta_seconds)
        return position, velocity

    def _moon_state(seconds_from_reference: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _position_and_velocity(_moon_position, _seconds_to_days_since_j2000(seconds_from_reference))

    def _sun_state(seconds_from_reference: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _position_and_velocity(_sun_position, _seconds_to_days_since_j2000(seconds_from_reference))

    def _earth_acceleration(position: jnp.ndarray) -> jnp.ndarray:
        radius = _safe_norm(position)
        return position * (-EARTH_MU / (radius**3))

    def _earth_j2_acceleration(position: jnp.ndarray) -> jnp.ndarray:
        if not base.earth_j2_enabled:
            return jnp.zeros(3, dtype=jnp.float64)
        x, y, z = position
        r2 = jnp.maximum(x * x + y * y + z * z, 1.0e-18)
        r = jnp.sqrt(r2)
        factor = 1.5 * 1.08262668e-3 * EARTH_MU * (EARTH_RADIUS_KM**2) / (r**5)
        z2_r2 = (z * z) / r2
        return jnp.asarray(
            (
                factor * x * (5.0 * z2_r2 - 1.0),
                factor * y * (5.0 * z2_r2 - 1.0),
                factor * z * (5.0 * z2_r2 - 3.0),
            ),
            dtype=jnp.float64,
        )

    def _third_body_acceleration(position: jnp.ndarray, source_position: jnp.ndarray, mu: float) -> jnp.ndarray:
        delta = position - source_position
        delta_norm = _safe_norm(delta)
        source_norm = _safe_norm(source_position)
        return delta * (-mu / (delta_norm**3)) - source_position * (mu / (source_norm**3))

    def _angular_radius(radius_km: float, distance_km: jnp.ndarray) -> jnp.ndarray:
        return jnp.arcsin(jnp.clip(radius_km / jnp.maximum(distance_km, 1.0e-6), 0.0, 1.0))

    def _soft_eclipse_fraction(sun_vector_km: jnp.ndarray, occulting_vector_km: jnp.ndarray, occulting_radius_km: float) -> jnp.ndarray:
        sun_distance = _safe_norm(sun_vector_km)
        occulting_distance = _safe_norm(occulting_vector_km)
        sun_radius = _angular_radius(SUN_RADIUS_KM, sun_distance)
        occ_radius = _angular_radius(occulting_radius_km, occulting_distance)
        cosine = jnp.clip(jnp.dot(sun_vector_km, occulting_vector_km) / (sun_distance * occulting_distance), -1.0, 1.0)
        separation = jnp.arccos(cosine)
        obscuration = jax.nn.sigmoid(((sun_radius + occ_radius) - separation) / 0.01)
        depth = jnp.clip((occ_radius / jnp.maximum(sun_radius, 1.0e-6)) ** 2, 0.0, 1.0)
        return 1.0 - obscuration * depth

    def _station_position(seconds: jnp.ndarray, moon_position: jnp.ndarray, constants: tuple[float, float, float, float, float, float, float, float], role: str) -> jnp.ndarray:
        _, _, _, _, _, _, orbit_radius_km, orbit_period_days = constants
        phase = 2.0 * math.pi * (seconds / SECONDS_PER_DAY) / jnp.maximum(orbit_period_days, 1.0e-6)
        if role == "perigee_boost":
            return jnp.asarray((orbit_radius_km * jnp.cos(phase), orbit_radius_km * jnp.sin(phase), 0.0), dtype=jnp.float64)
        radial = _unit(moon_position)
        tangential = _unit(jnp.asarray((-radial[1], radial[0], 0.0), dtype=jnp.float64))
        tangential = jnp.where(_safe_norm(tangential) > 1.0e-9, tangential, jnp.asarray((0.0, 1.0, 0.0), dtype=jnp.float64))
        normal = _unit(jnp.cross(radial, tangential))
        return moon_position + tangential * (orbit_radius_km * jnp.cos(phase)) + normal * (0.3 * orbit_radius_km * jnp.sin(phase))

    def _beam_gate(spacecraft_position: jnp.ndarray, station_position_km: jnp.ndarray, moon_position: jnp.ndarray, constants: tuple[float, float, float, float, float, float, float, float], role: str) -> jnp.ndarray:
        enabled, _, _, useful_range_km, duty_cycle, allowed_pointing_half_angle_rad, _, _ = constants
        line = spacecraft_position - station_position_km
        line_distance = _safe_norm(line)
        target_origin = jnp.zeros(3, dtype=jnp.float64) if role == "perigee_boost" else moon_position
        reference = _unit(target_origin - station_position_km)
        cosine = jnp.clip(jnp.dot(reference, _unit(line)), -1.0, 1.0)
        angle = jnp.arccos(cosine)
        range_gate = jax.nn.sigmoid((useful_range_km - line_distance) / 4_000.0)
        pointing_gate = jax.nn.sigmoid((allowed_pointing_half_angle_rad - angle) / 0.05)
        return enabled * duty_cycle * range_gate * pointing_gate

    def _received_beam_power(spacecraft_position: jnp.ndarray, absolute_seconds: jnp.ndarray, moon_position: jnp.ndarray, constants: tuple[float, float, float, float, float, float, float, float], role: str) -> jnp.ndarray:
        enabled, beam_power_w, aperture_m, _, _, _, _, _ = constants
        if enabled <= 0.0:
            return 0.0
        station_position_km = _station_position(absolute_seconds, moon_position, constants, role)
        line = spacecraft_position - station_position_km
        range_km = _safe_norm(line)
        gate = _beam_gate(spacecraft_position, station_position_km, moon_position, constants, role)
        sun_position, _ = _sun_state(absolute_seconds)
        sun_vector = sun_position - spacecraft_position
        earth_occulting = -spacecraft_position
        moon_occulting = moon_position - spacecraft_position
        shadow_factor = jnp.minimum(
            _soft_eclipse_fraction(sun_vector, earth_occulting, EARTH_RADIUS_KM),
            _soft_eclipse_fraction(sun_vector, moon_occulting, MOON_RADIUS_KM),
        )
        spot_radius_m = jnp.maximum(0.2, 0.5 * aperture_m + base.beam_quality_factor * base.beam_wavelength_m * range_km * 1_000.0 / jnp.maximum(aperture_m, 0.1))
        receiver_radius_m = jnp.sqrt(base.vehicle.receiver_area_m2 / math.pi)
        capture_fraction = jnp.minimum(1.0, (receiver_radius_m / jnp.maximum(spot_radius_m, 1.0e-6)) ** 2)
        return beam_power_w * gate * shadow_factor * capture_fraction * base.vehicle.beam_receiver_efficiency

    def _schedule_value(fraction: jnp.ndarray) -> jnp.ndarray:
        first = throttle_nodes[0] + (throttle_nodes[1] - throttle_nodes[0]) * fraction / 0.5
        second = throttle_nodes[1] + (throttle_nodes[2] - throttle_nodes[1]) * (fraction - 0.5) / 0.3
        third = throttle_nodes[2] + (throttle_nodes[3] - throttle_nodes[2]) * (fraction - 0.8) / 0.2
        return jnp.where(fraction <= 0.5, first, jnp.where(fraction <= 0.8, second, third))

    def _specific_energy(position: jnp.ndarray, velocity: jnp.ndarray, moon_position: jnp.ndarray, moon_velocity: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        relative_position = position - moon_position
        relative_velocity = velocity - moon_velocity
        radius = _safe_norm(relative_position)
        return 0.5 * jnp.dot(relative_velocity, relative_velocity) - MOON_MU / radius, radius

    def _thrust_context(
        position: jnp.ndarray,
        velocity: jnp.ndarray,
        mass_kg: jnp.ndarray,
        elapsed_seconds: jnp.ndarray,
        absolute_seconds: jnp.ndarray,
        moon_position: jnp.ndarray,
        departure_apogee_km: jnp.ndarray,
        boosted_perigee_passes: jnp.ndarray,
        perigee_boost_scale: jnp.ndarray,
        lunar_brake_scale: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        del velocity
        del mass_kg
        fraction = jnp.clip(elapsed_seconds / jnp.maximum(mission_duration_seconds, 1.0), 0.0, 1.0)
        throttle = jnp.clip(_schedule_value(fraction), 0.0, 1.0)
        semimajor_axis = 0.5 * (departure_apogee_km + perigee_radius)
        period_days = 2.0 * math.pi * jnp.sqrt(semimajor_axis**3 / EARTH_MU) / SECONDS_PER_DAY
        elapsed_days = elapsed_seconds / SECONDS_PER_DAY
        earth_distance = _safe_norm(position)
        moon_distance = _safe_norm(position - moon_position)
        sep_gate = jax.nn.sigmoid((elapsed_days - base.sep_start_delay_days) / 0.2)
        perigee_window = sep_gate * jax.nn.sigmoid((boosted_perigee_passes - elapsed_days / jnp.maximum(period_days, 0.1)) / 0.2) * jax.nn.sigmoid(((EARTH_RADIUS_KM + base.departure_perigee_altitude_km + 20_000.0) - earth_distance) / 3_000.0)
        lunar_window = sep_gate * jax.nn.sigmoid((jnp.minimum(120_000.0, lunar_constants[3]) - moon_distance) / 3_000.0)
        beam_power_perigee = perigee_window * perigee_boost_scale * _received_beam_power(position, absolute_seconds, moon_position, perigee_constants, "perigee_boost")
        beam_power_lunar = lunar_window * lunar_brake_scale * _received_beam_power(position, absolute_seconds, moon_position, lunar_constants, "lunar_brake")
        available_power = jnp.minimum(base.vehicle.thermal_power_cap_w, jnp.minimum(base.vehicle.propulsion_power_cap_w, base.vehicle.onboard_solar_power_w + beam_power_perigee + beam_power_lunar))
        return available_power * throttle, beam_power_perigee, beam_power_lunar

    def _dynamics(
        state: jnp.ndarray,
        elapsed_seconds: jnp.ndarray,
        absolute_seconds: jnp.ndarray,
        departure_apogee_km: jnp.ndarray,
        boosted_perigee_passes: jnp.ndarray,
        perigee_boost_scale: jnp.ndarray,
        lunar_brake_scale: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        position = state[:3]
        velocity = state[3:6]
        mass_kg = state[6]
        moon_position, moon_velocity = _moon_state(absolute_seconds)
        sun_position, _ = _sun_state(absolute_seconds)
        available_power, beam_power_perigee, beam_power_lunar = _thrust_context(
            position,
            velocity,
            mass_kg,
            elapsed_seconds,
            absolute_seconds,
            moon_position,
            departure_apogee_km,
            boosted_perigee_passes,
            perigee_boost_scale,
            lunar_brake_scale,
        )
        exhaust_velocity_mps = base.vehicle.isp_seconds * G0_MPS2
        thrust_newtons = 2.0 * base.vehicle.thruster_efficiency * available_power / jnp.maximum(exhaust_velocity_mps, 1.0)
        mdot_kg_s = thrust_newtons / jnp.maximum(exhaust_velocity_mps, 1.0)
        vel_dir = _unit(velocity)
        moon_brake_dir = -_unit(velocity - moon_velocity)
        lunar_bias = jax.nn.sigmoid((beam_power_lunar - beam_power_perigee) / 20.0)
        thrust_dir = _unit((1.0 - lunar_bias) * vel_dir + lunar_bias * moon_brake_dir)
        mass_gate = jax.nn.sigmoid((mass_kg - base.vehicle.minimum_final_mass_kg) / 0.1)
        thrust_acceleration = thrust_dir * (thrust_newtons / jnp.maximum(mass_kg, base.vehicle.minimum_final_mass_kg) / 1_000.0) * mass_gate
        mdot_kg_s = mdot_kg_s * mass_gate
        total_acceleration = (
            _earth_acceleration(position)
            + _earth_j2_acceleration(position)
            + _third_body_acceleration(position, moon_position, MOON_MU)
            + _third_body_acceleration(position, sun_position, SUN_MU)
            + thrust_acceleration
        )
        derivative = jnp.concatenate((velocity, total_acceleration, jnp.asarray((-mdot_kg_s,), dtype=jnp.float64)))
        return derivative, beam_power_perigee, beam_power_lunar

    def _rk4_step(
        state: jnp.ndarray,
        elapsed_seconds: jnp.ndarray,
        absolute_offset_seconds: jnp.ndarray,
        departure_apogee_km: jnp.ndarray,
        boosted_perigee_passes: jnp.ndarray,
        perigee_boost_scale: jnp.ndarray,
        lunar_brake_scale: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def derivative(test_state: jnp.ndarray, local_time: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            return _dynamics(
                test_state,
                local_time,
                absolute_offset_seconds + local_time,
                departure_apogee_km,
                boosted_perigee_passes,
                perigee_boost_scale,
                lunar_brake_scale,
            )

        k1, bp1, bl1 = derivative(state, elapsed_seconds)
        k2, bp2, bl2 = derivative(state + 0.5 * dt * k1, elapsed_seconds + 0.5 * dt)
        k3, bp3, bl3 = derivative(state + 0.5 * dt * k2, elapsed_seconds + 0.5 * dt)
        k4, bp4, bl4 = derivative(state + dt * k3, elapsed_seconds + dt)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state = next_state.at[6].set(jnp.maximum(base.vehicle.minimum_final_mass_kg, next_state[6]))
        beam_power_perigee = (bp1 + 2.0 * bp2 + 2.0 * bp3 + bp4) / 6.0
        beam_power_lunar = (bl1 + 2.0 * bl2 + 2.0 * bl3 + bl4) / 6.0
        return next_state, beam_power_perigee, beam_power_lunar

    def _surrogate_metrics(unit_vector: jnp.ndarray) -> dict[str, jnp.ndarray]:
        unit = jnp.clip(unit_vector, 0.0, 1.0)
        design = lower + unit * (upper - lower)
        departure_apogee_km = design[0]
        epoch_shift_days = design[1]
        departure_wait_days = design[2]
        speed_scale = design[3]
        boosted_perigee_passes = design[4]
        perigee_boost_scale = design[5]
        lunar_brake_scale = design[6]

        semimajor_axis = 0.5 * (departure_apogee_km + perigee_radius)
        speed = jnp.sqrt(jnp.maximum(0.0, EARTH_MU * (2.0 / perigee_radius - 1.0 / semimajor_axis))) * speed_scale
        initial_position = _rotate_orbit(jnp.asarray((perigee_radius, 0.0, 0.0), dtype=jnp.float64))
        tangential = _unit(jnp.asarray((0.0, jnp.cos(base.flight_path_angle_rad), jnp.sin(base.flight_path_angle_rad)), dtype=jnp.float64))
        initial_velocity = _rotate_orbit(tangential * speed)
        initial_state = jnp.concatenate((initial_position, initial_velocity, jnp.asarray((base.vehicle.wet_mass_kg,), dtype=jnp.float64)))
        absolute_offset_seconds = (epoch_shift_days + departure_wait_days) * SECONDS_PER_DAY

        def scan_step(carry: tuple[jnp.ndarray, ...], step_index: jnp.ndarray) -> tuple[tuple[jnp.ndarray, ...], None]:
            state, min_moon_distance_km, max_earth_distance_km, capture_duration_seconds, capture_weight_sum, capture_radius_sum, total_beam_energy_j, perigee_dwell_seconds, lunar_dwell_seconds = carry
            elapsed_seconds = step_index * dt
            absolute_seconds = absolute_offset_seconds + elapsed_seconds
            moon_position, moon_velocity = _moon_state(absolute_seconds)
            energy, moon_distance = _specific_energy(state[:3], state[3:6], moon_position, moon_velocity)
            soft_capture = jax.nn.sigmoid((-energy) / 0.04) * jax.nn.sigmoid((capture_radius_limit_km - moon_distance) / 4_000.0)
            next_state, beam_power_perigee, beam_power_lunar = _rk4_step(
                state,
                elapsed_seconds,
                absolute_offset_seconds,
                departure_apogee_km,
                boosted_perigee_passes,
                perigee_boost_scale,
                lunar_brake_scale,
            )
            return (
                next_state,
                jnp.minimum(min_moon_distance_km, moon_distance),
                jnp.maximum(max_earth_distance_km, _safe_norm(state[:3])),
                capture_duration_seconds + soft_capture * dt,
                capture_weight_sum + soft_capture,
                capture_radius_sum + soft_capture * moon_distance,
                total_beam_energy_j + (beam_power_perigee + beam_power_lunar) * dt,
                perigee_dwell_seconds + jax.nn.sigmoid((beam_power_perigee - 1.0) / 5.0) * dt,
                lunar_dwell_seconds + jax.nn.sigmoid((beam_power_lunar - 1.0) / 5.0) * dt,
            ), None

        carry0 = (
            initial_state,
            3.0e6,
            _safe_norm(initial_position),
            0.0,
            1.0e-6,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        carry_final, _ = jax.lax.scan(scan_step, carry0, jnp.arange(steps, dtype=jnp.float64))
        final_state, min_moon_distance_km, max_earth_distance_km, capture_duration_seconds, capture_weight_sum, capture_radius_sum, total_beam_energy_j, perigee_dwell_seconds, lunar_dwell_seconds = carry_final
        final_moon_position, final_moon_velocity = _moon_state(absolute_offset_seconds + steps * dt)
        final_energy, _ = _specific_energy(final_state[:3], final_state[3:6], final_moon_position, final_moon_velocity)
        stable_capture_duration_days = capture_duration_seconds / SECONDS_PER_DAY
        perilune_altitude_km = min_moon_distance_km - MOON_RADIUS_KM
        capture_apoapsis_km = jnp.where(capture_weight_sum > 0.01, capture_radius_sum / capture_weight_sum - MOON_RADIUS_KM, base.success.max_capture_apoapsis_km * 4.0)
        final_mass_kg = final_state[6]
        propellant_used_kg = base.vehicle.wet_mass_kg - final_mass_kg
        total_beam_energy_mj = total_beam_energy_j / 1.0e6

        objective = 0.0
        objective += 45.0 * jnp.square(jax.nn.relu(base.success.min_capture_duration_days - stable_capture_duration_days))
        objective += 0.002 * jnp.square(jax.nn.relu(base.success.min_perilune_altitude_km - perilune_altitude_km))
        objective += 0.00002 * jnp.square(jax.nn.relu(perilune_altitude_km - base.success.max_perilune_altitude_km))
        objective += 0.00002 * jnp.square(jax.nn.relu(capture_apoapsis_km - base.success.max_capture_apoapsis_km))
        objective += 8.0 * jnp.square(jax.nn.relu(base.success.min_final_mass_kg - final_mass_kg))
        objective += 25.0 * jnp.square(jax.nn.relu(final_energy + 0.02))
        objective += 0.15 * propellant_used_kg
        objective += 0.01 * total_beam_energy_mj
        objective += jnp.abs(perilune_altitude_km - 400.0) / 2_000.0
        objective -= 4.0 * stable_capture_duration_days

        return {
            "objective": objective,
            "stable_capture_duration_days": stable_capture_duration_days,
            "perilune_altitude_km": perilune_altitude_km,
            "capture_apoapsis_km": capture_apoapsis_km,
            "final_mass_kg": final_mass_kg,
            "propellant_used_kg": propellant_used_kg,
            "total_beam_energy_mj": total_beam_energy_mj,
            "perigee_boost_dwell_hours": perigee_dwell_seconds / 3600.0,
            "lunar_brake_dwell_hours": lunar_dwell_seconds / 3600.0,
            "min_moon_distance_km": min_moon_distance_km,
            "max_earth_distance_km": max_earth_distance_km,
            "final_moon_specific_energy_km2s2": final_energy,
        }

    metrics_fn = jax.jit(_surrogate_metrics)
    objective_fn = jax.jit(lambda unit_vector: metrics_fn(unit_vector)["objective"])
    value_and_grad_fn = jax.jit(jax.value_and_grad(objective_fn))
    batched_metrics_fn = jax.jit(jax.vmap(metrics_fn))
    batched_objective_fn = jax.jit(jax.vmap(objective_fn))
    return JaxSurrogateProblem(
        base_config=base,
        search_space=search_space,
        metrics_fn=metrics_fn,
        objective_fn=objective_fn,
        value_and_grad_fn=value_and_grad_fn,
        batched_metrics_fn=batched_metrics_fn,
        batched_objective_fn=batched_objective_fn,
    )
