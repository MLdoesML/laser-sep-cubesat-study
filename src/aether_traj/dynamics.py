from __future__ import annotations

from dataclasses import replace
import math

from aether_traj.ephemeris import (
    ANALYTIC_HIACC_TABLE_ID,
    DE440_HIACC_TABLE_ID,
    EphemerisState,
    get_ephemeris_table,
    sample_ephemeris_state,
    sample_ephemeris_state_direct,
    sample_moon_j2000_to_pa,
    sample_moon_j2000_to_pa_direct,
)
from aether_traj.gravity import (
    earth_gravity_model,
    earth_point_mass_acceleration,
    eclipse_fraction_python,
    moon_gravity_model,
    spherical_harmonic_acceleration_python,
    third_body_point_mass_acceleration,
)
from aether_traj.models import (
    EARTH_MU,
    EARTH_RADIUS_KM,
    G0_MPS2,
    MOON_MU,
    MOON_RADIUS_KM,
    SECONDS_PER_DAY,
    SUN_MU,
    BeamStationFamilyConfig,
    SimulationConfig,
    SimulationResult,
    State,
    Vec3,
)


def _rotate_z(vector: Vec3, angle_rad: float) -> Vec3:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return Vec3(
        cosine * vector.x - sine * vector.y,
        sine * vector.x + cosine * vector.y,
        vector.z,
    )


def _rotate_x(vector: Vec3, angle_rad: float) -> Vec3:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return Vec3(
        vector.x,
        cosine * vector.y - sine * vector.z,
        sine * vector.y + cosine * vector.z,
    )


def _rotate_by_orbit_angles(vector: Vec3, config: SimulationConfig) -> Vec3:
    rotated = _rotate_z(vector, config.departure_argument_of_latitude_rad)
    rotated = _rotate_x(rotated, config.departure_inclination_rad)
    rotated = _rotate_z(rotated, config.departure_raan_rad)
    return rotated


def _project_with_matrix(vector: Vec3, matrix: tuple[tuple[float, float, float], ...]) -> Vec3:
    return Vec3(
        matrix[0][0] * vector.x + matrix[0][1] * vector.y + matrix[0][2] * vector.z,
        matrix[1][0] * vector.x + matrix[1][1] * vector.y + matrix[1][2] * vector.z,
        matrix[2][0] * vector.x + matrix[2][1] * vector.y + matrix[2][2] * vector.z,
    )


def _expand_with_matrix(vector: Vec3, matrix: tuple[tuple[float, float, float], ...]) -> Vec3:
    return Vec3(
        matrix[0][0] * vector.x + matrix[1][0] * vector.y + matrix[2][0] * vector.z,
        matrix[0][1] * vector.x + matrix[1][1] * vector.y + matrix[2][1] * vector.z,
        matrix[0][2] * vector.x + matrix[1][2] * vector.y + matrix[2][2] * vector.z,
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _smoothstep_quintic(fraction: float) -> float:
    value = _clamp(fraction, 0.0, 1.0)
    return value * value * value * (10.0 + value * (-15.0 + 6.0 * value))


def departure_period_days(config: SimulationConfig) -> float:
    perigee_radius = EARTH_RADIUS_KM + config.departure_perigee_altitude_km
    semimajor_axis = 0.5 * (config.departure_apogee_km + perigee_radius)
    period_seconds = 2.0 * math.pi * math.sqrt(semimajor_axis**3 / EARTH_MU)
    return period_seconds / SECONDS_PER_DAY


def build_initial_state(config: SimulationConfig) -> State:
    perigee_radius = EARTH_RADIUS_KM + config.departure_perigee_altitude_km
    semimajor_axis = 0.5 * (config.departure_apogee_km + perigee_radius)
    speed = math.sqrt(max(0.0, EARTH_MU * (2.0 / perigee_radius - 1.0 / semimajor_axis))) * config.speed_scale
    radial = Vec3(1.0, 0.0, 0.0)
    tangential = Vec3(0.0, math.cos(config.flight_path_angle_rad), math.sin(config.flight_path_angle_rad))
    position = _rotate_by_orbit_angles(radial * perigee_radius, config)
    velocity = _rotate_by_orbit_angles(tangential.unit() * speed, config)
    return State(position=position, velocity=velocity, mass_kg=config.vehicle.wet_mass_kg)


def _ephemeris_table_for_config(config: SimulationConfig):
    if config.ephemeris_mode == "cached_table":
        return get_ephemeris_table(config.ephemeris_table_id, config.ephemeris_cache_step_seconds)
    if config.ephemeris_mode == "analytic_cache":
        return get_ephemeris_table(ANALYTIC_HIACC_TABLE_ID, config.ephemeris_cache_step_seconds)
    return None


def _ephemeris_state(body: str, seconds: float, config: SimulationConfig) -> EphemerisState:
    if config.ephemeris_mode == "spice_direct":
        return sample_ephemeris_state_direct(body, seconds)
    if config.ephemeris_mode == "analytic_direct":
        return sample_ephemeris_state_direct(body, seconds, prefer_analytic=True)
    table = _ephemeris_table_for_config(config)
    if table is None:
        raise ValueError(f"unsupported ephemeris mode: {config.ephemeris_mode}")
    return sample_ephemeris_state(table, body, seconds)


def moon_state(seconds: float, config: SimulationConfig) -> EphemerisState:
    return _ephemeris_state("moon", seconds, config)


def sun_state(seconds: float, config: SimulationConfig) -> EphemerisState:
    return _ephemeris_state("sun", seconds, config)


def _legacy_moon_body_matrix(moon_position: Vec3) -> tuple[tuple[float, float, float], ...]:
    radial = moon_position.unit()
    north = Vec3(0.0, 0.0, 1.0)
    east = north.cross(radial)
    if east.norm() < 1.0e-9:
        east = Vec3(0.0, 1.0, 0.0).cross(radial)
    east = east.unit()
    normal = radial.cross(east).unit()
    return (
        radial.as_tuple(),
        east.as_tuple(),
        normal.as_tuple(),
    )


def moon_j2000_to_pa_matrix(seconds: float, config: SimulationConfig) -> tuple[tuple[float, float, float], ...]:
    if config.ephemeris_mode == "spice_direct":
        return sample_moon_j2000_to_pa_direct(seconds)
    table = _ephemeris_table_for_config(config)
    if table is not None and table.moon_j2000_to_pa is not None:
        return sample_moon_j2000_to_pa(table, seconds)
    moon_ephem = moon_state(seconds, config)
    moon_position = Vec3(moon_ephem.position.x, moon_ephem.position.y, moon_ephem.position.z)
    return _legacy_moon_body_matrix(moon_position)


def station_position(seconds: float, station: BeamStationFamilyConfig, moon_position: Vec3) -> Vec3:
    phase = math.radians(station.phase_deg) + 2.0 * math.pi * (seconds / SECONDS_PER_DAY) / max(station.orbit_period_days, 1.0e-6)
    if station.role == "perigee_boost":
        return Vec3(
            station.orbit_radius_km * math.cos(phase),
            station.orbit_radius_km * math.sin(phase),
            0.0,
        )
    radial = moon_position.unit()
    tangential = Vec3(-radial.y, radial.x, 0.0).unit()
    if tangential.norm() <= 0.0:
        tangential = Vec3(0.0, 1.0, 0.0)
    out_of_plane = radial.cross(tangential).unit()
    return moon_position + tangential * (station.orbit_radius_km * math.cos(phase)) + out_of_plane * (0.3 * station.orbit_radius_km * math.sin(phase))


def _beam_gate_factor(spacecraft_position: Vec3, station_position_km: Vec3, moon_position: Vec3, station: BeamStationFamilyConfig) -> float:
    line = spacecraft_position - station_position_km
    line_distance = line.norm()
    if line_distance <= 0.0 or line_distance > station.useful_range_km:
        return 0.0
    target_origin = Vec3(0.0, 0.0, 0.0) if station.role == "perigee_boost" else moon_position
    reference = (target_origin - station_position_km).unit()
    if reference.norm() <= 0.0:
        return 0.0
    cosine = max(-1.0, min(1.0, reference.dot(line.unit())))
    pointing_angle_deg = math.degrees(math.acos(cosine))
    if pointing_angle_deg > station.allowed_pointing_half_angle_deg:
        return 0.0
    return max(0.0, min(1.0, station.duty_cycle))


def received_beam_power_w(
    spacecraft_position: Vec3,
    seconds: float,
    station: BeamStationFamilyConfig | None,
    config: SimulationConfig,
    moon_position: Vec3,
) -> float:
    if station is None:
        return 0.0
    station_pos = station_position(seconds, station, moon_position)
    line = spacecraft_position - station_pos
    range_km = line.norm()
    if range_km <= 0.0 or range_km > station.useful_range_km:
        return 0.0
    gate = _beam_gate_factor(spacecraft_position, station_pos, moon_position, station)
    if gate <= 0.0:
        return 0.0
    sun_ephem = sun_state(seconds, config)
    sun_vector = Vec3(sun_ephem.position.x, sun_ephem.position.y, sun_ephem.position.z) - spacecraft_position
    earth_occulting = spacecraft_position * -1.0
    moon_occulting = moon_position - spacecraft_position
    shadow_factor = min(
        eclipse_fraction_python(sun_vector.as_tuple(), earth_occulting.as_tuple(), EARTH_RADIUS_KM),
        eclipse_fraction_python(sun_vector.as_tuple(), moon_occulting.as_tuple(), MOON_RADIUS_KM),
    )
    spot_radius_m = max(
        0.2,
        0.5 * station.aperture_m + config.beam_quality_factor * config.beam_wavelength_m * range_km * 1_000.0 / max(station.aperture_m, 0.1),
    )
    receiver_radius_m = math.sqrt(config.vehicle.receiver_area_m2 / math.pi)
    capture_fraction = min(1.0, (receiver_radius_m / max(spot_radius_m, 1.0e-6)) ** 2)
    return station.beam_power_w * gate * shadow_factor * capture_fraction * config.vehicle.beam_receiver_efficiency


def _mission_fraction(seconds: float, config: SimulationConfig) -> float:
    return max(0.0, min(1.0, seconds / max(config.duration_days * SECONDS_PER_DAY, 1.0)))


def _schedule_value(nodes: tuple[float, float, float, float], fraction: float) -> float:
    knots = (0.0, 0.5, 0.8, 1.0)
    for index in range(len(knots) - 1):
        left = knots[index]
        right = knots[index + 1]
        if fraction <= right or index == len(knots) - 2:
            local_fraction = 0.0 if right == left else (fraction - left) / (right - left)
            return nodes[index] + (nodes[index + 1] - nodes[index]) * local_fraction
    return nodes[-1]


def thrust_context(seconds: float, state: State, config: SimulationConfig, moon_position: Vec3, absolute_seconds: float | None = None) -> tuple[float, dict[str, float]]:
    local_days = seconds / SECONDS_PER_DAY
    sample_seconds = seconds if absolute_seconds is None else absolute_seconds
    fraction = _mission_fraction(seconds, config)
    throttle = max(0.0, min(1.0, _schedule_value(config.throttle_nodes, fraction)))
    onboard = config.vehicle.onboard_solar_power_w
    perigee_station = config.beam_architecture.perigee_boost_station
    lunar_station = config.beam_architecture.lunar_brake_station
    moon_distance = (state.position - moon_position).norm()
    earth_distance = state.position.norm()
    period_days = departure_period_days(config)
    active_perigee_window = (
        perigee_station is not None
        and local_days >= config.sep_start_delay_days
        and int(local_days / max(period_days, 0.1)) < max(0, config.boosted_perigee_passes)
        and earth_distance <= EARTH_RADIUS_KM + config.departure_perigee_altitude_km + 20_000.0
    )
    active_lunar_window = (
        lunar_station is not None
        and local_days >= config.sep_start_delay_days
        and moon_distance <= min(120_000.0, lunar_station.useful_range_km)
    )
    beam_power_perigee = 0.0
    beam_power_lunar = 0.0
    if active_perigee_window and perigee_station is not None:
        beam_power_perigee = received_beam_power_w(state.position, sample_seconds, perigee_station, config, moon_position) * config.perigee_boost_power_scale
    if active_lunar_window and lunar_station is not None:
        beam_power_lunar = received_beam_power_w(state.position, sample_seconds, lunar_station, config, moon_position) * config.lunar_brake_power_scale
    available_power = min(config.vehicle.propulsion_power_cap_w, onboard + beam_power_perigee + beam_power_lunar)
    available_power = min(available_power, config.vehicle.thermal_power_cap_w)
    context = {
        "throttle": throttle,
        "beam_power_perigee_w": beam_power_perigee,
        "beam_power_lunar_w": beam_power_lunar,
        "available_power_w": available_power,
    }
    return available_power * throttle, context


def _earth_acceleration(position: Vec3, config: SimulationConfig) -> Vec3:
    degree = 0 if not config.earth_j2_enabled else max(0, config.earth_gravity_degree)
    if degree <= 0:
        return earth_point_mass_acceleration(position, EARTH_MU)
    acceleration = spherical_harmonic_acceleration_python(
        position.as_tuple(),
        earth_gravity_model(),
        degree,
        0,
    )
    return Vec3(*acceleration)


def _lunar_direct_acceleration(position: Vec3, moon_ephemeris: EphemerisState, seconds: float, config: SimulationConfig) -> Vec3:
    relative_position = position - Vec3(
        moon_ephemeris.position.x,
        moon_ephemeris.position.y,
        moon_ephemeris.position.z,
    )
    distance = relative_position.norm()
    if distance <= 0.0:
        return Vec3(0.0, 0.0, 0.0)
    point_mass = relative_position * (-MOON_MU / (distance**3))
    degree = max(0, config.moon_harmonics_degree)
    order = max(0, config.moon_harmonics_order)
    if degree < 2 or order < 0:
        return point_mass
    altitude = distance - MOON_RADIUS_KM
    if altitude >= config.moon_harmonics_blend_start_altitude_km:
        return point_mass
    j2000_to_pa = moon_j2000_to_pa_matrix(seconds, config)
    body_relative = _project_with_matrix(relative_position, j2000_to_pa)
    harmonic_body = Vec3(
        *spherical_harmonic_acceleration_python(
            body_relative.as_tuple(),
            moon_gravity_model(),
            degree,
            order,
        )
    )
    harmonic_inertial = _expand_with_matrix(harmonic_body, j2000_to_pa)
    if altitude <= config.moon_harmonics_blend_full_altitude_km:
        return harmonic_inertial
    blend_denominator = max(
        config.moon_harmonics_blend_start_altitude_km - config.moon_harmonics_blend_full_altitude_km,
        1.0,
    )
    fraction = (config.moon_harmonics_blend_start_altitude_km - altitude) / blend_denominator
    weight = _smoothstep_quintic(fraction)
    return point_mass * (1.0 - weight) + harmonic_inertial * weight


def gravity_with_indirect_terms(position: Vec3, seconds: float, config: SimulationConfig) -> Vec3:
    earth_acc = _earth_acceleration(position, config)
    moon_ephem = moon_state(seconds, config)
    moon_position_vector = Vec3(moon_ephem.position.x, moon_ephem.position.y, moon_ephem.position.z)
    moon_direct = _lunar_direct_acceleration(position, moon_ephem, seconds, config)
    moon_norm = moon_position_vector.norm()
    moon_indirect = Vec3(0.0, 0.0, 0.0) if moon_norm <= 0.0 else moon_position_vector * (-MOON_MU / (moon_norm**3))
    sun_ephem = sun_state(seconds, config)
    sun_position_vector = Vec3(sun_ephem.position.x, sun_ephem.position.y, sun_ephem.position.z)
    sun_acc = third_body_point_mass_acceleration(position, sun_position_vector, SUN_MU)
    return earth_acc + moon_direct + moon_indirect + sun_acc


def thrust_acceleration(state: State, seconds: float, config: SimulationConfig, absolute_seconds: float | None = None) -> tuple[Vec3, float, dict[str, float]]:
    sample_seconds = seconds if absolute_seconds is None else absolute_seconds
    moon_ephem = moon_state(sample_seconds, config)
    moon_position = Vec3(moon_ephem.position.x, moon_ephem.position.y, moon_ephem.position.z)
    available_power_w, context = thrust_context(seconds, state, config, moon_position, absolute_seconds=sample_seconds)
    if available_power_w <= 0.0 or state.mass_kg <= config.vehicle.minimum_final_mass_kg:
        return Vec3(0.0, 0.0, 0.0), 0.0, context
    exhaust_velocity_mps = config.vehicle.isp_seconds * G0_MPS2
    thrust_newtons = 2.0 * config.vehicle.thruster_efficiency * available_power_w / max(exhaust_velocity_mps, 1.0)
    mdot_kg_s = thrust_newtons / max(exhaust_velocity_mps, 1.0)
    moon_relative_velocity = state.velocity - Vec3(moon_ephem.velocity.x, moon_ephem.velocity.y, moon_ephem.velocity.z)
    if context["beam_power_lunar_w"] > context["beam_power_perigee_w"] and moon_relative_velocity.norm() > 0.0:
        direction = moon_relative_velocity.unit() * -1.0
    elif state.velocity.norm() > 0.0:
        direction = state.velocity.unit()
    else:
        direction = Vec3(0.0, 0.0, 0.0)
    acceleration_kms2 = thrust_newtons / max(state.mass_kg, 1.0e-9) / 1_000.0
    return direction * acceleration_kms2, mdot_kg_s, context


def total_acceleration_and_mass_flow(state: State, seconds: float, config: SimulationConfig, truth: bool = False, absolute_seconds: float | None = None) -> tuple[Vec3, float, dict[str, float]]:
    effective_config = replace(config, ephemeris_mode="spice_direct", physical_integrator="dop853") if truth else config
    sample_seconds = seconds if absolute_seconds is None else absolute_seconds
    acceleration = gravity_with_indirect_terms(state.position, sample_seconds, effective_config)
    thrust_accel, mdot_kg_s, context = thrust_acceleration(state, seconds, effective_config, absolute_seconds=sample_seconds)
    return acceleration + thrust_accel, mdot_kg_s, context


def rk4_step(state: State, seconds: float, dt: float, config: SimulationConfig, truth: bool = False, absolute_offset_seconds: float = 0.0) -> tuple[State, dict[str, float]]:
    def derivative(test_state: State, time_value: float) -> tuple[Vec3, Vec3, float, dict[str, float]]:
        acceleration, mdot_kg_s, context = total_acceleration_and_mass_flow(
            test_state,
            time_value,
            config,
            truth=truth,
            absolute_seconds=absolute_offset_seconds + time_value,
        )
        return test_state.velocity, acceleration, -mdot_kg_s, context

    def advance(base: State, k_position: Vec3, k_velocity: Vec3, k_mass: float, factor: float) -> State:
        return State(
            position=base.position + k_position * factor,
            velocity=base.velocity + k_velocity * factor,
            mass_kg=max(config.vehicle.minimum_final_mass_kg, base.mass_kg + k_mass * factor),
        )

    k1p, k1v, k1m, c1 = derivative(state, seconds)
    s2 = advance(state, k1p, k1v, k1m, 0.5 * dt)
    k2p, k2v, k2m, c2 = derivative(s2, seconds + 0.5 * dt)
    s3 = advance(state, k2p, k2v, k2m, 0.5 * dt)
    k3p, k3v, k3m, c3 = derivative(s3, seconds + 0.5 * dt)
    s4 = advance(state, k3p, k3v, k3m, dt)
    k4p, k4v, k4m, c4 = derivative(s4, seconds + dt)

    next_state = State(
        position=state.position + (k1p + k2p * 2.0 + k3p * 2.0 + k4p) * (dt / 6.0),
        velocity=state.velocity + (k1v + k2v * 2.0 + k3v * 2.0 + k4v) * (dt / 6.0),
        mass_kg=max(config.vehicle.minimum_final_mass_kg, state.mass_kg + (k1m + 2.0 * k2m + 2.0 * k3m + k4m) * (dt / 6.0)),
    )
    context = {
        key: (c1[key] + 2.0 * c2[key] + 2.0 * c3[key] + c4[key]) / 6.0
        for key in c1
    }
    return next_state, context


def _specific_energy_to_moon(state: State, moon_ephem: EphemerisState) -> tuple[float, float]:
    moon_position = Vec3(moon_ephem.position.x, moon_ephem.position.y, moon_ephem.position.z)
    moon_velocity = Vec3(moon_ephem.velocity.x, moon_ephem.velocity.y, moon_ephem.velocity.z)
    relative_position = state.position - moon_position
    relative_velocity = state.velocity - moon_velocity
    radius = relative_position.norm()
    if radius <= 0.0:
        return float("inf"), 0.0
    energy = 0.5 * relative_velocity.dot(relative_velocity) - MOON_MU / radius
    return energy, radius


def _state_to_vector(state: State) -> list[float]:
    return [
        state.position.x,
        state.position.y,
        state.position.z,
        state.velocity.x,
        state.velocity.y,
        state.velocity.z,
        state.mass_kg,
    ]


def _state_from_vector(vector: list[float] | tuple[float, ...]) -> State:
    return State(
        position=Vec3(vector[0], vector[1], vector[2]),
        velocity=Vec3(vector[3], vector[4], vector[5]),
        mass_kg=vector[6],
    )


def _unique_sorted_times(values: list[float], minimum_step: float = 1.0e-9) -> list[float]:
    ordered = sorted(values)
    deduped: list[float] = []
    for value in ordered:
        if not deduped or abs(value - deduped[-1]) > minimum_step:
            deduped.append(value)
    return deduped


def _summarize_samples(
    config: SimulationConfig,
    label: str,
    analysis_samples: list[tuple[float, State]],
    stored_samples: list[tuple[float, State]],
) -> SimulationResult:
    min_moon_distance = float("inf")
    perilune_altitude = float("inf")
    max_earth_distance = 0.0
    final_energy = float("inf")
    total_beam_energy_j = 0.0
    dwell_seconds = {"perigee_boost": 0.0, "lunar_brake": 0.0}
    best_capture_seconds = 0.0
    best_capture_radius = 0.0
    current_capture_seconds = 0.0
    current_capture_radius = 0.0
    absolute_offset_seconds = (config.epoch_shift_days + config.departure_wait_days) * SECONDS_PER_DAY

    for index, (seconds, state) in enumerate(analysis_samples):
        absolute_seconds = absolute_offset_seconds + seconds
        moon_ephem = moon_state(absolute_seconds, config)
        moon_position = Vec3(moon_ephem.position.x, moon_ephem.position.y, moon_ephem.position.z)
        energy, moon_distance = _specific_energy_to_moon(state, moon_ephem)
        final_energy = energy
        min_moon_distance = min(min_moon_distance, moon_distance)
        perilune_altitude = min(perilune_altitude, moon_distance - MOON_RADIUS_KM)
        max_earth_distance = max(max_earth_distance, state.position.norm())

        if index + 1 < len(analysis_samples):
            dt = analysis_samples[index + 1][0] - seconds
            _, context = thrust_context(seconds, state, config, moon_position, absolute_seconds=absolute_seconds)
            total_beam_energy_j += (context["beam_power_perigee_w"] + context["beam_power_lunar_w"]) * dt
            if context["beam_power_perigee_w"] > 0.0:
                dwell_seconds["perigee_boost"] += dt
            if context["beam_power_lunar_w"] > 0.0:
                dwell_seconds["lunar_brake"] += dt
        else:
            dt = 0.0

        if energy < 0.0 and moon_distance <= 120_000.0:
            current_capture_radius = max(current_capture_radius, moon_distance)
            current_capture_seconds += dt
        else:
            if current_capture_seconds > best_capture_seconds:
                best_capture_seconds = current_capture_seconds
                best_capture_radius = current_capture_radius
            current_capture_seconds = 0.0
            current_capture_radius = 0.0

    if current_capture_seconds > best_capture_seconds:
        best_capture_seconds = current_capture_seconds
        best_capture_radius = current_capture_radius

    final_state = analysis_samples[-1][1]
    capture_days = best_capture_seconds / SECONDS_PER_DAY
    capture_apoapsis = best_capture_radius - MOON_RADIUS_KM if best_capture_seconds > 0.0 else None
    unsafe = perilune_altitude < 0.0
    escaped = final_state.position.norm() > config.max_distance_km or final_energy > 0.0
    classification = "flyby"
    if unsafe:
        classification = "unsafe_perilune"
    elif (
        capture_days >= config.success.min_capture_duration_days
        and capture_apoapsis is not None
        and config.success.min_perilune_altitude_km <= perilune_altitude <= config.success.max_perilune_altitude_km
        and capture_apoapsis <= config.success.max_capture_apoapsis_km
        and final_state.mass_kg >= config.success.min_final_mass_kg
    ):
        classification = "direct_capture"
    elif capture_days > 0.1:
        classification = "capture_like"
    elif escaped:
        classification = "escape"

    path = [
        (
            seconds,
            state.position.x,
            state.position.y,
            state.position.z,
            state.velocity.x,
            state.velocity.y,
            state.velocity.z,
            state.mass_kg,
        )
        for seconds, state in stored_samples
    ]
    return SimulationResult(
        label=label,
        config=config,
        classification=classification,
        stable_capture_duration_days=capture_days,
        perilune_altitude_km=perilune_altitude,
        capture_apoapsis_km=capture_apoapsis,
        final_mass_kg=final_state.mass_kg,
        propellant_used_kg=config.vehicle.wet_mass_kg - final_state.mass_kg,
        total_beam_energy_mj=total_beam_energy_j / 1.0e6,
        per_station_dwell_hours={key: value / 3600.0 for key, value in dwell_seconds.items()},
        min_moon_distance_km=min_moon_distance,
        max_earth_distance_km=max_earth_distance,
        final_moon_specific_energy_km2s2=final_energy,
        escaped=escaped,
        unsafe_perilune=unsafe,
        path=path,
    )


def _propagate_fixed_step(config: SimulationConfig, label: str, store_stride: int = 1) -> SimulationResult:
    state = build_initial_state(config)
    dt = config.surrogate_step_seconds
    total_steps = config.total_steps(truth=False)
    absolute_offset_seconds = (config.epoch_shift_days + config.departure_wait_days) * SECONDS_PER_DAY
    analysis_samples: list[tuple[float, State]] = []
    stored_samples: list[tuple[float, State]] = []
    for step in range(total_steps + 1):
        seconds = step * dt
        analysis_samples.append((seconds, state))
        if step % max(1, store_stride) == 0:
            stored_samples.append((seconds, state))
        if step == total_steps or state.position.norm() > config.max_distance_km:
            break
        state, _ = rk4_step(state, seconds, dt, config, truth=False, absolute_offset_seconds=absolute_offset_seconds)
    return _summarize_samples(config, label, analysis_samples, stored_samples)


def _propagate_truth_dop853(config: SimulationConfig, label: str, store_stride: int = 4) -> SimulationResult:
    import numpy as np
    from scipy.integrate import solve_ivp

    truth_config = replace(config, ephemeris_mode="spice_direct", physical_integrator="dop853")
    initial_state = build_initial_state(truth_config)
    initial_offset = (truth_config.epoch_shift_days + truth_config.departure_wait_days) * SECONDS_PER_DAY
    total_duration_seconds = truth_config.duration_days * SECONDS_PER_DAY

    def dynamics(local_seconds: float, vector: np.ndarray) -> np.ndarray:
        current_state = _state_from_vector(tuple(float(value) for value in vector.tolist()))
        acceleration, mdot_kg_s, _ = total_acceleration_and_mass_flow(
            current_state,
            local_seconds,
            truth_config,
            truth=True,
            absolute_seconds=initial_offset + local_seconds,
        )
        return np.array(
            [
                current_state.velocity.x,
                current_state.velocity.y,
                current_state.velocity.z,
                acceleration.x,
                acceleration.y,
                acceleration.z,
                -mdot_kg_s,
            ],
            dtype=np.float64,
        )

    def escape_event(local_seconds: float, vector: np.ndarray) -> float:
        del local_seconds
        current_state = _state_from_vector(tuple(float(value) for value in vector.tolist()))
        return current_state.position.norm() - truth_config.max_distance_km

    escape_event.direction = 1.0
    escape_event.terminal = True

    solution = solve_ivp(
        dynamics,
        (0.0, total_duration_seconds),
        np.array(_state_to_vector(initial_state), dtype=np.float64),
        method="DOP853",
        dense_output=True,
        events=[escape_event],
        rtol=truth_config.physical_rtol,
        atol=truth_config.physical_atol,
        max_step=truth_config.truth_step_seconds,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"truth propagation failed: {solution.message}")

    final_time = float(solution.t[-1])
    analysis_times = list(np.arange(0.0, final_time + truth_config.truth_step_seconds, truth_config.truth_step_seconds, dtype=float))
    analysis_times.append(final_time)
    for event_set in solution.t_events:
        analysis_times.extend(float(value) for value in event_set.tolist())
    metric_times = _unique_sorted_times(analysis_times)
    metric_states = solution.sol(metric_times)
    analysis_samples = [
        (
            local_seconds,
            _state_from_vector(tuple(float(component) for component in metric_states[:, index].tolist())),
        )
        for index, local_seconds in enumerate(metric_times)
    ]
    stored_samples = [
        sample
        for index, sample in enumerate(analysis_samples)
        if index % max(1, store_stride) == 0 or index == len(analysis_samples) - 1
    ]
    return _summarize_samples(truth_config, label, analysis_samples, stored_samples)


def build_truth_validation_config(config: SimulationConfig) -> SimulationConfig:
    return replace(
        config,
        ephemeris_mode="spice_direct",
        physical_integrator="dop853",
        earth_gravity_degree=max(4, config.earth_gravity_degree),
        moon_harmonics_degree=max(8, config.moon_harmonics_degree),
        moon_harmonics_order=max(8, config.moon_harmonics_order),
        physical_rtol=min(config.physical_rtol, 1.0e-10),
        physical_atol=min(config.physical_atol, 1.0e-12),
    )


def propagate(config: SimulationConfig, label: str, truth: bool = False, store_stride: int = 1) -> SimulationResult:
    if truth or config.physical_integrator == "dop853" and config.ephemeris_mode == "spice_direct":
        return _propagate_truth_dop853(config, label, store_stride=max(1, store_stride))
    return _propagate_fixed_step(config, label, store_stride=max(1, store_stride))


def physical_constraint_violation(result: SimulationResult, config: SimulationConfig) -> float:
    violation = 0.0
    if result.stable_capture_duration_days < config.success.min_capture_duration_days:
        violation += config.success.min_capture_duration_days - result.stable_capture_duration_days
    if result.perilune_altitude_km < config.success.min_perilune_altitude_km:
        violation += (config.success.min_perilune_altitude_km - result.perilune_altitude_km) / 100.0
    if result.perilune_altitude_km > config.success.max_perilune_altitude_km:
        violation += (result.perilune_altitude_km - config.success.max_perilune_altitude_km) / 1_000.0
    if result.capture_apoapsis_km is None:
        violation += 10.0
    elif result.capture_apoapsis_km > config.success.max_capture_apoapsis_km:
        violation += (result.capture_apoapsis_km - config.success.max_capture_apoapsis_km) / 5_000.0
    if result.final_mass_kg < config.success.min_final_mass_kg:
        violation += config.success.min_final_mass_kg - result.final_mass_kg
    return max(0.0, violation)


def physics_priority_score(result: SimulationResult, config: SimulationConfig) -> float:
    mapping = {
        "direct_capture": 12.0,
        "capture_like": 7.0,
        "flyby": 2.0,
        "escape": 0.5,
        "unsafe_perilune": -4.0,
    }
    score = mapping.get(result.classification, 0.0)
    score += min(result.stable_capture_duration_days, config.success.min_capture_duration_days) * 2.0
    score -= result.propellant_used_kg * 0.3
    score -= result.total_beam_energy_mj * 0.01
    score -= physical_constraint_violation(result, config) * 3.0
    return score
