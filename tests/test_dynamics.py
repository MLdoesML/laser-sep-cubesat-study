from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from aether_traj.config import build_laser_perigee_boost_config, build_sep_baseline_direct_capture_config  # noqa: E402
from aether_traj.dynamics import (  # noqa: E402
    _beam_gate_factor,
    _lunar_direct_acceleration,
    build_initial_state,
    build_truth_validation_config,
    moon_state,
    propagate,
    station_position,
    thrust_context,
)
from aether_traj.ephemeris import (  # noqa: E402
    DE440_HIACC_TABLE_ID,
    get_ephemeris_table,
    sample_ephemeris_state,
    sample_ephemeris_state_direct,
    sample_moon_j2000_to_pa,
    sample_moon_j2000_to_pa_direct,
)
from aether_traj.gravity import third_body_point_mass_acceleration  # noqa: E402
from aether_traj.jax_surrogate import build_jax_problem  # noqa: E402
from aether_traj.models import MOON_MU, SECONDS_PER_DAY, SUN_MU, Vec3  # noqa: E402


def matrix_angle_error_rad(left: tuple[tuple[float, float, float], ...], right: tuple[tuple[float, float, float], ...]) -> float:
    left_array = np.array(left, dtype=np.float64)
    right_array = np.array(right, dtype=np.float64)
    delta = left_array @ right_array.T
    trace = float(np.trace(delta))
    cosine = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    return math.acos(cosine)


def explicit_third_body_acceleration(position: Vec3, source_position: Vec3, mu: float) -> Vec3:
    delta = position - source_position
    delta_norm = delta.norm()
    source_norm = source_position.norm()
    if delta_norm <= 0.0 or source_norm <= 0.0:
        return Vec3(0.0, 0.0, 0.0)
    return delta * (-mu / (delta_norm**3)) - source_position * (mu / (source_norm**3))


class HighAccuracyDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.table = get_ephemeris_table(DE440_HIACC_TABLE_ID, 600.0)
        cls.cached_config = replace(
            build_sep_baseline_direct_capture_config(),
            duration_days=2.0,
            surrogate_step_seconds=1_200.0,
            truth_step_seconds=600.0,
            ephemeris_mode="cached_table",
            ephemeris_table_id=DE440_HIACC_TABLE_ID,
            ephemeris_cache_step_seconds=600.0,
            earth_gravity_degree=4,
            moon_harmonics_degree=6,
            moon_harmonics_order=6,
        )
        cls.direct_fixed_config = replace(
            cls.cached_config,
            ephemeris_mode="spice_direct",
            physical_integrator="rk4",
        )
        cls.truth_config = build_truth_validation_config(replace(cls.cached_config, duration_days=0.75))

    def test_ephemeris_cache_matches_direct_spice(self) -> None:
        for seconds in (-10.0 * SECONDS_PER_DAY, 0.0, 35.5 * SECONDS_PER_DAY):
            with self.subTest(seconds=seconds):
                moon_cached = sample_ephemeris_state(self.table, "moon", seconds)
                moon_direct = sample_ephemeris_state_direct("moon", seconds)
                sun_cached = sample_ephemeris_state(self.table, "sun", seconds)
                sun_direct = sample_ephemeris_state_direct("sun", seconds)
                moon_position_error = math.dist(moon_cached.position.as_tuple(), moon_direct.position.as_tuple())
                moon_velocity_error = math.dist(moon_cached.velocity.as_tuple(), moon_direct.velocity.as_tuple())
                sun_position_error = math.dist(sun_cached.position.as_tuple(), sun_direct.position.as_tuple())
                orientation_cached = sample_moon_j2000_to_pa(self.table, seconds)
                orientation_direct = sample_moon_j2000_to_pa_direct(seconds)
                attitude_error = matrix_angle_error_rad(orientation_cached, orientation_direct)
                self.assertLessEqual(moon_position_error, 1.0)
                self.assertLessEqual(moon_velocity_error, 1.0e-5)
                self.assertLessEqual(sun_position_error, 10.0)
                self.assertLessEqual(attitude_error, 1.0e-5)

    def test_matched_fixed_step_propagation_agreement(self) -> None:
        cached_result = propagate(self.cached_config, "cached", truth=False, store_stride=12)
        direct_result = propagate(self.direct_fixed_config, "direct", truth=False, store_stride=12)
        self.assertLessEqual(abs(cached_result.perilune_altitude_km - direct_result.perilune_altitude_km), 100.0)
        self.assertLessEqual(abs(cached_result.stable_capture_duration_days - direct_result.stable_capture_duration_days), 0.05)
        self.assertLessEqual(abs(cached_result.final_mass_kg - direct_result.final_mass_kg), 0.1)

    def test_truth_dop853_convergence(self) -> None:
        nominal = replace(
            self.truth_config,
            physical_rtol=1.0e-10,
            physical_atol=1.0e-12,
        )
        tighter = replace(
            nominal,
            physical_rtol=1.0e-11,
            physical_atol=1.0e-13,
        )
        nominal_result = propagate(nominal, "nominal", truth=True, store_stride=12)
        tighter_result = propagate(tighter, "tighter", truth=True, store_stride=12)
        self.assertLessEqual(abs(nominal_result.perilune_altitude_km - tighter_result.perilune_altitude_km), 5.0)
        self.assertLessEqual(abs(nominal_result.stable_capture_duration_days - tighter_result.stable_capture_duration_days), 0.01)
        self.assertLessEqual(abs(nominal_result.final_mass_kg - tighter_result.final_mass_kg), 0.01)
        self.assertEqual(nominal_result.classification, tighter_result.classification)

    def test_third_body_terms_cancel_at_earth_center(self) -> None:
        absolute_seconds = 0.0
        origin = Vec3(0.0, 0.0, 0.0)
        moon_ephem = moon_state(absolute_seconds, self.direct_fixed_config)
        moon_position = Vec3(moon_ephem.position.x, moon_ephem.position.y, moon_ephem.position.z)
        moon_direct = _lunar_direct_acceleration(origin, moon_ephem, absolute_seconds, self.direct_fixed_config)
        moon_indirect = moon_position * (-MOON_MU / (moon_position.norm() ** 3))
        moon_total = moon_direct + moon_indirect
        self.assertLessEqual(moon_total.norm(), 1.0e-18)

        sun_ephem = sample_ephemeris_state_direct("sun", absolute_seconds)
        sun_position = Vec3(sun_ephem.position.x, sun_ephem.position.y, sun_ephem.position.z)
        sun_total = third_body_point_mass_acceleration(origin, sun_position, SUN_MU)
        self.assertLessEqual(sun_total.norm(), 1.0e-18)

    def test_third_body_matches_analytic_differential_form(self) -> None:
        absolute_seconds = 1.75 * SECONDS_PER_DAY
        position = Vec3(210_000.0, -52_000.0, 14_000.0)

        moon_ephem = moon_state(absolute_seconds, self.direct_fixed_config)
        moon_position = Vec3(moon_ephem.position.x, moon_ephem.position.y, moon_ephem.position.z)
        moon_expected = explicit_third_body_acceleration(position, moon_position, MOON_MU)
        moon_direct = _lunar_direct_acceleration(position, moon_ephem, absolute_seconds, self.direct_fixed_config)
        moon_indirect = moon_position * (-MOON_MU / (moon_position.norm() ** 3))
        moon_actual = moon_direct + moon_indirect
        self.assertLessEqual((moon_actual - moon_expected).norm(), 1.0e-18)

        sun_ephem = sample_ephemeris_state_direct("sun", absolute_seconds)
        sun_position = Vec3(sun_ephem.position.x, sun_ephem.position.y, sun_ephem.position.z)
        sun_expected = explicit_third_body_acceleration(position, sun_position, SUN_MU)
        sun_actual = third_body_point_mass_acceleration(position, sun_position, SUN_MU)
        self.assertLessEqual((sun_actual - sun_expected).norm(), 1.0e-18)

    def test_beam_gate_respects_geometry_and_range(self) -> None:
        config = build_laser_perigee_boost_config()
        station = config.beam_architecture.perigee_boost_station
        self.assertIsNotNone(station)
        moon_ephem = moon_state(0.0, config)
        moon_position = Vec3(moon_ephem.position.x, moon_ephem.position.y, moon_ephem.position.z)
        station_pos = station_position(0.0, station, moon_position)
        good_position = Vec3(20_000.0, 0.0, 0.0)
        bad_position = Vec3(250_000.0, 250_000.0, 0.0)
        self.assertGreater(_beam_gate_factor(good_position, station_pos, moon_position, station), 0.0)
        self.assertEqual(_beam_gate_factor(bad_position, station_pos, moon_position, station), 0.0)

    def test_beam_augmented_context_exceeds_solar_only_power_when_geometry_is_favorable(self) -> None:
        sep_only = build_sep_baseline_direct_capture_config()
        beam_config = replace(build_laser_perigee_boost_config(), sep_start_delay_days=0.0)
        state = build_initial_state(beam_config)
        moon_ephem = moon_state(20.0 * SECONDS_PER_DAY, beam_config)
        moon_position = Vec3(moon_ephem.position.x, moon_ephem.position.y, moon_ephem.position.z)
        sep_power, _ = thrust_context(20.0 * SECONDS_PER_DAY, state, sep_only, moon_position)
        beam_power, context = thrust_context(20.0 * SECONDS_PER_DAY, state, beam_config, moon_position)
        self.assertGreaterEqual(beam_power, sep_power)
        self.assertGreaterEqual(context["available_power_w"], sep_only.vehicle.onboard_solar_power_w)

    def test_truth_validation_config_promotes_fidelity(self) -> None:
        promoted = build_truth_validation_config(replace(self.cached_config, moon_harmonics_degree=4, moon_harmonics_order=4))
        self.assertEqual(promoted.ephemeris_mode, "spice_direct")
        self.assertEqual(promoted.physical_integrator, "dop853")
        self.assertGreaterEqual(promoted.earth_gravity_degree, 4)
        self.assertGreaterEqual(promoted.moon_harmonics_degree, 8)
        self.assertGreaterEqual(promoted.moon_harmonics_order, 8)

    def test_jax_surrogate_value_and_grad_are_finite(self) -> None:
        problem = build_jax_problem(replace(self.cached_config, duration_days=4.0, surrogate_step_seconds=7_200.0))
        objective, gradient = problem.value_and_grad_fn(problem.search_space.default_unit)
        self.assertTrue(math.isfinite(float(objective)))
        self.assertEqual(len(gradient), problem.search_space.dimension)
        self.assertTrue(all(math.isfinite(float(value)) for value in gradient))


if __name__ == "__main__":
    unittest.main()
