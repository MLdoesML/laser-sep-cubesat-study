from __future__ import annotations

from pathlib import Path
import sys
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from aether_traj.config import config_for_profile, load_vehicle_config


class ScaffoldTests(unittest.TestCase):
    def test_core_docs_exist(self) -> None:
        for relative in (
            "README.md",
            "docs/STUDY_PLAN.md",
            "docs/mission_definition.md",
            "docs/publication_positioning.md",
            "docs/experiment_matrix.md",
            "docs/review_questions.md",
            "campaign.toml",
            "manager_program.md",
        ):
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).exists())

    def test_vehicle_config_loads(self) -> None:
        payload = tomllib.loads(
            (ROOT / "configs/vehicles/capstone_class_demo_v1.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["name"], "capstone_class_demo_v1")
        self.assertEqual(payload["mass"]["wet_mass_kg"], 25.0)
        self.assertEqual(payload["power"]["propulsion_power_cap_w"], 600.0)

    def test_profile_config_loads(self) -> None:
        payload = tomllib.loads(
            (ROOT / "configs/profiles/laser_dual_window_fixed.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["beam_architecture"], "dual_window_fixed")
        self.assertTrue(payload["beam_roles"]["perigee_boost"])
        self.assertTrue(payload["beam_roles"]["lunar_brake"])

    def test_vehicle_loader_builds_public_interface(self) -> None:
        vehicle = load_vehicle_config(ROOT / "configs/vehicles/capstone_class_demo_v1.toml")
        self.assertEqual(vehicle.name, "capstone_class_demo_v1")
        self.assertEqual(vehicle.minimum_final_mass_kg, 20.0)

    def test_declared_profiles_build_configs(self) -> None:
        for profile in (
            "sep_baseline_direct_capture",
            "laser_perigee_boost",
            "laser_lunar_brake",
            "laser_dual_window_fixed",
        ):
            with self.subTest(profile=profile):
                config = config_for_profile(profile)
                self.assertEqual(config.name, profile)
                self.assertEqual(config.vehicle.name, "capstone_class_demo_v1")
                self.assertGreater(config.duration_days, 0.0)


if __name__ == "__main__":
    unittest.main()
