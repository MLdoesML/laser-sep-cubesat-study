from __future__ import annotations

from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ScaffoldTests(unittest.TestCase):
    def test_core_docs_exist(self) -> None:
        for relative in (
            "README.md",
            "docs/STUDY_PLAN.md",
            "docs/mission_definition.md",
            "docs/publication_positioning.md",
            "docs/experiment_matrix.md",
            "docs/review_questions.md",
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
            (ROOT / "configs/profiles/laser_dual_window_codesign.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["beam_architecture"], "dual_window")
        self.assertTrue(payload["beam_roles"]["perigee_boost"])
        self.assertTrue(payload["beam_roles"]["lunar_brake"])


if __name__ == "__main__":
    unittest.main()
