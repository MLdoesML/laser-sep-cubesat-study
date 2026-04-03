from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from aether_traj.config import build_sep_run_profile  # noqa: E402
from aether_traj.experiments import WORKFLOW_SPECS, aggregate_run_manifests, expected_script_targets, load_target  # noqa: E402
from aether_traj.sep_de_workflow import run_sep_de_workflow  # noqa: E402
from aether_traj.sep_jax_workflow import run_sep_jax_workflow  # noqa: E402
from aether_traj.sep_lbfgs_workflow import run_sep_lbfgs_workflow  # noqa: E402
from aether_traj.sep_pso_workflow import run_sep_pso_workflow  # noqa: E402


class WorkflowMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        cls.project_scripts = cls.pyproject["project"]["scripts"]
        cls.expected_scripts = expected_script_targets()
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_pyproject_scripts_match_registry(self) -> None:
        self.assertEqual(self.expected_scripts, self.project_scripts)

    def test_all_script_targets_import(self) -> None:
        for script_name, target in self.project_scripts.items():
            with self.subTest(script=script_name):
                load_target(target)

    def test_all_declared_profiles_build(self) -> None:
        for workflow_id, spec in WORKFLOW_SPECS.items():
            checker = load_target(spec.profile_check_target)
            for profile in spec.profiles:
                with self.subTest(workflow=workflow_id, profile=profile):
                    checker(profile)

    def test_readme_mentions_all_scripts(self) -> None:
        for script_name in self.expected_scripts:
            with self.subTest(script=script_name):
                self.assertIn(script_name, self.readme)

    def test_aggregate_manifest_falls_back_to_manifest_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "smoke-sep-baseline"
            run_dir.mkdir()
            manifest = {
                "run_id": "smoke-sep-baseline",
                "workflow_id": "sep_jax",
                "profile": "sep_baseline_direct_capture",
                "script_name": "aether-study-jax",
                "git_sha": "deadbeef",
                "summary_filename": "missing_summary.json",
                "classification": "escape",
                "capture_duration_days": 0.0,
                "perilune_altitude_km": 4500.0,
            }
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            rows = aggregate_run_manifests(Path(tmpdir))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["classification"], "escape")
        self.assertEqual(rows[0]["capture_duration_days"], 0.0)

    def test_scalar_optimizer_runs_complete_end_to_end(self) -> None:
        small_profile = build_sep_run_profile("sep_baseline_direct_capture")
        small_profile = replace(
            small_profile,
            base_config=replace(small_profile.base_config, duration_days=1.0, surrogate_step_seconds=7200.0, truth_step_seconds=3600.0),
            candidate_count=4,
            truth_candidate_count=2,
            iterations=4,
        )
        workflows = {
            "sep_jax": run_sep_jax_workflow,
            "sep_lbfgs": run_sep_lbfgs_workflow,
            "sep_de": run_sep_de_workflow,
            "sep_pso": run_sep_pso_workflow,
        }
        with TemporaryDirectory() as tmpdir, patch("aether_traj.optimizer_workflow.build_sep_run_profile", return_value=small_profile):
            for workflow_id, runner in workflows.items():
                with self.subTest(workflow=workflow_id):
                    output_dir = Path(tmpdir) / workflow_id
                    runner("sep_baseline_direct_capture", output_dir)
                    self.assertTrue((output_dir / f"{workflow_id}_summary.json").is_file())
                    self.assertTrue((output_dir / f"{workflow_id}_validation_candidates.csv").is_file())
                    self.assertTrue((output_dir / "run_manifest.json").is_file())
                    summary = json.loads((output_dir / f"{workflow_id}_summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(summary["workflow_id"], workflow_id)
                    self.assertIn(summary["backend"], {"jax_adam", "jax_lbfgs", "jax_de", "jax_pso"})


if __name__ == "__main__":
    unittest.main()
