from __future__ import annotations

from contextlib import contextmanager
import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from aether_traj.campaign_manager import (  # noqa: E402
    cancel_job,
    create_campaign_from_payload,
    load_campaign,
    load_campaign_leaderboard,
    load_campaign_observations,
    pause_campaign,
    resume_campaign,
    run_next_job,
    validate_mutation_paths,
)
from aether_traj.models import WorkflowSpec  # noqa: E402


TEST_WORKFLOW_SPECS = {
    "test_scalar": WorkflowSpec(
        workflow_id="test_scalar",
        kind="scalar",
        runner_target="aether_traj.campaign_test_support:run_fake_workflow",
        profile_check_target="aether_traj.campaign_test_support:build_fake_profile",
        profiles=("baseline",),
        scripts={"baseline": "aether-test-scalar"},
    ),
    "test_scalar_alt": WorkflowSpec(
        workflow_id="test_scalar_alt",
        kind="scalar",
        runner_target="aether_traj.campaign_test_support:run_fake_alt_workflow",
        profile_check_target="aether_traj.campaign_test_support:build_fake_profile",
        profiles=("baseline",),
        scripts={"baseline": "aether-test-scalar-alt"},
    ),
}


@contextmanager
def patched_test_workflows():
    from aether_traj import campaign_manager, experiments, run_catalog

    scalar_ids = set(run_catalog.SCALAR_WORKFLOW_IDS) | set(TEST_WORKFLOW_SPECS)
    with (
        patch.object(campaign_manager, "WORKFLOW_SPECS", TEST_WORKFLOW_SPECS),
        patch.object(experiments, "WORKFLOW_SPECS", TEST_WORKFLOW_SPECS),
        patch.object(run_catalog, "SCALAR_WORKFLOW_IDS", scalar_ids),
    ):
        yield


class CampaignManagerTests(unittest.TestCase):
    def test_mutation_guardrails_block_core_python_paths(self) -> None:
        with self.assertRaises(ValueError):
            validate_mutation_paths(["src/aether_traj/dynamics.py"])

    def test_managed_campaign_run_emits_catalog_and_observations(self) -> None:
        with TemporaryDirectory() as tmpdir, patched_test_workflows():
            root = Path(tmpdir)
            payload = {
                "campaign_id": "autoresearch-smoke",
                "git_sha": "HEAD",
                "physics_model_id": "analytic_sep_v1",
                "design_space_id": "capstone_direct_capture_v1",
                "objective_set_id": "direct_capture_first",
                "workflows": ["test_scalar", "test_scalar_alt"],
                "profiles": ["baseline"],
                "replicates": 1,
                "truth_budget": 3,
                "max_parallel": 2,
                "base_config_overrides": {"departure_apogee_km": 420000.0},
            }

            created = create_campaign_from_payload(root, payload)
            self.assertEqual(created["planned_jobs"], 2)
            campaign_id = created["campaign_id"]

            first = run_next_job(root, campaign_id=campaign_id)
            second = run_next_job(root, campaign_id=campaign_id)
            self.assertEqual(first["status"], "completed")
            self.assertEqual(second["status"], "completed")

            campaign = load_campaign(root, campaign_id)
            self.assertEqual(campaign["completed_jobs"], 2)
            self.assertGreaterEqual(campaign["leaderboard_count"], 2)
            self.assertGreaterEqual(campaign["observation_count"], 1)

            leaderboard = load_campaign_leaderboard(root, campaign_id)
            observations = load_campaign_observations(root, campaign_id)
            self.assertTrue(any(row["classification"] == "direct_capture" for row in leaderboard))
            self.assertTrue(any(obs["kind"] == "consensus" for obs in observations))

            from aether_traj.run_catalog import load_run, load_runs

            managed_runs = load_runs(root)
            self.assertEqual(len(managed_runs), 2)
            self.assertTrue(all(run["campaign_id"] == campaign_id for run in managed_runs))

            detailed_run = load_run(root, managed_runs[0]["run_id"])
            self.assertTrue(detailed_run["managed"])
            self.assertEqual(detailed_run["design_space_id"], "capstone_direct_capture_v1")

            output_dir = Path(str(detailed_run["output_dir"]))
            self.assertTrue((output_dir / "normalized_summary.json").is_file())
            self.assertTrue((output_dir / "candidate_catalog.csv").is_file())
            self.assertTrue((output_dir / "observation_signals.json").is_file())

            with (output_dir / "candidate_catalog.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["departure_apogee_km"]), 420000.0)

    def test_pause_resume_and_cancel_pending_jobs(self) -> None:
        with TemporaryDirectory() as tmpdir, patched_test_workflows():
            root = Path(tmpdir)
            payload = {
                "campaign_id": "pause-smoke",
                "git_sha": "HEAD",
                "physics_model_id": "analytic_sep_v1",
                "design_space_id": "baseline_v1",
                "objective_set_id": "direct_capture_first",
                "workflows": ["test_scalar"],
                "profiles": ["baseline"],
                "replicates": 2,
                "truth_budget": 2,
                "max_parallel": 1,
            }

            created = create_campaign_from_payload(root, payload)
            campaign_id = created["campaign_id"]
            paused = pause_campaign(root, campaign_id)
            self.assertEqual(paused["status"], "paused")
            self.assertIsNone(run_next_job(root, campaign_id=campaign_id))

            resumed = resume_campaign(root, campaign_id)
            self.assertEqual(resumed["status"], "queued")

            job = run_next_job(root, campaign_id=campaign_id)
            self.assertEqual(job["status"], "completed")

            remaining = load_campaign(root, campaign_id)["jobs"]
            pending_job_id = next(job["job_id"] for job in remaining if job["status"] == "pending")
            cancelled = cancel_job(root, pending_job_id)
            self.assertEqual(cancelled["status"], "cancelled")

    def test_surrogate_truth_gap_observation_is_detected(self) -> None:
        with TemporaryDirectory() as tmpdir, patched_test_workflows():
            root = Path(tmpdir)
            payload = {
                "campaign_id": "gap-smoke",
                "git_sha": "HEAD",
                "physics_model_id": "analytic_sep_v1",
                "design_space_id": "gap_v1",
                "objective_set_id": "direct_capture_first",
                "workflows": ["test_scalar"],
                "profiles": ["baseline"],
                "replicates": 1,
                "truth_budget": 2,
                "max_parallel": 1,
                "base_config_overrides": {"departure_apogee_km": 320000.0},
            }

            created = create_campaign_from_payload(root, payload)
            campaign_id = created["campaign_id"]
            job = run_next_job(root, campaign_id=campaign_id)
            self.assertEqual(job["status"], "completed")

            observations = load_campaign_observations(root, campaign_id)
            self.assertTrue(any(obs["kind"] == "surrogate_truth_gap" for obs in observations))


if __name__ == "__main__":
    unittest.main()
