import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.batches.run_phase1_stages import (
    CONFIG,
    MANIFESTS,
    build_command,
    rows,
    sha256,
    validate_authorization,
    validate_preflight,
)


class Phase1StageRunnerTest(unittest.TestCase):
    def test_frozen_manifests_and_preflight_keys(self):
        expected = {"B_screen": 300, "C_confirmation": 360, "D_temporal_robustness": 144}
        for stage, count in expected.items():
            manifest = rows(MANIFESTS[stage])
            self.assertEqual(len(manifest), count)
            _, _, problems = validate_preflight(manifest)
            self.assertEqual(problems, [])

    def test_date_stage_command_preserves_explicit_boundaries(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        row = next(r for r in rows(MANIFESTS["D_temporal_robustness"]) if r["asset"] == "AAPL")
        registry = {r["asset"]: r for r in rows(CONFIG.parents[1] / "results/dataset_registry.csv")}
        command = build_command(row, config["common"], registry[row["asset"]], 1, "cuda")
        for value in ("--split_mode", "dates", "--train_end", row["train_end"],
                      "--val_end", row["val_end"], "--test_end", row["test_end"], "--deterministic"):
            self.assertIn(value, command)
        self.assertIn("--gpu_type", command)

    def test_supplemental_auth_is_bound_to_manifest_environment_and_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            manifest.write_text("batch_id\nlocal-5060-bridge-v1\n", encoding="utf-8")
            fingerprint = root / "environment.json"
            fingerprint.write_text(json.dumps({"mismatches": {}, "observed": {
                "python": "3.11.15", "torch": "2.7.1+cu128", "torch_cuda": "12.8",
                "numpy": "2.1.2", "pandas": "2.3.3", "scikit_learn": "1.7.2",
                "cuda_available": True, "gpu_names": ["NVIDIA GeForce RTX 5060 Ti"],
            }}), encoding="utf-8")
            output = root / "fresh-output"
            authorization = root / "authorization.json"
            authorization.write_text(json.dumps({
                "authorization_version": "phase1-local-supplement-auth/v1",
                "protocol_version": "phase1-v1.1-2026-09-19",
                "execution_track": "nonconfirmatory_local_supplement",
                "stages": ["B_screen"], "execution_device": "cuda",
                "approved_by": "test", "approved_at_utc": "test", "decision_record": "test",
                "git_commit": "test-commit", "environment_fingerprint_sha256": sha256(fingerprint),
                "manifest_sha256": sha256(manifest), "output_root": str(output.resolve()),
                "batch_id": "local-5060-bridge-v1", "pooling_authorized": False,
                "gpu_model": "NVIDIA GeForce RTX 5060 Ti",
            }), encoding="utf-8")
            actual_commit = unittest.mock.Mock()
            actual_commit.return_value = "test-commit\n"
            with patch("reproduction.batches.run_phase1_stages.subprocess.check_output", actual_commit):
                auth, env_hash = validate_authorization(
                    authorization, fingerprint, {"B_screen"}, "cuda", manifest,
                    "local-5060-bridge-v1", output)
            self.assertEqual(auth["batch_id"], "local-5060-bridge-v1")
            self.assertEqual(env_hash, sha256(fingerprint))
            manifest.write_text("batch_id\nchanged\n", encoding="utf-8")
            with patch("reproduction.batches.run_phase1_stages.subprocess.check_output", actual_commit):
                with self.assertRaisesRegex(RuntimeError, "manifest hash"):
                    validate_authorization(authorization, fingerprint, {"B_screen"}, "cuda", manifest,
                                           "local-5060-bridge-v1", output)

    def test_rtx8000_supplement_is_bound_to_frozen_stack_and_gpu_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            manifest.write_text("batch_id\nrtx8000-timesnet-remaining-v1\n", encoding="utf-8")
            fingerprint = root / "environment.json"
            fingerprint.write_text(json.dumps({"mismatches": {}, "observed": {
                "python": "3.11.15", "torch": "2.5.1+cu121", "torch_cuda": "12.1",
                "numpy": "2.1.2", "pandas": "2.3.3", "scikit_learn": "1.7.2",
                "cuda_available": True, "gpu_names": ["Quadro RTX 8000"],
                "gpu_uuid": "GPU-test",
            }}), encoding="utf-8")
            output = root / "fresh-output"
            authorization = root / "authorization.json"
            lock = {"python": "3.11.15", "torch": "2.5.1+cu121", "torch_cuda": "12.1",
                    "numpy": "2.1.2", "pandas": "2.3.3", "scikit_learn": "1.7.2"}
            authorization.write_text(json.dumps({
                "authorization_version": "phase1-remote-supplement-auth/v1",
                "protocol_version": "phase1-v1.1-2026-09-19",
                "execution_track": "nonconfirmatory_remote_gpu_supplement",
                "stages": ["B_screen"], "execution_device": "cuda",
                "approved_by": "test", "approved_at_utc": "test", "decision_record": "test",
                "git_commit": "test-commit", "environment_fingerprint_sha256": sha256(fingerprint),
                "manifest_sha256": sha256(manifest), "output_root": str(output.resolve()),
                "batch_id": "rtx8000-timesnet-remaining-v1", "pooling_authorized": False,
                "environment_lock": lock, "gpu_model": "Quadro RTX 8000", "gpu_uuid": "GPU-test",
            }), encoding="utf-8")
            actual_commit = unittest.mock.Mock()
            actual_commit.return_value = "test-commit\n"
            with patch("reproduction.batches.run_phase1_stages.subprocess.check_output", actual_commit):
                auth, env_hash = validate_authorization(
                    authorization, fingerprint, {"B_screen"}, "cuda", manifest,
                    "rtx8000-timesnet-remaining-v1", output)
            self.assertEqual(auth["gpu_uuid"], "GPU-test")
            self.assertEqual(env_hash, sha256(fingerprint))
            auth["environment_lock"]["python"] = "3.10.16"
            authorization.write_text(json.dumps(auth), encoding="utf-8")
            with patch("reproduction.batches.run_phase1_stages.subprocess.check_output", actual_commit):
                with self.assertRaisesRegex(RuntimeError, "environment lock"):
                    validate_authorization(authorization, fingerprint, {"B_screen"}, "cuda", manifest,
                                           "rtx8000-timesnet-remaining-v1", output)


if __name__ == "__main__":
    unittest.main()
