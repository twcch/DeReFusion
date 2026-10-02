import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.batches import run_phase1_stages as runner
from reproduction.batches import write_stage_c_5060_authorization as writer
from reproduction.batches.write_stage_c_5060_authorization import build_authorization


class StageCAuthorizationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fingerprint = self.root / "environment.json"
        self.fingerprint.write_text(json.dumps({"mismatches": {}, "observed": {
            **runner.STAGE_C_LOCK, "cuda_available": True, "cudnn": 90701,
            "gpu_count": 1, "gpu_names": [runner.STAGE_C_GPU],
            "gpu_uuid": runner.STAGE_C_GPU_UUID, "driver_version": runner.STAGE_C_DRIVER,
        }}), encoding="utf-8")
        self.env_hash = runner.sha256(self.fingerprint)
        # Runtime fingerprints are intentionally untracked; register fixture bytes
        # as reviewed for these tests, preserving all production pin checks.
        for module in (runner, writer):
            registered_hash = patch.object(module, "STAGE_C_ENV_SHA256", self.env_hash)
            registered_hash.start()
            self.addCleanup(registered_hash.stop)
        self.path = self.root / "authorization.json"
        self.auth = build_authorization("user: explicit Stage C approval", "2026-09-29T08:25:00Z",
                                        "test-commit", self.fingerprint)

    def validate(self, **changes):
        auth = {**self.auth, **changes}
        self.path.write_text(json.dumps(auth), encoding="utf-8")
        with patch.object(runner.subprocess, "check_output", return_value="test-commit\n"):
            return runner.validate_authorization(
                self.path, self.fingerprint, {"C_confirmation"}, "cuda",
                runner.MANIFESTS["C_confirmation"], runner.STAGE_C_BATCH,
                runner.PHASE / runner.STAGE_C_BATCH)

    def test_complete_frozen_stage_c_authorization(self):
        auth, digest = self.validate()
        self.assertEqual(auth["execution_track"], runner.STAGE_C_TRACK)
        self.assertEqual(digest, runner.STAGE_C_ENV_SHA256)
        manifest = runner.rows(runner.MANIFESTS["C_confirmation"])
        self.assertEqual(len(manifest), 360)
        self.assertEqual({r["model"] for r in manifest}, {"revin-DLinear", "DeReFusion"})
        self.assertEqual({r["seed"] for r in manifest}, {"2022", "2023", "2024"})
        self.assertTrue(all("batch_id" not in row for row in manifest))

    def test_cannot_change_manifest_output_or_hardware(self):
        changes = [
            {"manifest_sha256": "0" * 64}, {"manifest_count": 359},
            {"output_root": str(runner.PHASE / "attempts")},
            {"gpu_uuid": "GPU-wrong"}, {"driver_version": "550.54.15"},
            {"gpu_model": "Quadro RTX 8000"}, {"cudnn": 0},
            {"environment_lock": {**runner.STAGE_C_LOCK, "torch": "2.5.1+cu121"}},
            {"execution_amendment_id": "wrong"}, {"stages": ["C_confirmation", "B_screen"]},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.validate(**change)

    def test_wrong_environment_bytes_and_commit_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "git_commit"):
            self.validate(git_commit="wrong")
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            self.validate(environment_fingerprint_sha256="0" * 64)

    def test_stage_c_authorization_cannot_launch_b(self):
        self.path.write_text(json.dumps(self.auth), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "limited"):
            runner.validate_authorization(self.path, self.fingerprint, {"B_screen"}, "cuda",
                                           runner.MANIFESTS["B_screen"], runner.STAGE_C_BATCH,
                                           runner.PHASE / runner.STAGE_C_BATCH)

    def test_b_authorization_cannot_launch_c_and_unknown_track_is_rejected(self):
        for track in ("nonconfirmatory_local_supplement", "nonconfirmatory_remote_gpu_supplement", "unknown"):
            with self.subTest(track=track), self.assertRaises(RuntimeError):
                self.validate(execution_track=track)

    def test_receipt_metadata_does_not_change_existing_tracks(self):
        self.assertEqual(runner.stage_c_receipt_metadata({"execution_track": "frozen_protocol"}), {})
        self.assertEqual(runner.stage_c_receipt_metadata({"execution_track": "nonconfirmatory_local_supplement"}), {})
        self.assertEqual(runner.stage_c_receipt_metadata(self.auth)["execution_amendment_id"], runner.STAGE_C_AMENDMENT)

    def test_isolated_cli_requires_full_queue_and_preserves_b_guard(self):
        valid = ["runner", "--stage", "C_confirmation", "--output-root",
                 str(runner.PHASE / runner.STAGE_C_BATCH), "--batch-id", runner.STAGE_C_BATCH,
                 "--run-label", runner.STAGE_C_RUN_LABEL]
        for extra in (["--limit", "1"], ["--models", "DeReFusion"], ["--manifest-path", "custom.csv"]):
            with self.subTest(extra=extra), patch("sys.argv", valid + extra), self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code, 2)
        with patch("sys.argv", ["runner", "--stage", "B_screen", "--output-root", "custom"]), self.assertRaises(SystemExit) as error:
            runner.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
