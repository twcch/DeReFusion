"""Small CPU fixtures for immutable completed-package intake; no fitting/replay."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from reproduction.analysis import intake_phase1_completed_packages as intake
from reproduction.analysis.audit_phase1_checkpoint_forward import BoundInputs, file_hash


class CompletedPackageIntakeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "attempts"
        self.source.mkdir()
        self.row = {"protocol_version": "test-v1", "stage": "B_screen", "asset": "asset",
                    "model": "model", "horizon": "1", "seed": "2021", "origin": "fixed",
                    "split_mode": "ratio", "train_end": "", "val_end": "", "test_end": "",
                    "status": "planned", "block_reason": "", "batch_id": "preserve-me"}
        self.config = {"common": {"seq_len": 2}}
        self.split = {"split_manifest_id": "split", "prediction_keys_sha256": "keys", "window_count": "3"}
        self.data = self.root / "data.csv"
        self.data.write_text("frozen", encoding="utf-8")
        self.registry = {"asset": {"file": str(self.data), "sha256": file_hash(self.data)}}
        self.truth = np.asarray([1, 2, 3], dtype=np.float32).reshape(3, 1, 1)
        self.attempt, self.receipt = self.package(1)
        self.manifest = self.root / "manifest.csv"
        intake.write_csv(self.manifest, list(self.row), [self.row])
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        for name, value in (("ROOT", self.root), ("CONFIG", self.config_path),
                            ("REGISTRY", self.data), ("CONTRACT", self.data),
                            ("SPLITS", self.data), ("KEYS", self.data)):
            mock = patch.object(intake, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        for name, value in (("split_for", self.split), ("expected_truth", self.truth),
                            ("validate_preflight", (self.registry, {}, []))):
            mock = patch.object(intake, name, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)

    def package(self, ordinal, status="completed_unreviewed"):
        logical = intake.logical_id(self.row)
        attempt = self.source / f"{logical}__attempt-{ordinal:02d}"
        attempt.mkdir()
        pred = self.truth + np.float32(0.1)
        for name, array in (("pred.npy", pred), ("true.npy", self.truth),
                            ("metrics.npy", np.asarray(intake.metric(pred, self.truth)))):
            np.save(attempt / name, array, allow_pickle=False)
        (attempt / "checkpoint.pth").write_bytes(b"no-model-loading")
        (attempt / "run.log").write_text("retained", encoding="utf-8")
        receipt = {"status": status, "logical_run_id": logical, "attempt": ordinal, "attempt_id": attempt.name,
                   "protocol_version": self.row["protocol_version"], "stage": self.row["stage"],
                   "authorization_commit": "a" * 40, "authorization_record": "decision.md",
                   "environment_fingerprint_sha256": "b" * 64, "execution_device": "cuda",
                   "dataset_sha256": self.registry["asset"]["sha256"], **self.split,
                   "config_fingerprint": intake.canonical_hash({"row": self.row, "common": self.config["common"], "device": "cuda"}),
                   "sha256": {name: file_hash(attempt / name) for name in intake.ARTIFACTS},
                   "shapes": {"pred.npy": [3, 1, 1], "true.npy": [3, 1, 1], "metrics.npy": [6]}}
        (attempt / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        return attempt, receipt

    def rewrite(self):
        (self.attempt / "receipt.json").write_text(json.dumps(self.receipt), encoding="utf-8")

    def run_intake(self, output="fresh"):
        return intake.intake(self.source, self.manifest, self.root / output, "rtx8000")

    def test_cli_requires_explicit_paths_and_label(self):
        args = intake.parser().parse_args(["--source-root", "src", "--manifest", "m.csv",
                                           "--output-root", "fresh", "--source-label", "p4"])
        self.assertEqual(args.output_root, Path("fresh"))

    def test_success_preserves_manifest_fields_and_legacy_track(self):
        report = self.run_intake()
        self.assertEqual(report["status"], "completed_descriptive_only")
        selected = intake.rows(self.root / "fresh/selected-manifest.csv")
        self.assertEqual(selected, [self.row])
        reference = intake.rows(self.root / "fresh/reference-index.csv")[0]
        self.assertEqual((self.root / reference["p4_attempt"]).resolve(), self.attempt)
        self.assertEqual(reference["source_commit"], "a" * 40)
        self.assertTrue(report["input_immutability_verified"])
        self.assertEqual(report["attempts"][0]["identity"]["execution_track"], "frozen_protocol")
        self.assertFalse(report["pooling_authorized"])

    def test_running_and_interrupted_artifacts_are_never_read(self):
        for ordinal, status in ((2, "running"), (3, "interrupted")):
            attempt, _ = self.package(ordinal, status)
            (attempt / "checkpoint.pth").unlink()
        report = self.run_intake()
        self.assertEqual(report["summary"]["eligible_packages"], 1)
        self.assertEqual(report["summary"]["statuses"]["excluded"], 2)

    def test_hash_and_config_mismatches_are_retained(self):
        (self.attempt / "checkpoint.pth").write_bytes(b"changed")
        report = self.run_intake()
        self.assertEqual(report["status"], "requires_resolution")
        self.assertIn("hash mismatch", report["attempts"][0]["reason"])
        self.assertTrue(self.attempt.exists())
        self.receipt["config_fingerprint"] = "wrong"
        self.rewrite()
        report = self.run_intake("config-fresh")
        self.assertIn("config_fingerprint", report["attempts"][0]["reason"])

    def test_truth_and_metrics_are_independently_recomputed(self):
        for name in ("true.npy", "metrics.npy"):
            original = np.load(self.attempt / name)
            np.save(self.attempt / name, original + 1, allow_pickle=False)
            self.receipt["sha256"][name] = file_hash(self.attempt / name)
            self.rewrite()
            report = self.run_intake(name + "-fresh")
            self.assertEqual(report["status"], "requires_resolution")
            np.save(self.attempt / name, original, allow_pickle=False)
            self.receipt["sha256"][name] = file_hash(self.attempt / name)
            self.rewrite()

    def test_no_overwrite_or_source_writes(self):
        before = {str(path): file_hash(path) for path in self.source.rglob("*") if path.is_file()}
        self.run_intake()
        with self.assertRaises(FileExistsError):
            self.run_intake()
        with self.assertRaises(RuntimeError):
            intake.intake(self.source, self.manifest, self.source / "output", "p4")
        after = {str(path): file_hash(path) for path in self.source.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_duplicate_completed_attempts_are_preserved_without_arbitrary_selection(self):
        second, _ = self.package(2)
        report = self.run_intake()
        self.assertEqual(report["summary"]["replay_settings"], 0)
        self.assertTrue(self.attempt.exists() and second.exists())
        self.assertEqual(report["summary"]["statuses"]["requires-resolution"], 2)

    def test_failure_keeps_partial_report_and_rejects_input_mutation(self):
        with patch.object(BoundInputs, "verify", side_effect=RuntimeError("read-only input changed")):
            report = self.run_intake()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["input_immutability_verified"])
        self.assertTrue((self.root / "fresh/report.json").is_file())
        self.assertFalse((self.root / "fresh/reference-index.csv").exists())


if __name__ == "__main__":
    unittest.main()
