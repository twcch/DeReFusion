import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.analysis import watch_phase1_completed_intake as consumer


class IntakeConsumerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.phase = self.root / "reproduction/results/phase1"
        self.phase.mkdir(parents=True)
        self.source = self.phase / "source"
        self.source.mkdir()
        self.manifest = self.root / "manifest.csv"
        self.manifest.write_text("frozen", encoding="utf-8")
        self.output = self.phase / "consumer"
        mock = patch.object(consumer, "ROOT", self.root)
        mock.start()
        self.addCleanup(mock.stop)

    def receipt(self, name, status="completed_unreviewed"):
        directory = self.source / name
        directory.mkdir(exist_ok=True)
        raw = json.dumps({"status": status}).encode()
        (directory / "receipt.json").write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def test_only_completed_and_changed_receipt_blocks(self):
        digest = self.receipt("ready__attempt-01")
        self.receipt("running__attempt-01", "running")
        self.assertEqual(consumer.candidates(self.source, {}), {"ready__attempt-01": digest})
        self.assertEqual(consumer.candidates(self.source, {"ready__attempt-01": digest}), {})
        with self.assertRaisesRegex(RuntimeError, "changed"):
            consumer.candidates(self.source, {"ready__attempt-01": "bad"})

    def test_pass_acknowledgement_restores_without_mutating_source(self):
        digest = self.receipt("ready__attempt-01")
        consumed = consumer.prepare(self.source, self.manifest, self.output, "test")
        with patch.object(consumer, "intake", return_value={"status": "completed_descriptive_only"}) as run:
            result = consumer.consume_pass(self.source, self.manifest, self.output, "test", consumed)
            self.assertEqual(run.call_args.args[-1], {"ready__attempt-01"})
            self.assertEqual(result["receipt_hashes"], {"ready__attempt-01": digest})
            self.assertFalse(result["gpu_used"])
        self.assertEqual(consumer.prepare(self.source, self.manifest, self.output, "test"), consumed)
        self.assertEqual(consumer.candidates(self.source, {}), consumed)
        self.assertIsNone(consumer.consume_pass(self.source, self.manifest, self.output, "test", consumed))

    def test_wrong_root_or_identity_rejected(self):
        with self.assertRaises(ValueError):
            consumer.prepare(self.source, self.manifest, self.source / "bad", "test")
        consumer.prepare(self.source, self.manifest, self.output, "test")
        with self.assertRaises(FileExistsError):
            consumer.prepare(self.source, self.manifest, self.output, "different")

    def test_absent_source_waits(self):
        self.assertEqual(consumer.candidates(self.phase / "not_started", {}), {})


if __name__ == "__main__":
    unittest.main()
