"""No network, SSH credentials, GPU, or training: small mock SFTP fixtures."""
import hashlib
import io
import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reproduction.analysis import receive_phase1_completed as receiver


class MockSFTP:
    def __init__(self, files):
        self.files = files
        self.reads = []
        self.mutate_on_artifact = False
        self.fail_once = False

    def lstat(self, path):
        if path in self.files:
            return SimpleNamespace(st_mode=stat.S_IFREG, st_size=len(self.files[path]))
        if any(name.startswith(path + "/") for name in self.files):
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_size=0)
        raise FileNotFoundError(path)

    def listdir_attr(self, path):
        names = {name[len(path) + 1:].split("/")[0] for name in self.files if name.startswith(path + "/")}
        return [SimpleNamespace(filename=name, st_mode=self.lstat(path + "/" + name).st_mode)
                for name in names]

    def open(self, path, mode):
        assert mode == "rb", "remote writes forbidden"
        self.reads.append(path)
        if path.endswith("checkpoint.pth") and self.fail_once:
            self.fail_once = False
            raise EOFError("transport dropped")
        if path.endswith("run.log") and self.mutate_on_artifact:
            marker = path.rsplit("/", 1)[0] + "/receipt.json"
            self.files[marker] += b" "
        return io.BytesIO(self.files[path])


class CompletedReceiverTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "rtx8000-intake-test"
        self.remote = "/results/batch"
        self.attempt = "B_screen_ASSET_MODEL_h1_s2021__attempt-01"
        self.files = {self.remote + "/" + name: b"{}" for name in receiver.REQUIRED_METADATA}
        self.receipt = {"status": "completed_unreviewed", "attempt_id": self.attempt,
                        "logical_run_id": self.attempt.split("__attempt-")[0], "attempt": 1,
                        "sha256": {}}
        for name in receiver.ARTIFACTS:
            payload = (name + " frozen bytes").encode()
            self.files[self.remote + "/" + self.attempt + "/" + name] = payload
            self.receipt["sha256"][name] = hashlib.sha256(payload).hexdigest()
        self.rewrite()
        self.sftp = MockSFTP(self.files)
        self.instance = receiver.Receiver(self.root, self.remote, progress=lambda value: None)

    def rewrite(self):
        self.files[self.remote + "/" + self.attempt + "/receipt.json"] = json.dumps(self.receipt).encode()

    def test_publish_six_original_files_and_cache_unchanged_receipt(self):
        report = self.instance.receive_pass(self.sftp)
        self.assertEqual(report["published"], [self.attempt])
        published = self.root / "attempts" / self.attempt
        self.assertEqual({p.name for p in published.iterdir()}, set(receiver.ARTIFACTS) | {"receipt.json"})
        self.assertFalse(report["pooling_authorized"])
        self.assertTrue(all((Path(report["metadata_snapshot"]) / n).exists() for n in receiver.REQUIRED_METADATA))
        self.sftp.reads.clear()
        report = self.instance.receive_pass(self.sftp)
        self.assertEqual(report["skipped"], [self.attempt])
        self.assertFalse(any(path.endswith("checkpoint.pth") for path in self.sftp.reads))

    def test_hash_mismatch_quarantines_and_never_publishes(self):
        self.files[self.remote + "/" + self.attempt + "/checkpoint.pth"] = b"corrupt"
        report = self.instance.receive_pass(self.sftp)
        self.assertEqual(report["status"], "requires_resolution")
        self.assertFalse((self.root / "attempts" / self.attempt).exists())
        self.assertTrue(list((self.root / "quarantine").iterdir()))

    def test_receipt_change_quarantines_complete_download(self):
        self.sftp.mutate_on_artifact = True
        report = self.instance.receive_pass(self.sftp)
        self.assertIn("receipt changed", report["issues"][0]["error"])
        self.assertFalse((self.root / "attempts" / self.attempt).exists())

    def test_network_retry_keeps_partial_and_uses_new_directory(self):
        self.sftp.fail_once = True
        with self.assertRaises(EOFError):
            self.instance.receive_pass(self.sftp)
        partial = list((self.root / "incoming").iterdir())
        self.assertEqual(len(partial), 1)
        self.assertEqual(self.instance.receive_pass(self.sftp)["published"], [self.attempt])
        self.assertTrue(partial[0].exists())

    def test_restart_verifies_full_checkpoint_and_never_overwrites(self):
        self.instance.receive_pass(self.sftp)
        checkpoint = self.root / "attempts" / self.attempt / "checkpoint.pth"
        checkpoint.write_bytes(b"local corruption")
        restarted = receiver.Receiver(self.root, self.remote, progress=lambda value: None)
        with self.assertRaises(receiver.IntegrityError):
            restarted.receive_pass(self.sftp)
        self.assertEqual(checkpoint.read_bytes(), b"local corruption")

    def test_running_receipt_does_not_read_artifacts(self):
        self.receipt["status"] = "running"
        self.rewrite()
        self.assertEqual(self.instance.receive_pass(self.sftp)["published"], [])
        self.assertFalse(any(path.endswith("checkpoint.pth") for path in self.sftp.reads))

    def test_reject_existing_unowned_roots_and_traversal(self):
        with self.assertRaises(FileExistsError):
            receiver.Receiver(Path(self.temp.name), self.remote)
        with self.assertRaises(ValueError):
            receiver.Receiver(self.root / "bad", "/results/../secret")
        with self.assertRaises(ValueError):
            receiver.Receiver(receiver.ROOT / "dataset/intake", self.remote)
        for name in ("../checkpoint.pth", "/absolute", "bad\\name"):
            with self.assertRaises(receiver.IntegrityError):
                receiver.safe_name(name)

    def test_missing_authorization_prevents_publish(self):
        del self.files[self.remote + "/authorization.json"]
        with self.assertRaises(receiver.IntegrityError):
            self.instance.receive_pass(self.sftp)
        self.assertFalse(list((self.root / "attempts").iterdir()))

    def test_empty_precreated_dedicated_root_is_owned_without_overwrite(self):
        fresh = Path(self.temp.name) / "precreated-intake"
        fresh.mkdir()
        instance = receiver.Receiver(fresh, self.remote, progress=lambda value: None)
        self.assertTrue((fresh / receiver.MARKER).is_file())
        self.assertEqual(instance.receive_pass(self.sftp)["published"], [self.attempt])

    def test_deadline_retains_partial_without_publishing(self):
        self.instance.deadline = 0
        with self.assertRaises(TimeoutError):
            self.instance.receive_pass(self.sftp)
        self.assertFalse(list((self.root / "attempts").iterdir()))
        self.assertTrue(list((self.root / "incoming").iterdir()))


if __name__ == "__main__":
    unittest.main()
