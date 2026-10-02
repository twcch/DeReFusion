"""The save-point contract for long Phase 1 fits.

A resumed fit is only admissible if it continues the *same* trajectory.  The
central assertion here is therefore not "the snapshot round-trips" but "the
process that resumes draws exactly what the process that was never
interrupted would have drawn next" -- a save point that passes every other
test while failing that one would silently turn one fit into two.

Run from the repository root:

    python -m unittest discover -s tests -p "test_phase1_recovery.py"
"""

import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils import recovery
from utils.tools import EarlyStopping


class RngContinuityTest(unittest.TestCase):
    """The part of the save point that decides whether resuming is honest."""

    def _advance_every_stream(self):
        random.random()
        np.random.rand()
        torch.rand(4)
        if torch.cuda.is_available():
            torch.rand(4, device="cuda")

    def _draw_every_stream(self):
        drawn = [random.random(), float(np.random.rand()), torch.rand(4)]
        if torch.cuda.is_available():
            drawn.append(torch.rand(4, device="cuda"))
        return drawn

    def test_resume_draws_what_an_uninterrupted_run_would_have_drawn(self):
        seed = 2021

        # The uninterrupted run: three epochs' worth of draws, then the epoch
        # that a crash would have interrupted.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for _ in range(3):
            self._advance_every_stream()
        expected = self._draw_every_stream()

        # The interrupted run: the same three epochs, snapshotted at the epoch
        # boundary, then killed before it draws again.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for _ in range(3):
            self._advance_every_stream()
        snapshot = recovery.capture_rng()

        # A fresh process restores the snapshot and continues.
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        self._advance_every_stream()  # work the fresh process does before resuming
        recovery.restore_rng(snapshot)
        resumed = self._draw_every_stream()

        self.assertEqual(resumed[0], expected[0], "python stream diverged")
        self.assertEqual(resumed[1], expected[1], "numpy stream diverged")
        self.assertTrue(torch.equal(resumed[2], expected[2]), "torch stream diverged")
        if torch.cuda.is_available():
            self.assertTrue(torch.equal(resumed[3], expected[3]), "cuda stream diverged")


class SnapshotFileTest(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.path = Path(self._directory.name) / recovery.SNAPSHOT_NAME

    def tearDown(self):
        self._directory.cleanup()

    def test_a_published_snapshot_reads_back_under_its_own_identity(self):
        payload = {
            "snapshot_version": recovery.SNAPSHOT_VERSION,
            "identity": "setting-A",
            "next_epoch": 7,
            "elapsed_seconds": 123.5,
            "optimizer": {"lr": 0.0001},
            "rng": recovery.capture_rng(),
        }
        recovery.write_snapshot(str(self.path), payload)
        read_back = recovery.read_snapshot(str(self.path), "setting-A")
        self.assertEqual(read_back["next_epoch"], 7)
        self.assertEqual(read_back["elapsed_seconds"], 123.5)
        self.assertTrue(torch.equal(read_back["rng"]["torch"], payload["rng"]["torch"]))

    def test_a_missing_snapshot_is_absence_not_an_error(self):
        self.assertIsNone(recovery.read_snapshot(str(self.path), "setting-A"))

    def test_another_runs_snapshot_is_refused_rather_than_replayed(self):
        """Resuming into a mismatched configuration would be a second experiment."""
        recovery.write_snapshot(str(self.path), {
            "snapshot_version": recovery.SNAPSHOT_VERSION,
            "identity": "setting-A", "next_epoch": 1, "elapsed_seconds": 0.0,
            "rng": recovery.capture_rng(),
        })
        with self.assertRaises(RuntimeError) as raised:
            recovery.read_snapshot(str(self.path), "setting-B")
        self.assertIn("different run", str(raised.exception))

    def test_an_unknown_snapshot_version_is_refused(self):
        recovery.write_snapshot(str(self.path), {
            "snapshot_version": recovery.SNAPSHOT_VERSION + 1,
            "identity": "setting-A", "next_epoch": 1, "elapsed_seconds": 0.0,
            "rng": recovery.capture_rng(),
        })
        with self.assertRaises(RuntimeError) as raised:
            recovery.read_snapshot(str(self.path), "setting-A")
        self.assertIn("version", str(raised.exception))

    def test_discarding_is_idempotent(self):
        recovery.write_snapshot(str(self.path), {
            "snapshot_version": recovery.SNAPSHOT_VERSION,
            "identity": "setting-A", "next_epoch": 1, "elapsed_seconds": 0.0,
            "rng": recovery.capture_rng(),
        })
        recovery.discard_snapshot(str(self.path))
        recovery.discard_snapshot(str(self.path))
        self.assertFalse(os.path.exists(str(self.path)))

    def test_no_temporary_file_survives_a_publish(self):
        recovery.write_snapshot(str(self.path), {
            "snapshot_version": recovery.SNAPSHOT_VERSION,
            "identity": "setting-A", "next_epoch": 1, "elapsed_seconds": 0.0,
            "rng": recovery.capture_rng(),
        })
        self.assertEqual(sorted(os.listdir(self._directory.name)), [recovery.SNAPSHOT_NAME])


class EarlyStoppingStateTest(unittest.TestCase):
    def test_the_stopping_decision_survives_a_round_trip(self):
        """`counter` and `best_score` are what decide when the fit ends."""
        stopping = EarlyStopping(patience=5)

        class _Model:
            def state_dict(self):
                return {}

        with tempfile.TemporaryDirectory() as directory:
            stopping(1.0, _Model(), directory)   # first epoch: becomes the best
            stopping(1.5, _Model(), directory)   # worse -> counter 1
            stopping(2.0, _Model(), directory)   # worse -> counter 2

        state = stopping.state_dict()
        self.assertEqual(state["counter"], 2)

        resumed = EarlyStopping(patience=5)
        resumed.load_state_dict(state)
        self.assertEqual(resumed.counter, 2)
        self.assertEqual(resumed.best_score, stopping.best_score)
        self.assertFalse(resumed.early_stop)

        # Replaying the same losses from here must stop at the same epoch.
        with tempfile.TemporaryDirectory() as directory:
            for loss in (2.5, 3.0, 3.5):
                resumed(loss, _Model(), directory)
        self.assertTrue(resumed.early_stop)


if __name__ == "__main__":
    unittest.main()
