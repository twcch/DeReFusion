"""End-to-end proof that a Phase 1 fit survives being killed mid-training.

The unit tests in ``test_phase1_recovery.py`` prove the snapshot round-trips.
This script proves the thing the protocol actually needs, on a real fit driven
through ``run.py`` with the frozen Stage A command line:

    baseline   the fit run straight through, with no save point at all;
    interrupted the same fit, snapshotted per epoch, killed at an epoch
                boundary, then restarted and resumed from the snapshot.

The two must agree byte for byte on ``pred.npy`` and ``true.npy``.  That is
the whole claim: resuming continues the *same* trajectory rather than
producing a second, unrecorded experiment.  ``metrics.npy`` carries an
inference-time column that legitimately differs, so its error metrics are
compared as values, not bytes.

Usage (from the repository root, in the frozen environment):

    python tests/verify_phase1_recovery.py
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_NAME = "recovery.pt"

# The frozen Stage A A01 command line, with the run identity replaced so this
# verification can never collide with a real attempt package.
FROZEN_COMMAND = [
    "run.py", "--task_name", "long_term_forecast", "--is_training", "1",
    "--model_id", "{model_id}", "--model", "revin-DLinear", "--data", "custom",
    "--root_path", "./dataset/", "--data_path", "BOND10Y-2016-2025.csv",
    "--features", "MS", "--target", "Close", "--freq", "b",
    "--seq_len", "96", "--label_len", "48", "--pred_len", "24",
    "--enc_in", "4", "--dec_in", "4", "--c_out", "1", "--d_model", "32",
    "--n_heads", "8", "--e_layers", "2", "--d_layers", "1", "--d_ff", "2048",
    "--moving_avg", "25", "--factor", "1", "--dropout", "0.1", "--embed", "timeF",
    "--train_epochs", "{epochs}", "--batch_size", "32", "--patience", "100",
    "--learning_rate", "0.0001", "--lradj", "cosine", "--rand_seed", "2021",
    "--deterministic", "--num_workers", "0",
    "--des", "recovery_verification",
    "--result_log", "{result_log}",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_command(model_id: str, epochs: int, result_log: Path, recovery: bool,
                  device: str) -> list:
    command = [part.format(model_id=model_id, epochs=epochs, result_log=result_log)
               for part in FROZEN_COMMAND]
    command = [sys.executable] + command
    if device == "cpu":
        command.append("--no_use_gpu")
    if recovery:
        command.append("--enable_recovery")
    return command


def run_to_completion(command: list, log: Path) -> None:
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(
            "the fit exited with {}; see {}".format(result.returncode, log))


def wait_for_epoch(snapshot: Path, minimum: int, timeout: float) -> int:
    """Block until the save point reports `minimum` completed epochs."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if snapshot.is_file():
            payload = torch.load(snapshot, map_location="cpu", weights_only=False)
            if payload["next_epoch"] >= minimum:
                return payload["next_epoch"]
        time.sleep(0.05)
    raise RuntimeError(
        "no save point reached epoch {} within {}s; the fit is not "
        "snapshotting at all".format(minimum, timeout))


def only(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError("expected one {!r} under {}; found {}".format(
            pattern, directory, [m.name for m in matches]))
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--kill-after-epoch", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="verify the RNG stream the production device actually uses")
    args = parser.parse_args()

    if args.kill_after_epoch >= args.epochs:
        parser.error("--kill-after-epoch must leave epochs to resume")

    tag = "recoverycheck"
    model_id = "{}_{}".format(tag, int(time.time()))
    setting_glob = "*{}*".format(model_id)
    scratch = ROOT / "tests" / "_recovery_scratch"

    for stale in (ROOT / "results", ROOT / "checkpoints", ROOT / "test_results"):
        if stale.is_dir():
            for match in stale.glob(setting_glob):
                shutil.rmtree(match)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    checkpoint_dir = None
    try:
        # ---- 1. baseline: one clean pass, no save point -------------------
        run_to_completion(build_command(model_id, args.epochs, scratch / "baseline.log", False, args.device),
                          scratch / "baseline.stdout")
        baseline_results = only(ROOT / "results", setting_glob)
        checkpoint_dir = only(ROOT / "checkpoints", setting_glob)
        baseline_pred = digest(baseline_results / "pred.npy")
        baseline_true = digest(baseline_results / "true.npy")
        baseline_metrics = np.load(baseline_results / "metrics.npy")
        print("baseline   : {} epochs, pred {}...".format(args.epochs, baseline_pred[:12]))

        # ---- 2. interrupted: snapshot per epoch, then killed --------------
        shutil.rmtree(baseline_results)
        shutil.rmtree(checkpoint_dir)
        snapshot = checkpoint_dir / SNAPSHOT_NAME

        worker = subprocess.Popen(
            build_command(model_id, args.epochs, scratch / "interrupted.log", True, args.device),
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace")
        reached = wait_for_epoch(snapshot, args.kill_after_epoch, args.timeout_seconds)
        worker.kill()
        worker.wait()
        print("interrupted: killed with a save point at epoch {}".format(reached))

        if not snapshot.is_file():
            raise RuntimeError("the kill destroyed the save point; nothing was durable")

        # ---- 3. resumed: restart and continue from the save point ---------
        run_to_completion(build_command(model_id, args.epochs, scratch / "resumed.log", True, args.device),
                          scratch / "resumed.stdout")
        resumed_text = (scratch / "resumed.stdout").read_text(encoding="utf-8")
        if "resuming :" not in resumed_text:
            raise RuntimeError(
                "the restarted fit never resumed; it trained from scratch, so the "
                "save point is not being read back")
        resumed_results = only(ROOT / "results", setting_glob)
        print("resumed    : {}".format(
            [line for line in resumed_text.splitlines() if "resuming :" in line][0].strip()))

        if snapshot.exists():
            raise RuntimeError(
                "the save point outlived a completed fit; a later restart could "
                "resume a run that already finished")

        # ---- 4. the two fits must be the same experiment ------------------
        problems = []
        if digest(resumed_results / "pred.npy") != baseline_pred:
            problems.append("pred.npy differs from the uninterrupted fit")
        if digest(resumed_results / "true.npy") != baseline_true:
            problems.append("true.npy differs from the uninterrupted fit")

        resumed_metrics = np.load(resumed_results / "metrics.npy")
        error_columns = min(len(baseline_metrics), len(resumed_metrics), 5)
        for column in range(error_columns):
            if not np.allclose(baseline_metrics[column], resumed_metrics[column],
                               rtol=0, atol=0):
                problems.append(
                    "metric {} differs: {} vs {}".format(
                        column, baseline_metrics[column], resumed_metrics[column]))

        if problems:
            print("\nFAILED")
            for problem in problems:
                print("  - {}".format(problem))
            return 1

        print("\nOK: the interrupted-and-resumed fit is byte-identical to the fit "
              "that was never interrupted, on {} error metrics and both arrays."
              .format(error_columns))
        return 0
    finally:
        for stale in (ROOT / "results", ROOT / "checkpoints", ROOT / "test_results"):
            if stale.is_dir():
                for match in stale.glob(setting_glob):
                    shutil.rmtree(match, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
