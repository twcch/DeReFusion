"""Run and package the frozen Phase 1 calibration grid.

The default is a dry run. Use ``--execute`` to train. Calibration metrics are
retained only to validate the execution chain and must not be used as research
evidence or for model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "reproduction/configs/phase1_modern_baselines.json"
MANIFEST_PATH = ROOT / "reproduction/results/phase1/A_calibration.manifest.csv"
REGISTRY_PATH = ROOT / "reproduction/results/dataset_registry.csv"
PACKAGE_ROOT = ROOT / "reproduction/results/phase1/calibration"
LOG_ROOT = ROOT / "reproduction/logs/phase1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_id(index: int, row: dict[str, str]) -> str:
    return (
        f"A{index:02d}_{row['asset']}_{row['model']}_"
        f"h{row['horizon']}_s{row['seed']}"
    )


def attempt_directory(logical_run_id: str, attempt: int) -> Path:
    return PACKAGE_ROOT / f"{logical_run_id}__attempt-{attempt:02d}"


def read_verified_receipt(destination: Path) -> dict | None:
    receipt_path = destination / "receipt.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for name, expected in receipt.get("sha256", {}).items():
        path = destination / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"existing receipt hash mismatch: {path}")
    return receipt


def existing_attempts(logical_run_id: str) -> list[tuple[int, Path, dict]]:
    found: list[tuple[int, Path, dict]] = []
    legacy = PACKAGE_ROOT / logical_run_id
    receipt = read_verified_receipt(legacy)
    if receipt is not None:
        found.append((int(receipt.get("attempt", 1)), legacy, receipt))
    for path in sorted(PACKAGE_ROOT.glob(f"{logical_run_id}__attempt-*")):
        match = re.fullmatch(re.escape(logical_run_id) + r"__attempt-(\d+)", path.name)
        receipt = read_verified_receipt(path)
        if match and receipt is not None:
            attempt = int(receipt.get("attempt", int(match.group(1))))
            found.append((attempt, path, receipt))
    return sorted(found, key=lambda item: item[0])


def choose_attempt(logical_run_id: str, retry_authorization: str | None) -> tuple[int, Path, str | None]:
    attempts = existing_attempts(logical_run_id)
    successes = [item for item in attempts if item[2].get("status") == "calibration_pass"]
    if len(successes) > 1:
        raise RuntimeError(f"multiple successful attempts for {logical_run_id}; fail-stop")
    if successes:
        return 0, successes[0][1], None
    if not attempts:
        return 1, attempt_directory(logical_run_id, 1), None
    if not retry_authorization:
        return -1, attempts[-1][1], None
    previous_attempt, previous_path, previous_receipt = attempts[-1]
    if previous_receipt.get("status") not in {
        "resource_blocked", "failed", "timed_out", "artifact_invalid", "infrastructure_interruption"
    }:
        raise RuntimeError(
            f"{logical_run_id} status {previous_receipt.get('status')!r} is not retryable"
        )
    if previous_attempt >= 2:
        raise RuntimeError(f"{logical_run_id} requires escalation approval for attempt 3+")
    return previous_attempt + 1, attempt_directory(logical_run_id, previous_attempt + 1), previous_path.name


def config_fingerprint(row: dict[str, str], config: dict, attempt: int, device: str) -> str:
    payload = {"row": row, "common": config["common"], "attempt": attempt, "device": device}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def build_command(
    row: dict[str, str], registry: dict[str, dict[str, str]], config: dict, attempt: int,
    device: str,
) -> tuple[list[str], str, str]:
    common = config["common"]
    data_file = Path(registry[row["asset"]]["file"])
    fingerprint = config_fingerprint(row, config, attempt, device)
    model_id = f"phase1A_{row['asset']}_h{row['horizon']}_s{row['seed']}__attempt-{attempt:02d}_{fingerprint}"
    description = f"phase1A_calibration_{fingerprint}"
    command = [
        sys.executable,
        "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", model_id,
        "--model", row["model"],
        "--data", "custom",
        "--root_path", str(data_file.parent).replace("\\", "/") + "/",
        "--data_path", data_file.name,
        "--features", "MS",
        "--target", "Close",
        "--freq", "b",
        "--seq_len", str(common["seq_len"]),
        "--label_len", str(common["label_len"]),
        "--pred_len", row["horizon"],
        "--enc_in", str(common["enc_in"]),
        "--dec_in", str(common["dec_in"]),
        "--c_out", str(common["c_out"]),
        "--d_model", str(common["d_model"]),
        "--n_heads", str(common["n_heads"]),
        "--e_layers", str(common["e_layers"]),
        "--d_layers", str(common["d_layers"]),
        "--d_ff", str(common["d_ff"]),
        "--moving_avg", str(common["moving_avg"]),
        "--factor", str(common["factor"]),
        "--dropout", str(common["dropout"]),
        "--embed", common["embed"],
        "--train_epochs", str(common["train_epochs"]),
        "--batch_size", str(common["batch_size"]),
        "--patience", str(common["patience"]),
        "--learning_rate", str(common["learning_rate"]),
        "--lradj", common["lradj"],
        "--rand_seed", row["seed"],
        "--num_workers", "0",
        "--des", description,
        "--result_log", "reproduction/logs/phase1/calibration_result_log.txt",
    ]
    if device == "cpu":
        command.append("--no_use_gpu")
    else:
        command.extend(["--gpu_type", "cuda", "--gpu", "0"])
    if common.get("deterministic"):
        command.append("--deterministic")
    return command, model_id, fingerprint


def find_setting(row: dict[str, str], model_id: str) -> str:
    candidates = [
        path for path in (ROOT / "results").glob(f"*{model_id}*")
        if path.is_dir() and f"_{row['model']}_custom_" in path.name
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one result directory for {model_id}/{row['model']}, found {len(candidates)}")
    return candidates[0].name


def parse_diagnostics(log_text: str) -> dict[str, float | int | None]:
    lines = [line for line in log_text.splitlines() if line.startswith("mse:")]
    if not lines:
        raise RuntimeError("run log has no final metric line")
    line = lines[-1]

    def number(pattern: str, cast=float):
        match = re.search(pattern, line)
        return cast(match.group(1)) if match else None

    return {
        "train_time_seconds": number(r"train_time:([0-9.]+)s"),
        "inference_ms_per_sample": number(r"inference_time:([0-9.]+)ms/sample"),
        "parameter_count": number(r"params:([0-9]+)", int),
        "gpu_peak_memory_mb": number(r"gpu_mem_peak_mb:([0-9.]+)"),
    }


def package_run(
    index: int,
    row: dict[str, str],
    registry: dict[str, dict[str, str]],
    config: dict,
    command: list[str],
    log_path: Path,
    git_commit: str,
    destination: Path,
    logical_run_id: str,
    attempt: int,
    supersedes_attempt: str | None,
    retry_authorization: str | None,
    fingerprint: str,
    model_id: str,
    requested_device: str,
) -> Path:
    setting = find_setting(row, model_id)
    result_dir = ROOT / "results" / setting
    checkpoint = ROOT / "checkpoints" / setting / "checkpoint.pth"
    required = {
        "pred.npy": result_dir / "pred.npy",
        "true.npy": result_dir / "true.npy",
        "metrics.npy": result_dir / "metrics.npy",
        "checkpoint.pth": checkpoint,
        "run.log": log_path,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError("missing run artifacts: " + ", ".join(missing))

    pred = np.load(required["pred.npy"])
    true = np.load(required["true.npy"])
    metrics = np.load(required["metrics.npy"])
    expected_shape = (pred.shape[0], int(row["horizon"]), 1)
    if pred.shape != expected_shape or true.shape != expected_shape:
        raise RuntimeError(f"unexpected pred/true shapes: {pred.shape} / {true.shape}")
    if metrics.shape != (6,):
        raise RuntimeError(f"unexpected metrics shape: {metrics.shape}")
    if not all(np.isfinite(array).all() for array in (pred, true, metrics)):
        raise RuntimeError("non-finite value in calibration arrays")

    if destination.exists() and read_verified_receipt(destination) is None:
        raise RuntimeError(f"refusing to overwrite incomplete destination: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in required.items():
        shutil.copy2(source, destination / name)

    copied_hashes = {name: sha256(destination / name) for name in required}
    environment = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "device": requested_device,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    receipt = {
        "logical_run_id": logical_run_id,
        "attempt": attempt,
        "attempt_id": destination.name,
        "supersedes_attempt": supersedes_attempt,
        "retry_reason": "prior_technical_failure" if supersedes_attempt else None,
        "retry_authorization": retry_authorization,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": fingerprint,
        "status": "calibration_pass",
        "scientific_use": "none; execution calibration only",
        "protocol_version": config["protocol_version"],
        "git_commit": git_commit,
        "asset": row["asset"],
        "dataset_sha256": registry[row["asset"]]["sha256"],
        "model": row["model"],
        "horizon": int(row["horizon"]),
        "seed": int(row["seed"]),
        "split_mode": row["split_mode"],
        "command": subprocess.list2cmdline(command),
        "environment": environment,
        "shapes": {
            "pred.npy": list(pred.shape),
            "true.npy": list(true.shape),
            "metrics.npy": list(metrics.shape),
        },
        "all_arrays_finite": True,
        "sha256": copied_hashes,
        "diagnostics": parse_diagnostics(log_path.read_text(encoding="utf-8", errors="replace")),
    }
    (destination / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def validate_source_state() -> str:
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked:
        raise RuntimeError("tracked worktree changes must be committed before calibration")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def resource_blocked_models() -> set[str]:
    blocked = set()
    for receipt_path in PACKAGE_ROOT.glob("*/receipt.json"):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") == "resource_blocked":
            blocked.add(receipt["model"])
    return blocked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="run training; default is dry-run")
    parser.add_argument("--limit", type=int, default=None, help="maximum number of missing fits to run")
    parser.add_argument("--only-asset", choices=["BOND10Y", "BONDETF", "CITICSEC"])
    parser.add_argument("--only-model")
    parser.add_argument("--exclude-model", action="append", default=[])
    parser.add_argument("--retry-blocked-model", action="append", default=[],
                        help="explicitly override a model-family resource pause")
    parser.add_argument("--retry-authorization",
                        help="required authorization identifier for a retry attempt")
    parser.add_argument("--timeout-seconds", type=int, default=300,
                        help="hard wall-clock limit per calibration fit")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="execution device; CUDA must pass the environment preflight")
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = load_csv(MANIFEST_PATH)
    registry = {row["asset"]: row for row in load_csv(REGISTRY_PATH)}
    blocked_models = resource_blocked_models()
    if len(rows) != 15:
        raise RuntimeError(f"expected 15 calibration rows, found {len(rows)}")

    selected: list[tuple[int, dict[str, str], Path, int, str | None]] = []
    for index, row in enumerate(rows, start=1):
        if args.only_asset and row["asset"] != args.only_asset:
            continue
        if args.only_model and row["model"] != args.only_model:
            continue
        if row["model"] in set(args.exclude_model):
            continue
        if row["model"] in blocked_models and row["model"] not in set(args.retry_blocked_model):
            print(f"PAUSE family-blocked {run_id(index, row)}")
            continue
        data_path = ROOT / registry[row["asset"]]["file"]
        if sha256(data_path) != registry[row["asset"]]["sha256"]:
            raise RuntimeError(f"dataset hash mismatch: {row['asset']}")
        logical_run_id = run_id(index, row)
        attempt, destination, supersedes = choose_attempt(logical_run_id, args.retry_authorization)
        if attempt == 0:
            print(f"SKIP verified-success {destination.name}")
            continue
        if attempt < 0:
            print(f"PAUSE retry-authorization-required {logical_run_id}")
            continue
        selected.append((index, row, destination, attempt, supersedes))

    if args.limit is not None:
        selected = selected[:args.limit]
    print(f"missing calibration fits selected: {len(selected)}")
    for index, row, destination, attempt, _ in selected:
        print("DRY" if not args.execute else "RUN", destination.name, f"attempt={attempt}")
    if not args.execute:
        return 0

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    git_commit = validate_source_state()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for index, row, destination, attempt, supersedes in selected:
        identifier = run_id(index, row)
        command, model_id, fingerprint = build_command(
            row, registry, config, attempt, args.device
        )
        log_path = LOG_ROOT / f"{destination.name}.log"
        print(f"START {destination.name}")
        started = time.time()
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w", encoding="utf-8", newline="") as log:
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=args.timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"{identifier} exceeded {args.timeout_seconds}s and was terminated; see {log_path}"
                ) from error
        if completed.returncode != 0:
            raise RuntimeError(f"{identifier} failed with exit code {completed.returncode}; see {log_path}")
        destination = package_run(
            index, row, registry, config, command, log_path, git_commit, destination,
            identifier, attempt, supersedes, args.retry_authorization, fingerprint, model_id,
            args.device,
        )
        print(f"PASS {destination.name} elapsed={time.time() - started:.1f}s package={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
