"""Shared, fail-closed launcher for Phase 1 stages B/C/D.

Dry-run is the default. Training requires an explicit, versioned authorization
record and a matching worker environment fingerprint. Attempts are immutable.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "reproduction/results/phase1"
CONFIG = ROOT / "reproduction/configs/phase1_modern_baselines.json"
REGISTRY = ROOT / "reproduction/results/dataset_registry.csv"
CONTRACT = PHASE / "phase1_data_contract.csv"
SPLITS = PHASE / "phase1_split_manifest.csv"
KEYS = PHASE / "phase1_prediction_keys.csv.gz"
MANIFESTS = {
    "B_screen": PHASE / "B_screen.manifest.csv",
    "C_confirmation": PHASE / "C_confirmation.manifest.csv",
    "D_temporal_robustness": PHASE / "D_temporal_robustness.manifest.csv",
}
OUTPUT = PHASE / "attempts"
STAGE_C_TRACK = "confirmatory_local_stage_c"
STAGE_C_AUTH_VERSION = "phase1-local-stage-c-auth/v1"
STAGE_C_AMENDMENT = "phase1-stage-c-parallel-5060-20260929"
STAGE_C_BATCH = "local-5060-stage-c-v1"
STAGE_C_RUN_LABEL = "stagec_5060_v1"
STAGE_C_MANIFEST_SHA256 = "bda784f0dd816939cdfd02e30109369df6a3e2e76aecd79aa8ee87b3910e6bfa"
STAGE_C_ENV_SHA256 = "76b3b80aae7b1addcbc75088331d5d186feb07233b56d323d78bb1e88633c755"
STAGE_C_LOCK = {"python": "3.11.15", "torch": "2.7.1+cu128", "torch_cuda": "12.8",
                "numpy": "2.1.2", "pandas": "2.3.3", "scikit_learn": "1.7.2"}
STAGE_C_GPU = "NVIDIA GeForce RTX 5060 Ti"
STAGE_C_GPU_UUID = "GPU-8b312e09-ea91-9e3a-c5d3-06b9362181b7"
STAGE_C_DRIVER = "610.62"


def stage_c_receipt_metadata(auth: dict) -> dict:
    """Add new-track provenance without changing historical receipt schemas."""
    if auth.get("execution_track") != STAGE_C_TRACK:
        return {}
    return {"authorization_version": auth["authorization_version"],
            "execution_amendment_id": auth["execution_amendment_id"],
            "manifest_sha256": auth["manifest_sha256"], "output_root": auth["output_root"],
            "gpu_model": auth["gpu_model"], "gpu_uuid": auth["gpu_uuid"],
            "driver_version": auth["driver_version"], "cudnn": auth["cudnn"]}


def atomic_json(path: Path, value: dict) -> None:
    """Publish one durable JSON commit marker; never expose a partial receipt."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_copy(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    if target.exists():
        raise RuntimeError(f"refusing to overwrite packaged artifact: {target}")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    with source.open("rb") as incoming, temporary.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    os.replace(temporary, target)


def run_worker(command: list[str], log, env: dict, timeout: float) -> tuple[int, bool]:
    """End the worker on budget expiry, allowing its recovery handler to flush."""
    worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=os.name != "nt")
    try:
        return worker.wait(timeout=timeout), False
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        if os.name == "nt":
            worker.terminate()
        else:
            os.killpg(worker.pid, signal.SIGTERM)
        try:
            worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                worker.kill()
            else:
                os.killpg(worker.pid, signal.SIGKILL)
            worker.wait()
        if isinstance(error, KeyboardInterrupt):
            raise
        return worker.returncode, True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def logical_id(row: dict[str, str]) -> str:
    origin = f"_o{row['origin']}" if row["stage"] == "D_temporal_robustness" else ""
    return f"{row['stage']}_{row['asset']}_{row['model']}_h{row['horizon']}_s{row['seed']}{origin}"


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_preflight(selected: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], list[str]]:
    issues: list[str] = []
    registry = {r["asset"]: r for r in rows(REGISTRY)}
    contract = {r["series_group_id"]: r for r in rows(CONTRACT)}
    split_rows = rows(SPLITS)
    split_index = {(r["asset"], r["forecast_horizon"], r["split_mode"], r["train_end"], r["val_end"], r["test_end"]): r for r in split_rows}
    if len(split_index) != len(split_rows):
        issues.append("split manifest has duplicate split identities")
    data_ids: dict[str, dict[str, str]] = {}
    for asset in sorted({r["asset"] for r in selected}):
        reg = registry.get(asset)
        con = contract.get(asset)
        if not reg or not con:
            issues.append(f"{asset}: missing registry or data-contract record")
            continue
        data_path = ROOT / reg["file"]
        if not data_path.is_file() or sha256(data_path) != reg["sha256"] or con["sha256"] != reg["sha256"]:
            issues.append(f"{asset}: source bytes do not match both frozen registry and contract")
        if con["status"] != "eligible":
            issues.append(f"{asset}: data contract status is {con['status']}")
        data_ids[asset] = reg
    for row in selected:
        key = (row["asset"], row["horizon"], row["split_mode"], row["train_end"], row["val_end"], row["test_end"])
        split = split_index.get(key)
        if not split:
            issues.append(f"{logical_id(row)}: no exact split-manifest match")
            continue
        if row["stage"] not in split["stage_scope"].split(";"):
            issues.append(f"{logical_id(row)}: stage not included in split scope")
        if int(split["prediction_point_count"]) != int(split["window_count"]) * int(row["horizon"]):
            issues.append(f"{logical_id(row)}: prediction-key cardinality mismatch")
        if split["split_manifest_id"] not in {x["split_manifest_id"] for x in split_rows}:
            issues.append(f"{logical_id(row)}: unknown split id")
    if not KEYS.is_file():
        issues.append("precomputed prediction-key file is missing")
    elif not SPLITS.is_file() or not CONTRACT.is_file():
        issues.append("preflight artifacts are incomplete")
    else:
        needed = {v["split_manifest_id"]: v for v in split_index.values()
                  if any((r["asset"], r["horizon"], r["split_mode"], r["train_end"], r["val_end"], r["test_end"])
                         == (v["asset"], v["forecast_horizon"], v["split_mode"], v["train_end"], v["val_end"], v["test_end"])
                         for r in selected)}
        digests = {split_id: hashlib.sha256() for split_id in needed}
        counts = {split_id: 0 for split_id in needed}
        with gzip.open(KEYS, "rt", encoding="utf-8", newline="") as stream:
            for key in csv.DictReader(stream):
                split_id = key["split_manifest_id"]
                if split_id in needed:
                    digests[split_id].update((key["key_hash"] + "\n").encode("ascii"))
                    counts[split_id] += 1
        for split_id, split in needed.items():
            if counts[split_id] != int(split["prediction_point_count"]):
                issues.append(f"{split_id}: key row count differs from split manifest")
            if digests[split_id].hexdigest() != split["prediction_keys_sha256"]:
                issues.append(f"{split_id}: key hash differs from split manifest")
    return data_ids, split_index, issues


def build_command(row: dict[str, str], common: dict, registry_row: dict[str, str], attempt: int, device: str,
                  run_label: str = "phase1_frozen") -> list[str]:
    data = ROOT / registry_row["file"]
    model_id = f"{logical_id(row)}__attempt-{attempt:02d}"
    cmd = [sys.executable, "run.py", "--task_name", "long_term_forecast", "--is_training", "1",
           "--model_id", model_id, "--model", row["model"], "--data", "custom",
           "--root_path", data.parent.as_posix() + "/", "--data_path", data.name,
           "--features", "MS", "--target", "Close", "--freq", "b",
           "--seq_len", str(common["seq_len"]), "--label_len", str(common["label_len"]),
           "--pred_len", row["horizon"], "--enc_in", str(common["enc_in"]),
           "--dec_in", str(common["dec_in"]), "--c_out", str(common["c_out"]),
           "--d_model", str(common["d_model"]), "--n_heads", str(common["n_heads"]),
           "--e_layers", str(common["e_layers"]), "--d_layers", str(common["d_layers"]),
           "--d_ff", str(common["d_ff"]), "--moving_avg", str(common["moving_avg"]),
           "--factor", str(common["factor"]), "--dropout", str(common["dropout"]),
           "--embed", common["embed"], "--train_epochs", str(common["train_epochs"]),
           "--batch_size", str(common["batch_size"]), "--patience", str(common["patience"]),
           "--learning_rate", str(common["learning_rate"]), "--lradj", common["lradj"],
           "--rand_seed", row["seed"], "--num_workers", "0", "--des", run_label,
           "--result_log", f"reproduction/logs/phase1/{row['stage']}.txt"]
    if row["split_mode"] == "dates":
        cmd += ["--split_mode", "dates", "--train_end", row["train_end"],
                "--val_end", row["val_end"], "--test_end", row["test_end"]]
    if common.get("deterministic"):
        cmd.append("--deterministic")
    if device == "cuda":
        cmd += ["--gpu_type", "cuda", "--gpu", "0"]
    else:
        cmd.append("--no_use_gpu")
    return cmd


def validate_authorization(path: Path, env_path: Path, stage_set: set[str], device: str,
                           manifest_path: Path, batch_id: str | None,
                           output_root: Path) -> tuple[dict, str]:
    authorization = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = json.loads(env_path.read_text(encoding="utf-8"))
    track = authorization.get("execution_track", "frozen_protocol")
    if track == STAGE_C_TRACK:
        expected = {"authorization_version": STAGE_C_AUTH_VERSION,
                    "protocol_version": "phase1-v1.1-2026-09-19",
                    "execution_track": STAGE_C_TRACK,
                    "execution_amendment_id": STAGE_C_AMENDMENT,
                    "run_label": STAGE_C_RUN_LABEL, "manifest_count": 360,
                    "gpu_model": STAGE_C_GPU, "gpu_uuid": STAGE_C_GPU_UUID,
                    "driver_version": STAGE_C_DRIVER, "cudnn": 90701,
                    "environment_lock": STAGE_C_LOCK,
                    "cross_stack_pooling_authorized": False}
        if stage_set != {"C_confirmation"} or authorization.get("stages") != ["C_confirmation"]:
            raise RuntimeError("Stage C authorization is limited to the complete C_confirmation stage")
        if device != "cuda":
            raise RuntimeError("Stage C local authorization requires CUDA")
        if batch_id != STAGE_C_BATCH or authorization.get("batch_id") != STAGE_C_BATCH:
            raise RuntimeError("Stage C batch_id does not match the approved isolated batch")
        if output_root.resolve() != (PHASE / STAGE_C_BATCH).resolve():
            raise RuntimeError("Stage C output root must be its approved isolated direct child")
        if authorization.get("output_root") != str(output_root.resolve()):
            raise RuntimeError("Stage C output root does not match authorization")
        if (sha256(manifest_path) != STAGE_C_MANIFEST_SHA256
                or authorization.get("manifest_sha256") != STAGE_C_MANIFEST_SHA256
                or len(rows(manifest_path)) != 360):
            raise RuntimeError("Stage C requires the full frozen 360-row manifest hash")
        if sha256(env_path) != STAGE_C_ENV_SHA256:
            raise RuntimeError("Stage C environment fingerprint differs from the reviewed local fingerprint")
        pinned = STAGE_C_LOCK
    elif track == "nonconfirmatory_local_supplement":
        expected = {"authorization_version": "phase1-local-supplement-auth/v1",
                    "protocol_version": "phase1-v1.1-2026-09-19",
                    "execution_track": "nonconfirmatory_local_supplement"}
        if stage_set != {"B_screen"}:
            raise RuntimeError("local supplemental authorization is limited to B_screen rows")
        if not batch_id or authorization.get("batch_id") != batch_id:
            raise RuntimeError("supplemental batch_id does not match the authorization")
        if authorization.get("manifest_sha256") != sha256(manifest_path):
            raise RuntimeError("supplemental manifest hash does not match the authorization")
        if authorization.get("output_root") != str(output_root.resolve()):
            raise RuntimeError("supplemental output root does not match the authorization")
        if authorization.get("pooling_authorized") is not False:
            raise RuntimeError("local supplemental results must explicitly prohibit pooling")
        pinned = {"python": "3.11.15", "torch": "2.7.1+cu128", "torch_cuda": "12.8",
                  "numpy": "2.1.2", "pandas": "2.3.3", "scikit_learn": "1.7.2"}
    elif track == "nonconfirmatory_remote_gpu_supplement":
        expected = {"authorization_version": "phase1-remote-supplement-auth/v1",
                    "protocol_version": "phase1-v1.1-2026-09-19",
                    "execution_track": "nonconfirmatory_remote_gpu_supplement"}
        if stage_set != {"B_screen"}:
            raise RuntimeError("remote GPU supplemental authorization is limited to B_screen rows")
        if not batch_id or authorization.get("batch_id") != batch_id:
            raise RuntimeError("remote supplemental batch_id does not match the authorization")
        if authorization.get("manifest_sha256") != sha256(manifest_path):
            raise RuntimeError("remote supplemental manifest hash does not match the authorization")
        if authorization.get("output_root") != str(output_root.resolve()):
            raise RuntimeError("remote supplemental output root does not match the authorization")
        if authorization.get("pooling_authorized") is not False:
            raise RuntimeError("remote supplemental results must explicitly prohibit pooling")
        pinned = {"python": "3.11.15", "torch": "2.5.1+cu121", "torch_cuda": "12.1",
                  "numpy": "2.1.2", "pandas": "2.3.3", "scikit_learn": "1.7.2"}
        if authorization.get("environment_lock") != pinned:
            raise RuntimeError("remote supplemental authorization must repeat the reviewed environment lock")
    elif track == "frozen_protocol":
        expected = {"authorization_version": "phase1-execution-auth/v1",
                    "protocol_version": "phase1-v1.1-2026-09-19"}
        pinned = {"python": "3.11.15", "torch": "2.5.1+cu121", "torch_cuda": "12.1",
                  "numpy": "2.1.2", "pandas": "2.3.3", "scikit_learn": "1.7.2"}
    else:
        raise RuntimeError(f"unknown explicit execution_track: {track!r}")
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise RuntimeError(f"authorization {key} must equal {value!r}")
    if not stage_set or stage_set - set(authorization.get("stages", [])):
        raise RuntimeError("authorization does not explicitly cover every selected stage")
    if authorization.get("execution_device") != device:
        raise RuntimeError("authorization execution_device does not match requested device")
    if not authorization.get("approved_by") or not authorization.get("approved_at_utc") or not authorization.get("decision_record"):
        raise RuntimeError("authorization must identify approver, time, and decision record")
    expected_commit = authorization.get("git_commit")
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if expected_commit != actual_commit:
        raise RuntimeError("authorization git_commit does not match current source")
    env_hash = sha256(env_path)
    if authorization.get("environment_fingerprint_sha256") != env_hash:
        raise RuntimeError("worker environment fingerprint does not match authorization")
    observed = fingerprint.get("observed", fingerprint)
    if fingerprint.get("mismatches", {}) or any(observed.get(k) != v for k, v in pinned.items()):
        raise RuntimeError("worker environment does not match the frozen Python/PyTorch lock")
    if device == "cuda":
        if observed.get("cuda_available") is not True:
            raise RuntimeError("CUDA is not available in the environment fingerprint")
        if authorization.get("gpu_model") not in observed.get("gpu_names", []):
            raise RuntimeError("authorized GPU model does not match worker fingerprint")
        if track == STAGE_C_TRACK:
            gpu_lock = {"gpu_uuid": STAGE_C_GPU_UUID, "driver_version": STAGE_C_DRIVER,
                        "cudnn": 90701, "gpu_count": 1, "gpu_names": [STAGE_C_GPU]}
            if any(observed.get(k) != v for k, v in gpu_lock.items()):
                raise RuntimeError("Stage C GPU UUID, driver, cuDNN or device count differs from the reviewed fingerprint")
        if track == "nonconfirmatory_local_supplement" and authorization["gpu_model"] != "NVIDIA GeForce RTX 5060 Ti":
            raise RuntimeError("local supplement may run only on the fingerprinted RTX 5060 Ti")
        if track == "nonconfirmatory_remote_gpu_supplement":
            if authorization["gpu_model"] != "Quadro RTX 8000":
                raise RuntimeError("remote supplemental batch is pinned to the approved Quadro RTX 8000")
            if authorization.get("gpu_uuid") != observed.get("gpu_uuid"):
                raise RuntimeError("authorized GPU UUID does not match worker fingerprint")
    return authorization, env_hash


def split_for(row: dict[str, str]) -> dict[str, str]:
    for split in rows(SPLITS):
        if (split["asset"], split["forecast_horizon"], split["split_mode"], split["train_end"], split["val_end"], split["test_end"]) == (
            row["asset"], row["horizon"], row["split_mode"], row["train_end"], row["val_end"], row["test_end"]
        ):
            return split
    raise RuntimeError(f"no split record for {logical_id(row)}")


def expected_truth(row: dict[str, str], split: dict[str, str], common: dict, data_path: Path) -> np.ndarray:
    frame = pd.read_csv(data_path)
    dates = pd.to_datetime(frame["date"], errors="raise")
    if row["split_mode"] == "ratio":
        train_stop = int(len(frame) * 0.7)
        test_rows = int(len(frame) * 0.2)
        test_start, test_stop = len(frame) - test_rows, len(frame)
    else:
        train_stop = int(dates.searchsorted(pd.Timestamp(row["train_end"]), side="left"))
        test_start = int(dates.searchsorted(pd.Timestamp(row["val_end"]), side="left"))
        test_stop = int(dates.searchsorted(pd.Timestamp(row["test_end"]), side="left"))
    ordered = frame[["Open", "High", "Low", "Close"]].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(ordered[:train_stop])
    scaled_close = scaler.transform(ordered)[:, -1]
    seq_len, horizon = int(common["seq_len"]), int(row["horizon"])
    first_target = test_start
    windows = test_stop - test_start - horizon + 1
    if windows != int(split["window_count"]):
        raise RuntimeError(f"window count mismatch for {logical_id(row)}")
    expected = np.stack([scaled_close[first_target + i:first_target + i + horizon] for i in range(windows)])
    return expected[..., None]


def package_attempt(row: dict[str, str], destination: Path, command: list[str], log_path: Path,
                    setting: str, auth: dict, env_hash: str, common: dict, registry_row: dict[str, str]) -> None:
    result_dir = ROOT / "results" / setting
    checkpoint = ROOT / "checkpoints" / setting / "checkpoint.pth"
    source = {"pred.npy": result_dir / "pred.npy", "true.npy": result_dir / "true.npy",
              "metrics.npy": result_dir / "metrics.npy", "checkpoint.pth": checkpoint, "run.log": log_path}
    missing = [str(p) for p in source.values() if not p.is_file()]
    if missing:
        raise RuntimeError("required output missing; attempt retained as incomplete: " + ", ".join(missing))
    if not destination.is_dir():
        raise RuntimeError(f"attempt directory is missing: {destination}")
    running_receipt = destination / "receipt.json"
    if not running_receipt.is_file() or json.loads(running_receipt.read_text(encoding="utf-8")).get("status") != "running":
        raise RuntimeError(f"attempt is not in the expected running state: {destination}")
    split = split_for(row)
    pred, true, metrics = (np.load(source[n], allow_pickle=False) for n in ("pred.npy", "true.npy", "metrics.npy"))
    wanted = expected_truth(row, split, common, ROOT / registry_row["file"])
    if pred.shape != wanted.shape or true.shape != wanted.shape:
        raise RuntimeError(f"prediction/target shape does not cover frozen key universe: {pred.shape}, {true.shape}, expected {wanted.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(true).all() or not np.isfinite(metrics).all():
        raise RuntimeError("non-finite output; attempt retained and flagged")
    if not np.allclose(true, wanted, rtol=2e-5, atol=2e-6):
        raise RuntimeError("true.npy does not align with frozen test targets; attempt retained and flagged")
    for name, path in source.items():
        atomic_copy(path, destination / name)
    hashes = {name: sha256(destination / name) for name in source}
    receipt = {"logical_run_id": logical_id(row), "attempt": 1, "attempt_id": destination.name,
               "protocol_version": row["protocol_version"], "stage": row["stage"], "status": "completed_unreviewed",
               "execution_track": auth.get("execution_track", "frozen_protocol"),
               "batch_id": auth.get("batch_id"),
               "scientific_use": "not interpreted; pending intake and stage gate review",
               "authorization_record": auth.get("decision_record"), "authorization_commit": auth["git_commit"],
               "environment_fingerprint_sha256": env_hash, "dataset_sha256": registry_row["sha256"],
               "split_manifest_id": split["split_manifest_id"], "prediction_keys_sha256": split["prediction_keys_sha256"],
               "execution_device": command_device(command),
               "config_fingerprint": canonical_hash({"row": row, "common": common, "device": command_device(command)}),
               "command": subprocess.list2cmdline(command), "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "shapes": {"pred.npy": list(pred.shape), "true.npy": list(true.shape), "metrics.npy": list(metrics.shape)},
               "sha256": hashes, **stage_c_receipt_metadata(auth)}
    atomic_json(running_receipt, receipt)


def command_device(command: list[str]) -> str:
    return "cuda" if "--gpu_type" in command else "cpu"


def interrupted_attempt(path: Path) -> bool:
    """True for an attempt the budget stopped mid-fit.

    This is deliberately narrower than "not completed": a genuinely failed fit
    must still block, because continuing one is a retry and needs its own
    authorization.  A budget stop is not a failure of the fit -- the same
    attempt simply has more to do -- so it continues in place, creating no
    second attempt and needing no new authorization.
    """
    receipt = path / "receipt.json"
    if not receipt.is_file():
        return False
    try:
        return json.loads(receipt.read_text(encoding="utf-8")).get("status") == "interrupted"
    except (json.JSONDecodeError, OSError):
        return False


def completed_attempt(path: Path, row: dict[str, str], split: dict[str, str], expected_config: str) -> bool:
    receipt_path = path / "receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"attempt exists without receipt; manual evidence review required: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "completed_unreviewed":
        raise RuntimeError(f"attempt is {receipt.get('status')!r}, not resumable: {path}")
    if receipt.get("logical_run_id") != logical_id(row) or receipt.get("protocol_version") != row["protocol_version"]:
        raise RuntimeError(f"attempt identity does not match the frozen manifest: {path}")
    if receipt.get("split_manifest_id") != split["split_manifest_id"] or receipt.get("config_fingerprint") != expected_config:
        raise RuntimeError(f"attempt split/config differs from current frozen inputs: {path}")
    if receipt.get("prediction_keys_sha256") != split["prediction_keys_sha256"]:
        raise RuntimeError(f"attempt prediction keys differ from current frozen inputs: {path}")
    for name, expected in receipt.get("sha256", {}).items():
        artifact = path / name
        if not artifact.is_file() or sha256(artifact) != expected:
            raise RuntimeError(f"completed attempt has missing or changed artifact: {artifact}")
    required = {"pred.npy", "true.npy", "metrics.npy", "checkpoint.pth", "run.log"}
    if set(receipt.get("sha256", {})) != required:
        raise RuntimeError(f"attempt artifact inventory is incomplete: {path}")
    return True


def main() -> int:
    global OUTPUT, MANIFESTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", choices=list(MANIFESTS), required=True)
    parser.add_argument("--manifest-path", type=Path,
                        help="separate frozen manifest (only allowed with a nonconfirmatory supplemental authorization)")
    parser.add_argument("--output-root", type=Path,
                        help="isolated attempt root; never defaults away from the frozen protocol output")
    parser.add_argument("--batch-id", help="immutable identifier for a separately authorized supplemental batch")
    parser.add_argument("--run-label", default="phase1_frozen",
                        help="run.py description label, useful to keep supplemental result directories distinct")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true", help="training is blocked without a formal authorization")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--environment-fingerprint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-hours", type=float, help="monotonic wall-time budget; gracefully stop the active fit at the deadline")
    parser.add_argument("--fit-timeout-hours", type=float, default=2.0)
    parser.add_argument("--shutdown-buffer-minutes", type=float, default=5.0)
    parser.add_argument("--enable-recovery", action="store_true", help="save worker recovery snapshots at completed epoch boundaries")
    parser.add_argument("--models", help="comma-separated model filter; lets a bounded budget finish the cheap fits first")
    parser.add_argument("--resume-interrupted", action="store_true",
                        help="continue an attempt the budget stopped, from its own save point")
    parser.add_argument("--stop-below-seconds", type=float, default=60.0,
                        help="do not start another fit when less than this remains in the budget; "
                             "set it above a TimesNet fit's runtime so a long fit is never started "
                             "with no room to finish")
    args = parser.parse_args()
    stage_c_isolated = (set(args.stage) == {"C_confirmation"}
                        and bool(args.output_root or args.batch_id or args.run_label != "phase1_frozen"))
    supplemental = bool(args.manifest_path or args.output_root or args.batch_id or args.run_label != "phase1_frozen")
    if stage_c_isolated:
        if (args.manifest_path or not args.output_root or args.batch_id != STAGE_C_BATCH
                or args.run_label != STAGE_C_RUN_LABEL
                or args.output_root.resolve() != (PHASE / STAGE_C_BATCH).resolve()):
            parser.error("isolated Stage C requires its frozen manifest, local-5060-stage-c-v1 root/batch, and stagec_5060_v1 label")
        if args.models or args.limit is not None:
            parser.error("isolated Stage C must retain the full 360-fit queue; --models and --limit are forbidden")
    elif supplemental and (not args.manifest_path or not args.output_root or not args.batch_id):
        parser.error("supplemental runs require --manifest-path, --output-root, and --batch-id together")
    if args.manifest_path and set(args.stage) != {"B_screen"}:
        parser.error("custom manifests are limited to B_screen")
    if args.manifest_path:
        MANIFESTS = dict(MANIFESTS, B_screen=args.manifest_path)
    if supplemental:
        OUTPUT = args.output_root.resolve()
        if OUTPUT.parent != PHASE.resolve():
            parser.error("supplemental output root must be a new direct child of the Phase 1 output directory")
    if args.resume_interrupted and not args.enable_recovery:
        parser.error("--resume-interrupted needs --enable-recovery, or the worker retrains from scratch")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.fit_timeout_hours <= 0 or args.shutdown_buffer_minutes < 0:
        parser.error("fit timeout must be positive and shutdown buffer nonnegative")
    if args.max_hours is not None and args.max_hours * 60 <= args.shutdown_buffer_minutes:
        parser.error("--max-hours must exceed the shutdown buffer")
    started = time.monotonic()
    deadline = None if args.max_hours is None else started + args.max_hours * 3600
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    selected = [r for stage in args.stage for r in rows(MANIFESTS[stage]) if r["status"] == "planned"]
    if supplemental and not stage_c_isolated and any(r.get("batch_id") != args.batch_id for r in selected):
        parser.error("every supplemental manifest row must carry the selected batch_id")
    if stage_c_isolated and (len(selected) != 360 or sha256(MANIFESTS["C_confirmation"]) != STAGE_C_MANIFEST_SHA256):
        parser.error("isolated Stage C requires all 360 unchanged frozen planned rows")
    if args.models:
        wanted = {name.strip() for name in args.models.split(",") if name.strip()}
        unknown = wanted - set(config["available_models"])
        if unknown:
            parser.error(f"--models names models the frozen config does not list: {sorted(unknown)}")
        selected = [row for row in selected if row["model"] in wanted]
        print(f"model filter={','.join(sorted(wanted))} matches={len(selected)}")
    _, _, issues = validate_preflight(selected)
    if issues:
        for issue in issues:
            print("BLOCK:", issue)
        return 2
    if len({logical_id(r) for r in selected}) != len(selected):
        print("BLOCK: duplicate logical run IDs in selected manifests")
        return 2
    print(f"selected={len(selected)} stages={','.join(args.stage)} protocol={config['protocol_version']}")
    pending = []
    already_done = 0
    resumable = set()
    for row in selected:
        attempts = sorted(OUTPUT.glob(logical_id(row) + "__attempt-*")) if OUTPUT.exists() else []
        if len(attempts) > 1:
            print(f"BLOCK: {logical_id(row)} has multiple attempts; explicit reconciliation required")
            return 2
        if attempts:
            try:
                split = split_for(row)
                current_config = {"row": row, "common": config["common"], "device": args.device}
                completed_attempt(attempts[0], row, split, canonical_hash(current_config))
            except (RuntimeError, json.JSONDecodeError) as error:
                if args.resume_interrupted and interrupted_attempt(attempts[0]):
                    resumable.add(logical_id(row))
                    pending.append(row)
                    continue
                print(f"BLOCK: {error}")
                return 2
            already_done += 1
        else:
            pending.append(row)
    selected = pending
    if args.limit is not None:
        selected = selected[:args.limit]
    print(f"already completed and hash-verified={already_done}")
    for row in selected[:5]:
        print("DRY", logical_id(row), f"split={row['split_mode']} horizon={row['horizon']}")
    if len(selected) > 5:
        print(f"... {len(selected) - 5} additional fits validated")
    if not args.execute:
        print("dry-run only; no training launched. Stage B/C/D execution remains separately gated.")
        return 0
    if not args.authorization or not args.environment_fingerprint:
        raise RuntimeError("--execute requires --authorization and --environment-fingerprint")
    auth, env_hash = validate_authorization(args.authorization, args.environment_fingerprint, set(args.stage),
                                            args.device, MANIFESTS[args.stage[0]], args.batch_id, OUTPUT)
    if stage_c_isolated and auth.get("execution_track") != STAGE_C_TRACK:
        raise RuntimeError("isolated Stage C requires its separate confirmatory_local_stage_c authorization")
    if supplemental:
        auth_hash = sha256(args.authorization)
        meta_path = OUTPUT / "batch_meta.json"
        expected_meta = {"batch_id": args.batch_id, "manifest_sha256": sha256(MANIFESTS[args.stage[0]]),
                         "authorization_sha256": auth_hash, "git_commit": auth["git_commit"],
                         "environment_fingerprint_sha256": env_hash}
        if stage_c_isolated:
            expected_meta.update(execution_track=STAGE_C_TRACK, **stage_c_receipt_metadata(auth))
        if OUTPUT.exists():
            if not OUTPUT.is_dir() or not meta_path.is_file():
                raise RuntimeError("existing supplemental root lacks its immutable batch metadata")
            actual_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if any(actual_meta.get(key) != value for key, value in expected_meta.items()):
                raise RuntimeError("existing supplemental root belongs to a different frozen batch")
        else:
            OUTPUT.mkdir(parents=False, exist_ok=False)
            atomic_json(meta_path, expected_meta)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was authorized but torch.cuda.is_available() is false")
    registry = {r["asset"]: r for r in rows(REGISTRY)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_root = ROOT / "reproduction/logs/phase1"
    log_root.mkdir(parents=True, exist_ok=True)
    progress_path = OUTPUT / "batch_progress.json" if supplemental else log_root / "batch_progress.json"
    progress = {"stages": args.stage, "total_selected": len(selected) + already_done,
                "previously_completed_verified": already_done, "newly_completed": 0,
                "max_hours": args.max_hours, "status": "running", "current_run": None}

    def save_progress(status: str, current: str | None = None) -> None:
        progress.update(status=status, current_run=current, elapsed_seconds=time.monotonic() - started,
                        updated_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(progress_path, progress)

    save_progress("running")
    bar = tqdm(total=len(selected) + already_done, initial=already_done, unit="fit", desc="Phase 1")
    for row in selected:
        remaining = (float("inf") if deadline is None
                     else deadline - time.monotonic() - args.shutdown_buffer_minutes * 60)
        if remaining < args.stop_below_seconds:
            save_progress("budget_exhausted")
            bar.close()
            print(f"BUDGET STOP: {remaining / 60:.1f} min left, below the "
                  f"{args.stop_below_seconds / 60:.1f} min floor for starting a fit. "
                  "Remaining fits stay planned; completed packages are durable and restart-verifiable.")
            return 0
        timeout = min(args.fit_timeout_hours * 3600, remaining)
        logical = logical_id(row)
        destination = OUTPUT / f"{logical}__attempt-01"
        resuming = logical in resumable
        if resuming:
            # The interrupted receipt is the record of what the budget cut
            # short.  A resumed fit must not erase it, so it is set aside
            # rather than overwritten.
            index = len(list(destination.glob("receipt.interrupted-*.json"))) + 1
            (destination / "receipt.json").rename(
                destination / f"receipt.interrupted-{index:02d}.json")
        else:
            if destination.exists():
                raise RuntimeError(f"refusing to overwrite attempt: {destination}")
            destination.mkdir(parents=True, exist_ok=False)
        run_started = datetime.now(timezone.utc).isoformat()
        atomic_json(destination / "receipt.json", {
            "logical_run_id": logical, "attempt": 1, "attempt_id": destination.name,
            "protocol_version": row["protocol_version"], "stage": row["stage"],
            "status": "running", "resumed_from_interrupted": resuming,
            "execution_track": auth.get("execution_track", "frozen_protocol"),
            "batch_id": auth.get("batch_id"),
            "created_at_utc": run_started,
            "authorization_record": auth.get("decision_record"), "authorization_commit": auth["git_commit"],
            "environment_fingerprint_sha256": env_hash,
            **stage_c_receipt_metadata(auth),
        })
        command = build_command(row, config["common"], registry[row["asset"]], 1, args.device,
                                args.run_label if supplemental else "phase1_frozen")
        if args.enable_recovery:
            command.append("--enable_recovery")
        model_id = f"{logical}__attempt-01"
        log_path = destination / "run.log"
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        save_progress("running", logical)
        bar.set_postfix_str(logical)
        with log_path.open("a" if resuming else "x", encoding="utf-8") as log:
            if resuming:
                log.write(f"\n===== resumed at {run_started} from this attempt's save point =====\n")
            returncode, timed_out = run_worker(command, log, env, timeout)
        if returncode or timed_out:
            failed = {"logical_run_id": logical, "attempt": 1, "attempt_id": destination.name,
                      "protocol_version": row["protocol_version"], "stage": row["stage"],
                      "execution_track": auth.get("execution_track", "frozen_protocol"),
                      "batch_id": auth.get("batch_id"),
                      "status": "interrupted" if timed_out else "failed",
                      "failure_class": "budget_or_fit_timeout" if timed_out else "nonzero_exit", "exit_code": returncode,
                      "created_at_utc": run_started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                      "authorization_record": auth.get("decision_record"), "authorization_commit": auth["git_commit"],
                      "environment_fingerprint_sha256": env_hash, "sha256": {"run.log": sha256(log_path)},
                      **stage_c_receipt_metadata(auth)}
            atomic_json(destination / "receipt.json", failed)
            save_progress("interrupted" if timed_out else "failed", logical)
            bar.close()
            print(f"STOP: {logical}: exit={returncode}, timeout={timed_out}; all partial artifacts retained, no retry.")
            return 3 if timed_out else 2
        matches = [p for p in (ROOT / "results").glob(f"*{model_id}*") if p.is_dir()]
        if stage_c_isolated:
            matches = [p for p in matches if p.name.endswith(f"_{STAGE_C_RUN_LABEL}_seed{row['seed']}_0")]
        if len(matches) != 1:
            raise RuntimeError(f"expected one result directory for {logical}; found {len(matches)}")
        try:
            package_attempt(row, destination, command, log_path, matches[0].name, auth, env_hash,
                            config["common"], registry[row["asset"]])
        except Exception as error:
            invalid = {"logical_run_id": logical, "attempt": 1, "attempt_id": destination.name,
                       "protocol_version": row["protocol_version"], "stage": row["stage"],
                       "execution_track": auth.get("execution_track", "frozen_protocol"),
                       "batch_id": auth.get("batch_id"),
                       "status": "artifact_invalid", "reason": str(error),
                       "created_at_utc": run_started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                       "authorization_record": auth.get("decision_record"), "authorization_commit": auth["git_commit"],
                       "environment_fingerprint_sha256": env_hash,
                       "sha256": {"run.log": sha256(log_path)}, **stage_c_receipt_metadata(auth)}
            atomic_json(destination / "receipt.json", invalid)
            save_progress("artifact_invalid", logical)
            bar.close()
            raise
        progress["newly_completed"] += 1
        save_progress("running")
        bar.update(1)
        print("PACKAGED", destination)
    bar.close()
    save_progress("queue_completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
