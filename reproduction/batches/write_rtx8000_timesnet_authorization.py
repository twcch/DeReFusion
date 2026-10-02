"""Fingerprint and authorize the isolated, nonconfirmatory RTX 8000 TimesNet batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy
import pandas
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "phase1-v1.1-2026-09-19"
BATCH_ID = "rtx8000-timesnet-remaining-v1"
LOCK = {"python": "3.11.15", "torch": "2.5.1+cu121", "torch_cuda": "12.1",
        "numpy": "2.1.2", "pandas": "2.3.3", "scikit_learn": "1.7.2"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.resolve() != (ROOT / "reproduction/results/phase1/supplemental-manifests/rtx8000-timesnet-remaining-v1.csv").resolve():
        raise SystemExit("manifest path is not the reviewed RTX 8000 manifest")
    if args.output_root.resolve() != (ROOT / "reproduction/results/phase1/rtx8000-timesnet-remaining-v1").resolve():
        raise SystemExit("output root is not the reviewed RTX 8000 batch directory")
    actual = {"python": platform.python_version(), "torch": torch.__version__,
              "torch_cuda": torch.version.cuda, "numpy": numpy.__version__,
              "pandas": pandas.__version__, "scikit_learn": sklearn.__version__}
    if actual != LOCK:
        raise SystemExit(f"environment differs from reviewed execution lock: {actual}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one available CUDA device")
    gpu = torch.cuda.get_device_name(0)
    if gpu != "Quadro RTX 8000":
        raise SystemExit(f"expected Quadro RTX 8000; found {gpu}")
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,driver_version,name", "--format=csv,noheader"],
        text=True, encoding="utf-8").strip().splitlines()
    if len(gpu_info) != 1:
        raise SystemExit("expected exactly one GPU in nvidia-smi")
    gpu_uuid, driver, nvidia_name = [part.strip() for part in gpu_info[0].split(",", 2)]
    if nvidia_name != gpu:
        raise SystemExit("PyTorch and nvidia-smi disagree about the GPU")
    observed = {**actual, "cudnn": torch.backends.cudnn.version(), "cuda_available": True,
                "gpu_count": 1, "gpu_names": [gpu], "gpu_uuid": gpu_uuid,
                "driver_version": driver, "platform": platform.platform(),
                "pip_freeze": subprocess.check_output(
                    [sys.executable, "-m", "pip", "freeze"], text=True, encoding="utf-8").splitlines()}
    fingerprint = {"track": "nonconfirmatory_remote_gpu_supplement", "observed": observed,
                   "mismatches": {}}
    fingerprint_path = args.output_root / "environment_fingerprint.json"
    write_new(fingerprint_path, fingerprint)
    manifest_rows = args.manifest.read_text(encoding="utf-8-sig").splitlines()
    if len(manifest_rows) != 56 or any(BATCH_ID not in row for row in manifest_rows[1:]):
        raise SystemExit("manifest must contain exactly 55 rows with the fixed RTX 8000 batch_id")
    auth = {"authorization_version": "phase1-remote-supplement-auth/v1",
            "protocol_version": PROTOCOL, "execution_track": "nonconfirmatory_remote_gpu_supplement",
            "stages": ["B_screen"], "execution_device": "cuda", "approved_by": "user (explicit authorization in Codex conversation)",
            "approved_at_utc": datetime.now(timezone.utc).isoformat(),
            "decision_record": "docs/PHASE1_GPU_BRIDGE_CALIBRATION_PROPOSAL.md",
            "scope": "55 previously unrun TimesNet B_screen rows; execution-only supplement; not confirmatory or poolable",
            "batch_id": BATCH_ID, "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "environment_lock": LOCK, "environment_fingerprint_sha256": sha256(fingerprint_path),
            "manifest_sha256": sha256(args.manifest), "output_root": str(args.output_root.resolve()),
            "gpu_model": gpu, "gpu_uuid": gpu_uuid, "driver_version": driver,
            "pooling_authorized": False, "ranking_authorized": False,
            "equivalence_claim_authorized": False}
    write_new(args.output_root / "authorization.json", auth)
    print(json.dumps({"batch_id": BATCH_ID, "gpu": gpu, "gpu_uuid": gpu_uuid,
                      "driver_version": driver, "environment_lock": LOCK,
                      "manifest_sha256": auth["manifest_sha256"],
                      "environment_fingerprint_sha256": auth["environment_fingerprint_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
