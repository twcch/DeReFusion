"""Record the observed local GPU stack and hash-bound supplemental approvals."""
from __future__ import annotations

import hashlib
import json
import os
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
OUT = ROOT / "reproduction/results/phase1/local-5060-manifests"
PROTOCOL = "phase1-v1.1-2026-09-19"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    if sys.version_info[:3] != (3, 11, 15):
        raise SystemExit(f"expected Python 3.11.15, found {platform.python_version()}")
    if torch.__version__ != "2.7.1+cu128" or torch.version.cuda != "12.8":
        raise SystemExit("local execution stack differs from the reviewed RTX 5060 Ti environment")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one available CUDA device")
    gpu = torch.cuda.get_device_name(0)
    if gpu != "NVIDIA GeForce RTX 5060 Ti":
        raise SystemExit(f"unexpected CUDA device: {gpu}")
    nvidia = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,driver_version,name", "--format=csv,noheader"],
        text=True, encoding="utf-8").strip().splitlines()
    if len(nvidia) != 1:
        raise SystemExit("could not fingerprint exactly one GPU using nvidia-smi")
    uuid, driver, nvidia_name = [part.strip() for part in nvidia[0].split(",", 2)]
    if nvidia_name != gpu:
        raise SystemExit("torch and nvidia-smi report different GPU names")
    fingerprint = {
        "track": "nonconfirmatory_local_supplement",
        "observed": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__, "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(), "gpu_names": [gpu], "gpu_uuid": uuid,
            "driver_version": driver, "platform": platform.platform(),
            "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"],
                                                    text=True, encoding="utf-8").splitlines(),
        },
        "mismatches": {},
    }
    fp_path = OUT / "environment_fingerprint.json"
    atomic_new(fp_path, fingerprint)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    now = datetime.now(timezone.utc).isoformat()
    specs = [
        ("bridge-v1.csv", "local-5060-bridge-v1", "local-5060-bridge-v1",
         "Descriptive paired execution-stack bridge only; no numerical equivalence claim."),
        ("timesnet-remaining-v1.csv", "local-5060-timesnet-remaining-v1", "local-5060-timesnet-remaining-v1",
         "Technical supplement for unrun TimesNet rows; not confirmatory and not poolable with P4."),
    ]
    for manifest_name, batch_id, root_name, scope in specs:
        manifest = OUT / manifest_name
        decision = "docs/PHASE1_GPU_BRIDGE_CALIBRATION_PROPOSAL.md"
        auth = {
            "authorization_version": "phase1-local-supplement-auth/v1",
            "protocol_version": PROTOCOL,
            "execution_track": "nonconfirmatory_local_supplement",
            "stages": ["B_screen"], "execution_device": "cuda",
            "approved_by": "user (explicit confirmation in Codex conversation)",
            "approved_at_utc": now, "decision_record": decision,
            "scope": scope, "batch_id": batch_id,
            "git_commit": commit,
            "environment_fingerprint_sha256": sha256(fp_path),
            "manifest_sha256": sha256(manifest),
            "output_root": str((ROOT / "reproduction/results/phase1" / root_name).resolve()),
            "gpu_model": gpu, "gpu_uuid": uuid, "driver_version": driver,
            "pooling_authorized": False, "ranking_authorized": False,
            "equivalence_claim_authorized": False,
        }
        atomic_new(OUT / f"authorization-{batch_id}.json", auth)
    summary_path = OUT / "freeze-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "authorized_nonconfirmatory_execution"
    summary["authorization_sha256"] = {
        batch_id: sha256(OUT / f"authorization-{batch_id}.json")
        for _, batch_id, _, _ in specs
    }
    temporary = summary_path.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, summary_path)
    print(json.dumps({"commit": commit, "gpu": gpu, "gpu_uuid": uuid, "driver": driver,
                      "fingerprint_sha256": sha256(fp_path)}, indent=2))


if __name__ == "__main__":
    main()
