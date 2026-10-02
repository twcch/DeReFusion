"""Write a new runtime Stage C credential after committing the reviewed launcher.

Consumes the existing immutable local fingerprint; does not probe or use the GPU.
The fingerprint's historical track label is retained; the authorization owns the
new execution-track identity.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from reproduction.batches.run_phase1_stages import (
    MANIFESTS, PHASE, ROOT, STAGE_C_AMENDMENT, STAGE_C_AUTH_VERSION,
    STAGE_C_BATCH, STAGE_C_DRIVER, STAGE_C_ENV_SHA256, STAGE_C_GPU,
    STAGE_C_GPU_UUID, STAGE_C_LOCK, STAGE_C_MANIFEST_SHA256,
    STAGE_C_RUN_LABEL, STAGE_C_TRACK, sha256, validate_authorization,
)


def build_authorization(approved_by: str, approved_at_utc: str, commit: str,
                        fingerprint: Path) -> dict:
    if not approved_by.strip():
        raise RuntimeError("an explicit user approver record is required")
    approved_time = datetime.fromisoformat(approved_at_utc.replace("Z", "+00:00"))
    if approved_time.tzinfo is None or approved_time.utcoffset().total_seconds() != 0:
        raise RuntimeError("approval recording time must include UTC timezone")
    if sha256(fingerprint) != STAGE_C_ENV_SHA256:
        raise RuntimeError("existing local environment fingerprint changed")
    if sha256(MANIFESTS["C_confirmation"]) != STAGE_C_MANIFEST_SHA256:
        raise RuntimeError("the full frozen Stage C manifest changed")
    return {
        "authorization_version": STAGE_C_AUTH_VERSION,
        "protocol_version": "phase1-v1.1-2026-09-19",
        "execution_track": STAGE_C_TRACK,
        "execution_amendment_id": STAGE_C_AMENDMENT,
        "stages": ["C_confirmation"], "execution_device": "cuda",
        "approved_by": approved_by, "approved_at_utc": approved_at_utc,
        "approval_statement": "批准 Stage C 并行运行",
        "decision_record": "docs/PHASE1_STAGE_C_EXECUTION_DECISION_2026-09-29.md",
        "scope": "All 360 frozen Stage C fits; both model arms on the same local GPU and stack; prospective parallel execution with Stage B",
        "batch_id": STAGE_C_BATCH, "run_label": STAGE_C_RUN_LABEL,
        "git_commit": commit, "environment_lock": dict(STAGE_C_LOCK),
        "environment_fingerprint_sha256": sha256(fingerprint),
        "manifest_sha256": STAGE_C_MANIFEST_SHA256, "manifest_count": 360,
        "output_root": str((PHASE / STAGE_C_BATCH).resolve()),
        "gpu_model": STAGE_C_GPU, "gpu_uuid": STAGE_C_GPU_UUID,
        "driver_version": STAGE_C_DRIVER, "cudnn": 90701,
        "cross_stack_pooling_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-by", required=True,
                        help="explicit user approval attribution; this approval is already granted")
    parser.add_argument("--approved-at-utc",
                        help="known approval UTC time, or omit to use current UTC recording time")
    parser.add_argument("--environment-fingerprint", type=Path,
                        default=PHASE / "local-5060-manifests/environment_fingerprint.json")
    parser.add_argument("--authorization", type=Path,
                        default=PHASE / "local-5060-manifests/authorization-local-5060-stage-c-v1.json")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    timestamp = args.approved_at_utc or datetime.now(timezone.utc).isoformat()
    auth = build_authorization(args.approved_by, timestamp, commit, args.environment_fingerprint)
    # Never update an existing credential; launch will revalidate every binding.
    if args.authorization.exists():
        raise RuntimeError("refusing to overwrite an existing authorization")
    args.authorization.parent.mkdir(parents=True, exist_ok=True)
    with args.authorization.open("x", encoding="utf-8") as stream:
        json.dump(auth, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    validate_authorization(args.authorization, args.environment_fingerprint,
                           {"C_confirmation"}, "cuda", MANIFESTS["C_confirmation"],
                           STAGE_C_BATCH, PHASE / STAGE_C_BATCH)
    print(json.dumps({"authorization": str(args.authorization.resolve()),
                      "authorization_sha256": sha256(args.authorization),
                      "git_commit": commit, "execution_amendment_id": STAGE_C_AMENDMENT}, indent=2))


if __name__ == "__main__":
    main()
