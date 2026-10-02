"""Freeze two nonconfirmatory RTX 5060 Ti batches from audited Phase 1 inputs."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

from run_phase1_stages import logical_id, rows, sha256

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "reproduction/results/phase1"
ARCHIVE = PHASE / "p4-partial-20260929"
BRIDGE_ID = "local-5060-bridge-v1"
SUPPLEMENT_ID = "local-5060-timesnet-remaining-v1"
BRIDGE_ASSETS = {"AAPL", "GSPC", "EURUSD", "BTCUSD"}
BRIDGE_MODELS = {"revin-DLinear", "DeReFusion", "revin-PatchTST", "revin-iTransformer"}


def digest_json(path: Path) -> tuple[dict, bool]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    ok = receipt.get("status") == "completed_unreviewed"
    for name, expected in receipt.get("sha256", {}).items():
        artifact = path.parent / name
        ok = ok and artifact.is_file() and sha256(artifact) == expected
    return receipt, ok


def write_csv(path: Path, fieldnames: list[str], records: list[dict[str, str]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    manifest = rows(PHASE / "B_screen.manifest.csv")
    archive_attempts = sorted((ARCHIVE / "attempts").glob("*/receipt.json"))
    if len(archive_attempts) != 245:
        raise SystemExit(f"expected 245 archived P4 receipts; found {len(archive_attempts)}")
    p4_by_id: dict[str, tuple[Path, dict]] = {}
    for path in archive_attempts:
        receipt, valid = digest_json(path)
        if not valid:
            raise SystemExit(f"invalid P4 receipt/artifact hash: {path}")
        p4_by_id[receipt["logical_run_id"]] = (path, receipt)
    if len(p4_by_id) != 245:
        raise SystemExit(f"expected unique 245 P4 logical IDs; found {len(p4_by_id)}")

    bridge: list[dict[str, str]] = []
    for row in manifest:
        if (row["asset"] in BRIDGE_ASSETS and row["model"] in BRIDGE_MODELS
                and row["horizon"] in {"1", "24"} and row["seed"] == "2021"):
            bridge.append(dict(row, batch_id=BRIDGE_ID, batch_role="descriptive_bridge"))
        elif (row["asset"] == "AAPL" and row["model"] == "revin-TimesNet"
              and row["horizon"] in {"1", "24"} and row["seed"] == "2021"):
            bridge.append(dict(row, batch_id=BRIDGE_ID, batch_role="descriptive_bridge"))
    if len(bridge) != 34:
        raise SystemExit(f"bridge design must contain 34 rows; found {len(bridge)}")
    missing_refs = [logical_id(row) for row in bridge if logical_id(row) not in p4_by_id]
    if missing_refs:
        raise SystemExit("missing P4 bridge reference receipts: " + ", ".join(missing_refs))

    remaining = [dict(row, batch_id=SUPPLEMENT_ID, batch_role="nonconfirmatory_timesnet_completion")
                 for row in manifest if row["model"] == "revin-TimesNet"
                 and logical_id(row) not in p4_by_id]
    if len(remaining) != 55:
        raise SystemExit(f"expected 55 unrun TimesNet rows; found {len(remaining)}")
    if set(logical_id(r) for r in remaining) & set(logical_id(r) for r in bridge):
        raise SystemExit("bridge and remaining TimesNet manifests overlap")

    out = PHASE / "local-5060-manifests"
    out.mkdir(parents=True, exist_ok=False)
    columns = list(manifest[0]) + ["batch_id", "batch_role"]
    write_csv(out / "bridge-v1.csv", columns, bridge)
    write_csv(out / "timesnet-remaining-v1.csv", columns, remaining)
    refs = [{"logical_run_id": logical_id(row),
             "p4_attempt": str(p4_by_id[logical_id(row)][0].parent.relative_to(ROOT)),
             "receipt_sha256": sha256(p4_by_id[logical_id(row)][0]),
             "source_commit": p4_by_id[logical_id(row)][1].get("authorization_commit", "")}
            for row in bridge]
    with (out / "bridge-p4-reference-index.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(refs[0]))
        writer.writeheader()
        writer.writerows(refs)
    summary = {
        "status": "frozen_not_yet_authorized",
        "protocol_version": "phase1-v1.1-2026-09-19",
        "manifest_frozen_from_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "bridge_batch_id": BRIDGE_ID,
        "bridge_rows": len(bridge),
        "bridge_manifest_sha256": sha256(out / "bridge-v1.csv"),
        "remaining_timesnet_batch_id": SUPPLEMENT_ID,
        "remaining_timesnet_rows": len(remaining),
        "remaining_timesnet_manifest_sha256": sha256(out / "timesnet-remaining-v1.csv"),
        "p4_reference_receipts_audited": len(p4_by_id),
        "bridge_p4_references": len(refs),
        "p4_artifact_hash_mismatches": 0,
        "pooling_authorized": False,
        "ranking_authorized": False,
    }
    (out / "freeze-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
