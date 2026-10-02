"""CPU-only intake of completed packages and preparation of checkpoint replay inputs.

No model is loaded or fitted. Every attempt remains in place, including running,
interrupted, failed, and ambiguous attempts. Exported references describe source
identity; the replay tool's legacy ``p4`` transport label is not a hardware claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction.analysis.audit_phase1_checkpoint_forward import (  # noqa: E402
    BoundInputs, new_output_root, write_json,
)
from reproduction.batches.run_phase1_stages import (  # noqa: E402
    CONFIG, CONTRACT, KEYS, REGISTRY, SPLITS, canonical_hash, expected_truth,
    logical_id, rows, split_for, validate_preflight,
)
from utils.metrics import metric  # noqa: E402

ARTIFACTS = {"checkpoint.pth", "pred.npy", "true.npy", "metrics.npy", "run.log"}
METRICS = ("MAE", "MSE", "RMSE", "MAPE", "MSPE", "R2")


class IntakeBoundInputs(BoundInputs):
    """Keep input timestamps alongside the independently verified byte hashes."""
    def __init__(self):
        super().__init__()
        self.stats: dict[str, dict] = {}

    def bind(self, path: Path, expected: str | None = None) -> str:
        digest = super().bind(path, expected)
        path = path.resolve()
        stat = path.stat()
        self.stats.setdefault(str(path), {"size_bytes": stat.st_size,
                                          "mtime_ns": stat.st_mtime_ns,
                                          "bound_at_utc": utc_now()})
        return digest


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", type=Path, required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True,
                        help="fresh immutable report directory, outside source/data inputs")
    result.add_argument("--source-label", required=True,
                        help="descriptive source label, e.g. p4 or rtx8000; never used for selection")
    result.add_argument("--attempt-name", action="append", default=None,
                        help="intake these newly completed attempts only; other evidence is preserved")
    return result


def write_csv(path: Path, fields: list[str], records: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def scan_attempts(source: Path) -> list[tuple[Path, dict | None, dict]]:
    """Snapshot receipts only; never inspect mutable artifacts of unfinished fits."""
    found = []
    for attempt in sorted(source.glob("*__attempt-*")):
        if not attempt.is_dir():
            continue
        record = {"attempt_path": str(attempt.resolve()), "attempt_id": attempt.name,
                  "status": "excluded", "reason": "", "receipt_status": None}
        receipt = None
        try:
            receipt = json.loads((attempt / "receipt.json").read_text(encoding="utf-8"))
            if not isinstance(receipt, dict):
                raise ValueError("receipt must be a JSON object")
            record["receipt_status"] = receipt.get("status")
            record["logical_run_id"] = receipt.get("logical_run_id")
            if receipt.get("status") != "completed_unreviewed":
                record["reason"] = "receipt is not completed_unreviewed; preserved without reading artifacts"
            else:
                record["status"] = "pending_intake"
        except (OSError, ValueError) as error:
            record.update(status="requires-resolution", reason=f"receipt unreadable: {error}")
        found.append((attempt, receipt, record))
    return found


def validate_package(attempt: Path, receipt: dict, row: dict[str, str],
                     registry: dict, config: dict, bound: BoundInputs,
                     truth_cache: dict) -> dict:
    logical = logical_id(row)
    # Re-read the bound commit marker to detect receipt replacement since scanning.
    receipt_hash = bound.bind(attempt / "receipt.json")
    if json.loads((attempt / "receipt.json").read_text(encoding="utf-8")) != receipt:
        raise RuntimeError("receipt changed since source snapshot")
    suffix = re.fullmatch(re.escape(logical) + r"__attempt-([0-9]+)", attempt.name)
    if (not suffix or int(suffix[1]) < 1 or receipt.get("attempt_id") != attempt.name
            or receipt.get("attempt") != int(suffix[1])):
        raise RuntimeError("receipt attempt identity differs from directory")
    device = receipt.get("execution_device")
    if device not in {"cpu", "cuda"}:
        raise RuntimeError("invalid execution_device")
    commit = receipt.get("authorization_commit", "")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("missing or invalid authorization_commit")
    track = receipt.get("execution_track", "frozen_protocol")
    if not isinstance(track, str) or not track.strip():
        raise RuntimeError("missing or invalid execution_track")
    if not receipt.get("authorization_record"):
        raise RuntimeError("missing authorization_record")
    env_hash = receipt.get("environment_fingerprint_sha256", "")
    if not isinstance(env_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", env_hash):
        raise RuntimeError("missing or invalid environment fingerprint identity")
    split = split_for(row)
    reg = registry[row["asset"]]
    wanted_identity = {
        "status": "completed_unreviewed", "logical_run_id": logical,
        "stage": row["stage"], "protocol_version": row["protocol_version"],
        "dataset_sha256": reg["sha256"], "split_manifest_id": split["split_manifest_id"],
        "prediction_keys_sha256": split["prediction_keys_sha256"],
        "config_fingerprint": canonical_hash({"row": row, "common": config["common"], "device": device}),
    }
    for name, wanted in wanted_identity.items():
        if receipt.get(name) != wanted:
            raise RuntimeError(f"receipt {name} differs from frozen inputs")
    if set(receipt.get("sha256", {})) != ARTIFACTS:
        raise RuntimeError("receipt must bind exactly the five required artifacts")
    for name in sorted(ARTIFACTS):
        bound.bind(attempt / name, receipt["sha256"][name])
    data_path = ROOT / reg["file"]
    if str(data_path.resolve()) not in bound.hashes:
        bound.bind(data_path, reg["sha256"])
    pred, true, packaged = (np.load(attempt / name, allow_pickle=False)
                            for name in ("pred.npy", "true.npy", "metrics.npy"))
    cache_key = split["split_manifest_id"]
    if cache_key not in truth_cache:
        truth_cache[cache_key] = expected_truth(row, split, config["common"], data_path)
    truth = truth_cache[cache_key]
    shape = (int(split["window_count"]), int(row["horizon"]), 1)
    if truth.shape != shape or pred.shape != shape or true.shape != shape or packaged.shape != (6,):
        raise RuntimeError("array shapes differ from frozen target/key cardinality")
    for name, array in (("pred.npy", pred), ("true.npy", true), ("metrics.npy", packaged)):
        if not np.isfinite(array).all():
            raise RuntimeError(f"non-finite array: {name}")
        if receipt.get("shapes", {}).get(name) != list(array.shape):
            raise RuntimeError(f"receipt shape differs from artifact: {name}")
    if not np.allclose(true, truth, rtol=2e-5, atol=2e-6):
        raise RuntimeError("true.npy differs from frozen targets and train-prefix scaling")
    recomputed = np.asarray(metric(pred, true), dtype=np.float64)
    if not np.isfinite(recomputed).all() or not np.allclose(packaged, recomputed, rtol=1e-5, atol=1e-8):
        raise RuntimeError("six packaged metrics do not reproduce from pred/true")
    return {"logical_run_id": logical, "source_commit": commit, "receipt_sha256": receipt_hash,
            "p4_attempt": Path(os.path.relpath(attempt.resolve(), ROOT.resolve())).as_posix(),
            "execution_track": track, "execution_track_legacy_default": "execution_track" not in receipt,
            "execution_device": device, "environment_fingerprint_sha256": env_hash,
            "authorization_record": receipt["authorization_record"],
            "receipt_created_at_utc": receipt.get("created_at_utc"),
            "batch_id": receipt.get("batch_id"), "shape": list(shape),
            "artifact_sha256": receipt["sha256"], "metrics": dict(zip(METRICS, map(float, packaged)))}


def intake(source_root: Path, manifest: Path, output_root: Path, source_label: str,
           only_attempts: set[str] | None = None) -> dict:
    source_root, manifest = source_root.resolve(), manifest.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root missing: {source_root}")
    output = new_output_root(output_root, [source_root, ROOT / "dataset", manifest,
                                          CONFIG, REGISTRY, CONTRACT, SPLITS, KEYS])
    bound = IntakeBoundInputs()
    report = {"intake_version": "phase1-completed-package-intake/v1", "created_at_utc": utc_now(),
              "source_root": str(source_root), "source_label": source_label,
              "source_manifest": str(manifest), "status": "running", "attempts": [],
              "issues": [], "pooling_authorized": False, "ranking_authorized": False,
              "numerical_equivalence_threshold": "TBD", "numerical_equivalence_claim": False,
              "selection": "completed_unreviewed receipts only; never selected by score",
              "replay_transport": "p4 is the legacy reference-index mode, not a source GPU label"}
    write_json(output / "intake-plan.json", report)
    try:
        for path in (manifest, CONFIG, REGISTRY, CONTRACT, SPLITS, KEYS, Path(__file__)):
            bound.bind(path)
        manifest_rows = rows(manifest)
        if not manifest_rows:
            raise RuntimeError("source manifest has no rows")
        by_id = {logical_id(row): row for row in manifest_rows}
        if len(by_id) != len(manifest_rows):
            raise RuntimeError("duplicate logical identities in source manifest")
        scanned = scan_attempts(source_root)
        if only_attempts is not None:
            present = {attempt.name for attempt, _, _ in scanned}
            if only_attempts - present:
                raise RuntimeError("requested attempts are absent from the source snapshot")
            for attempt, _, record in scanned:
                if attempt.name not in only_attempts and record["status"] == "pending_intake":
                    record.update(status="deferred", reason="not newly queued in this intake pass; preserved")
            report["requested_attempts"] = sorted(only_attempts)
        report["attempts"] = [record for _, _, record in scanned]
        candidates = [by_id[receipt["logical_run_id"]] for _, receipt, record in scanned
                      if record["status"] == "pending_intake" and receipt.get("logical_run_id") in by_id]
        # One model-independent key scan per intake, irrespective of fit count.
        unique = {logical_id(row): row for row in candidates}
        registry, _, preflight_issues = validate_preflight(list(unique.values())) if unique else ({}, {}, [])
        if preflight_issues:
            raise RuntimeError(f"frozen data/key preflight failed: {preflight_issues}")
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        identities, truth_cache = [], {}
        for attempt, receipt, record in scanned:
            if record["status"] != "pending_intake":
                continue
            logical = receipt.get("logical_run_id")
            try:
                if logical not in by_id:
                    raise RuntimeError("completed attempt has no matching frozen manifest row")
                identity = validate_package(attempt, receipt, by_id[logical], registry, config, bound, truth_cache)
                identity["source_label"] = source_label
                identities.append(identity)
                record.update(status="eligible", reason="completed-package integrity verified", identity=identity)
            except Exception as error:
                record.update(status="requires-resolution", reason=f"{type(error).__name__}: {error}")
        # Any second completed package needs identity review even if it fails
        # another check: never silently choose one attempt from a retry pair.
        counts = Counter(receipt.get("logical_run_id") for _, receipt, record in scanned
                         if receipt is not None and record["receipt_status"] == "completed_unreviewed")
        for record in report["attempts"]:
            if record["status"] == "eligible" and counts[record["logical_run_id"]] > 1:
                record.update(status="requires-resolution", reason="multiple completed packages for logical setting; replay reference requires review")
        export = [item for item in identities if counts[item["logical_run_id"]] == 1]
        export_ids = {item["logical_run_id"] for item in export}
        selected = [row for row in manifest_rows if logical_id(row) in export_ids]
        index_by_id = {item["logical_run_id"]: item for item in export}
        fields = ["logical_run_id", "p4_attempt", "receipt_sha256", "source_commit"]
        reference_rows = [{field: index_by_id[logical_id(row)][field] for field in fields} for row in selected]
        # Re-hash every bound input before publishing the replay exports.
        bound.verify()
        report["input_immutability_verified"] = True
        write_csv(output / "selected-manifest.csv", list(manifest_rows[0]), selected)
        write_csv(output / "reference-index.csv", fields, reference_rows)
        write_json(output / "source-identities.json", {"source_label": source_label, "packages": identities})
        report["summary"] = {"manifest_settings": len(manifest_rows), "attempts_seen": len(scanned),
                             "eligible_packages": len(export), "replay_settings": len(selected),
                             "statuses": dict(Counter(record["status"] for record in report["attempts"])),
                             "absent_manifest_settings": len(set(by_id) - {record.get("logical_run_id") for record in report["attempts"]})}
        report["issues"] = [f"{record['attempt_id']}: {record['reason']}" for record in report["attempts"]
                            if record["status"] == "requires-resolution"]
        report["status"] = "requires_resolution" if report["issues"] else "completed_descriptive_only"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        for record in report["attempts"]:
            if record["status"] in {"pending_intake", "eligible"}:
                record.update(status="requires-resolution", reason="intake failed; see report error")
        try:
            bound.verify()
            report["input_immutability_verified"] = True
        except Exception as verification_error:
            report["input_immutability_verified"] = False
            report["immutability_error"] = str(verification_error)
    finally:
        report["input_file_sha256"] = bound.hashes
        report["input_file_metadata"] = bound.stats
        report["completed_at_utc"] = utc_now()
        report["gate_table"] = [
            {"check": "completed package integrity", "input": "receipt, five artifacts, frozen manifest/data/keys",
             "status": "eligible" if report["status"] == "completed_descriptive_only" else "requires-resolution",
             "evidence": "per-attempt records and input_file_sha256", "next_permitted_action": "review eligible packages; fixed-checkpoint diagnostic replay only"},
            {"check": "unfinished package", "input": "running/interrupted receipts", "status": "excluded-by-protocol",
             "evidence": "recorded receipt status; original files preserved", "next_permitted_action": "intake a new immutable snapshot after completion"},
            {"check": "cross-stack equivalence, pooling, ranking", "input": "descriptive intake",
             "status": "requires-resolution", "evidence": "threshold TBD; no score-based selection or pooling",
             "next_permitted_action": "approved research-question/precision protocol before scientific claims"}]
        write_json(output / "report.json", report)
    return report


def main() -> int:
    args = parser().parse_args()
    report = intake(args.source_root, args.manifest, args.output_root, args.source_label,
                    set(args.attempt_name) if args.attempt_name is not None else None)
    print(json.dumps({"status": report["status"], "summary": report.get("summary"),
                      "error": report.get("error"), "report": str(args.output_root / "report.json")}, indent=2))
    return 0 if report["status"] == "completed_descriptive_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
