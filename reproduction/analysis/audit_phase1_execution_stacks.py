"""Read-only integrity and paired-output audit for the P4/RTX 5060 bridge."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction.batches.run_phase1_stages import (  # noqa: E402
    CONFIG, MANIFESTS, canonical_hash, completed_attempt, expected_truth,
    logical_id, rows, sha256, split_for, validate_preflight,
)
from utils.metrics import metric  # noqa: E402

METRIC_NAMES = ("MAE", "MSE", "RMSE", "MAPE", "MSPE", "R2")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(array.size), "min": float(np.min(array)),
            "median": float(np.median(array)), "max": float(np.max(array))}


def audit_batch(name: str, manifest_rows: list[dict[str, str]], root: Path,
                registry: dict[str, dict[str, str]], config: dict,
                allow_partial: bool = False) -> tuple[dict, dict[str, Path]]:
    issues: list[str] = []
    found: dict[str, Path] = {}
    for row in manifest_rows:
        logical = logical_id(row)
        attempt = root / f"{logical}__attempt-01"
        if not attempt.is_dir():
            continue
        if logical in found:
            issues.append(f"{logical}: duplicate manifest identity")
        found[logical] = attempt
    expected_dirs = {f"{logical}__attempt-01" for logical in found}
    actual_dirs = {path.name for path in root.glob("*__attempt-*") if path.is_dir()}
    extras = sorted(actual_dirs - expected_dirs)
    if extras:
        issues.append(f"unexpected attempt directories ({len(extras)}): {extras[:10]}")
    missing = [logical_id(row) for row in manifest_rows if logical_id(row) not in found]
    if missing and not allow_partial:
        issues.append(f"missing attempts ({len(missing)}): {missing[:10]}")
    selected = [row for row in manifest_rows if logical_id(row) in found]
    _, _, preflight = validate_preflight(selected)
    issues.extend(f"preflight: {issue}" for issue in preflight)
    file_hashes_checked = 0
    metrics_recomputed = 0
    rows_audited = []
    for row in selected:
        logical = logical_id(row)
        attempt = found[logical]
        try:
            receipt_path = attempt / "receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("logical_run_id") != logical:
                raise RuntimeError("receipt logical_run_id differs from manifest")
            if receipt.get("status") != "completed_unreviewed":
                raise RuntimeError(f"unexpected receipt status {receipt.get('status')!r}")
            if receipt.get("stage") != row["stage"] or receipt.get("protocol_version") != row["protocol_version"]:
                raise RuntimeError("receipt stage/protocol differs from manifest")
            device = receipt.get("execution_device")
            if device not in {"cpu", "cuda"}:
                raise RuntimeError("receipt execution_device is missing or invalid")
            split = split_for(row)
            expected_config = canonical_hash({"row": row, "common": config["common"], "device": device})
            completed_attempt(attempt, row, split, expected_config)
            required = {"pred.npy", "true.npy", "metrics.npy", "checkpoint.pth", "run.log"}
            if set(receipt["sha256"]) != required:
                raise RuntimeError("receipt artifact inventory is not the required five files")
            file_hashes_checked += len(required)
            reg = registry[row["asset"]]
            if receipt.get("dataset_sha256") != reg["sha256"]:
                raise RuntimeError("receipt dataset hash differs from registry")
            if receipt.get("split_manifest_id") != split["split_manifest_id"]:
                raise RuntimeError("receipt split identity differs from frozen split manifest")
            if receipt.get("prediction_keys_sha256") != split["prediction_keys_sha256"]:
                raise RuntimeError("receipt prediction-key hash differs from frozen manifest")
            pred, true, metrics = (np.load(attempt / filename, allow_pickle=False)
                                   for filename in ("pred.npy", "true.npy", "metrics.npy"))
            truth = expected_truth(row, split, config["common"], ROOT / reg["file"])
            if pred.shape != truth.shape or true.shape != truth.shape or metrics.shape != (6,):
                raise RuntimeError(f"array shape mismatch: pred={pred.shape}, true={true.shape}, metrics={metrics.shape}, expected={truth.shape}")
            if not all(np.isfinite(array).all() for array in (pred, true, metrics)):
                raise RuntimeError("non-finite values in packaged arrays")
            if not np.allclose(true, truth, rtol=2e-5, atol=2e-6):
                raise RuntimeError("true.npy does not match frozen test targets and train-only scaling")
            recomputed = np.asarray(metric(pred, true), dtype=np.float64)
            if not np.allclose(metrics, recomputed, rtol=1e-5, atol=1e-8, equal_nan=True):
                raise RuntimeError("metrics.npy does not reproduce from packaged pred/true")
            metrics_recomputed += 1
            rows_audited.append({"logical_run_id": logical, "attempt_id": attempt.name,
                                 "execution_track": receipt.get("execution_track", "frozen_protocol"),
                                 "execution_device": device,
                                 "pred_sha256": receipt["sha256"]["pred.npy"],
                                 "true_sha256": receipt["sha256"]["true.npy"],
                                 "metrics_sha256": receipt["sha256"]["metrics.npy"],
                                 "metrics": [float(value) for value in metrics]})
        except Exception as error:
            issues.append(f"{logical}: {type(error).__name__}: {error}")
    return ({"name": name, "manifest_rows": len(manifest_rows), "attempts_found": len(found),
             "attempts_audited": len(rows_audited), "missing_attempts": len(missing),
             "artifact_hashes_checked": file_hashes_checked,
             "metrics_recomputed": metrics_recomputed, "issues": issues,
             "rows": rows_audited}, found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p4-root", type=Path, default=ROOT / "reproduction/results/phase1/p4-partial-20260929/attempts")
    parser.add_argument("--bridge-root", type=Path, default=ROOT / "reproduction/results/phase1/local-5060-bridge-v1")
    parser.add_argument("--bridge-manifest", type=Path, default=ROOT / "reproduction/results/phase1/local-5060-manifests/bridge-v1.csv")
    parser.add_argument("--reference-index", type=Path, default=ROOT / "reproduction/results/phase1/local-5060-manifests/bridge-p4-reference-index.csv")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    registry = {row["asset"]: row for row in rows(ROOT / "reproduction/results/dataset_registry.csv")}
    p4_manifest = rows(MANIFESTS["B_screen"])
    p4_rows = [row for row in p4_manifest if (args.p4_root / f"{logical_id(row)}__attempt-01").is_dir()]
    p4_report, p4_paths = audit_batch("P4 partial B_screen", p4_rows, args.p4_root, registry, config, allow_partial=True)
    bridge_manifest = read_csv(args.bridge_manifest)
    bridge_report, bridge_paths = audit_batch("RTX 5060 Ti bridge", bridge_manifest, args.bridge_root,
                                              registry, config, allow_partial=False)

    # Verify execution provenance independently of per-attempt artifact receipts.
    provenance_issues: list[str] = []
    p4_archive = args.p4_root.parent
    p4_auth_path = p4_archive / "authorization_stage_b.json"
    p4_env_path = p4_archive / "environment_fingerprint.json"
    p4_auth = json.loads(p4_auth_path.read_text(encoding="utf-8"))
    if sha256(p4_env_path) != p4_auth.get("environment_fingerprint_sha256"):
        provenance_issues.append("P4 environment fingerprint hash differs from authorization")
    if p4_auth.get("gpu_model") != "Tesla P4":
        provenance_issues.append("P4 archive authorization does not name Tesla P4")
    local_manifests = args.bridge_manifest.parent
    local_auth_path = local_manifests / "authorization-local-5060-bridge-v1.json"
    local_env_path = local_manifests / "environment_fingerprint.json"
    local_auth = json.loads(local_auth_path.read_text(encoding="utf-8"))
    if sha256(local_env_path) != local_auth.get("environment_fingerprint_sha256"):
        provenance_issues.append("RTX 5060 environment fingerprint hash differs from authorization")
    if local_auth.get("gpu_model") != "NVIDIA GeForce RTX 5060 Ti":
        provenance_issues.append("bridge authorization does not name RTX 5060 Ti")
    bridge_meta = json.loads((args.bridge_root / "batch_meta.json").read_text(encoding="utf-8"))
    if bridge_meta.get("git_commit") != local_auth.get("git_commit"):
        provenance_issues.append("bridge batch metadata and authorization commits differ")
    if bridge_meta.get("environment_fingerprint_sha256") != sha256(local_env_path):
        provenance_issues.append("bridge batch metadata does not bind the audited environment fingerprint")
    if bridge_meta.get("authorization_sha256") != sha256(local_auth_path):
        provenance_issues.append("bridge batch metadata does not bind the audited authorization")

    p4_by_id = {row["logical_run_id"]: row for row in p4_report["rows"]}
    bridge_by_id = {row["logical_run_id"]: row for row in bridge_report["rows"]}
    p4_ref = {row["logical_run_id"]: row for row in read_csv(args.reference_index)}
    paired = []
    pair_issues = []
    abs_by_metric: dict[str, list[float]] = defaultdict(list)
    rel_by_metric: dict[str, list[float]] = defaultdict(list)
    prediction_rel_norms: list[float] = []
    prediction_max_abs: list[float] = []
    truth_bitwise_equal = 0
    for logical, bridge_row in bridge_by_id.items():
        if logical not in p4_by_id or logical not in p4_ref:
            pair_issues.append(f"{logical}: missing audited P4 result or reference-index entry")
            continue
        reference = p4_ref[logical]
        p4_path = ROOT / Path(reference["p4_attempt"])
        p4_receipt_path = p4_path / "receipt.json"
        if sha256(p4_receipt_path) != reference["receipt_sha256"]:
            pair_issues.append(f"{logical}: P4 reference-index receipt hash mismatch")
            continue
        p4_receipt = json.loads(p4_receipt_path.read_text(encoding="utf-8"))
        bridge_path = bridge_paths[logical]
        p4_pred = np.load(p4_path / "pred.npy", allow_pickle=False)
        bridge_pred = np.load(bridge_path / "pred.npy", allow_pickle=False)
        p4_true = np.load(p4_path / "true.npy", allow_pickle=False)
        bridge_true = np.load(bridge_path / "true.npy", allow_pickle=False)
        if p4_pred.shape != bridge_pred.shape or p4_true.shape != bridge_true.shape:
            pair_issues.append(f"{logical}: P4/5060 array shapes differ")
            continue
        true_equal = bool(np.array_equal(p4_true, bridge_true))
        truth_bitwise_equal += int(true_equal)
        if not true_equal:
            pair_issues.append(f"{logical}: P4/5060 true arrays are not bitwise equal")
        p4_values = np.asarray(p4_by_id[logical]["metrics"], dtype=np.float64)
        bridge_values = np.asarray(bridge_row["metrics"], dtype=np.float64)
        metric_diff = np.abs(bridge_values - p4_values)
        metric_relative = np.divide(metric_diff, np.abs(p4_values),
                                    out=np.full_like(metric_diff, np.nan), where=np.abs(p4_values) > 1e-12)
        for index, metric_name in enumerate(METRIC_NAMES):
            abs_by_metric[metric_name].append(float(metric_diff[index]))
            if np.isfinite(metric_relative[index]):
                rel_by_metric[metric_name].append(float(metric_relative[index]))
        prediction_diff = bridge_pred.astype(np.float64) - p4_pred.astype(np.float64)
        p4_norm = float(np.linalg.norm(p4_pred.astype(np.float64).ravel()))
        rel_norm = float(np.linalg.norm(prediction_diff.ravel()) / p4_norm) if p4_norm > 1e-12 else None
        if rel_norm is not None:
            prediction_rel_norms.append(rel_norm)
        max_abs = float(np.max(np.abs(prediction_diff)))
        prediction_max_abs.append(max_abs)
        paired.append({"logical_run_id": logical, "asset": next(row["asset"] for row in bridge_manifest if logical_id(row) == logical),
                       "model": next(row["model"] for row in bridge_manifest if logical_id(row) == logical),
                       "horizon": next(row["horizon"] for row in bridge_manifest if logical_id(row) == logical),
                       "shapes": list(p4_pred.shape), "true_bitwise_equal": true_equal,
                       "true_sha256_p4": p4_receipt["sha256"]["true.npy"],
                       "true_sha256_5060": bridge_row["true_sha256"],
                       "p4_metrics": p4_values.tolist(), "rtx5060_metrics": bridge_values.tolist(),
                       "absolute_metric_differences": metric_diff.tolist(),
                       "relative_metric_differences": [float(value) if np.isfinite(value) else None for value in metric_relative],
                       "prediction_max_abs_difference": max_abs,
                       "prediction_relative_l2_difference": rel_norm})

    metric_summary = {name: {"absolute_difference": summarize(abs_by_metric[name]),
                             "relative_difference": summarize(rel_by_metric[name]) if rel_by_metric[name] else None}
                      for name in METRIC_NAMES}
    report = {
        "title": "Phase 1 completed-artifact audit and P4 vs RTX 5060 Ti bridge",
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "scientific_status": "descriptive audit only; no equivalence or pooling claim",
        "protocol_version": "phase1-v1.1-2026-09-19",
        "execution_tracks": {
            "P4": {"commit": p4_auth["git_commit"], "gpu": p4_auth.get("gpu_model"),
                   "driver": json.loads(p4_env_path.read_text(encoding="utf-8"))["observed"].get("driver_version"),
                   "torch": json.loads(p4_env_path.read_text(encoding="utf-8"))["observed"].get("torch"),
                   "python": json.loads(p4_env_path.read_text(encoding="utf-8"))["observed"].get("python")},
            "RTX_5060_Ti": {"commit": local_auth["git_commit"], "gpu": local_auth.get("gpu_model"),
                            "driver": json.loads(local_env_path.read_text(encoding="utf-8"))["observed"].get("driver_version"),
                            "torch": json.loads(local_env_path.read_text(encoding="utf-8"))["observed"].get("torch"),
                            "python": json.loads(local_env_path.read_text(encoding="utf-8"))["observed"].get("python")}},
        "p4_audit": p4_report, "rtx5060_bridge_audit": bridge_report,
        "provenance_issues": provenance_issues, "paired_comparison_issues": pair_issues,
        "paired_count": len(paired), "truth_bitwise_equal_count": truth_bitwise_equal,
        "metric_difference_summary": metric_summary,
        "prediction_difference_summary": {
            "relative_l2": summarize(prediction_rel_norms) if prediction_rel_norms else None,
            "max_abs": summarize(prediction_max_abs) if prediction_max_abs else None},
        "paired_rows": paired,
        "interpretation": [
            "Receipt hashes, exact frozen data/split/key identity, finite arrays, target alignment, and metrics recomputation are integrity checks, not evidence of cross-GPU numerical equivalence.",
            "The paired sample is small and selected for bridge coverage; its overlapping test windows are dependent. No p-values or equivalence claim are made.",
            "No precommitted numerical tolerance was found in the approved bridge record (the threshold is TBD). Therefore pooling P4 and RTX 5060 outputs remains unauthorized.",
            "A hardware difference cannot be repaired by post-hoc rescaling. For a single-stack confirmatory B_screen, all 300 fits must be run on one approved stack, or a separate prospective amendment must define an equivalence margin and decision rule."
        ]}
    report["summary"] = {
        "p4_manifest_total": len(p4_manifest), "p4_completed_found": p4_report["attempts_found"],
        "p4_validated": p4_report["attempts_audited"],
        "p4_missing_planned": len(p4_manifest) - p4_report["attempts_found"],
        "missing_p4_models": dict(Counter(row["model"] for row in p4_manifest
                                           if logical_id(row) not in p4_paths)),
        "bridge_manifest_total": len(bridge_manifest), "bridge_validated": bridge_report["attempts_audited"],
        "integrity_issues": len(p4_report["issues"]) + len(bridge_report["issues"]) + len(provenance_issues) + len(pair_issues),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        stream.write(rendered + "\n")
    print(json.dumps(report["summary"] | {"paired_count": report["paired_count"],
                     "truth_bitwise_equal_count": truth_bitwise_equal,
                     "metric_difference_summary": metric_summary,
                     "provenance_issues": provenance_issues,
                     "p4_issues": p4_report["issues"][:5],
                     "bridge_issues": bridge_report["issues"][:5],
                     "paired_issues": pair_issues[:5], "report": str(args.report)},
                     indent=2, allow_nan=False))
    return 0 if report["summary"]["integrity_issues"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
