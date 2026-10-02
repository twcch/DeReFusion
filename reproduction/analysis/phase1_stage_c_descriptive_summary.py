"""Reproduce descriptive Stage C paired-MSE summaries from frozen receipts.

This script is read-only with respect to experiment artifacts. It validates
receipt bindings and hashes for prediction, target, and metric files, then
writes asset-level and cohort-level descriptive CSVs. It does not bootstrap,
test hypotheses, rank models, or make confirmatory claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "reproduction" / "results" / "phase1"
EXPECTED_AUTH_ID = "local-5060-stage-c-v1"
EXPECTED_TRACK = "confirmatory_local_stage_c"
EXPECTED_STAGE = "C_confirmation"
EXPECTED_MODELS = ("revin-DLinear", "DeReFusion")
EXPECTED_HORIZONS = (1, 24)
EXPECTED_SEEDS = (2022, 2023, 2024)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def logical_id(row: dict[str, str]) -> str:
    return (
        f"{row['stage']}_{row['asset']}_{row['model']}"
        f"_h{row['horizon']}_s{row['seed']}"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty summary: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = args.manifest.resolve()
    auth_path = args.authorization.resolve()
    registry_path = args.registry.resolve()
    stage_root = args.stage_root.resolve()

    manifest = read_csv(manifest_path)
    manifest_sha = sha256(manifest_path)
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    if auth.get("batch_id") != EXPECTED_AUTH_ID:
        raise ValueError("unexpected Stage C batch authorization")
    if auth.get("stages") != [EXPECTED_STAGE]:
        raise ValueError("authorization does not cover only frozen Stage C")
    if auth.get("manifest_sha256") != manifest_sha or auth.get("manifest_count") != len(manifest):
        raise ValueError("Stage C manifest does not match authorization")
    if len(manifest) != 360:
        raise ValueError(f"expected the complete 360-row Stage C manifest, got {len(manifest)}")

    expected_ids = [logical_id(row) for row in manifest]
    if len(set(expected_ids)) != 360:
        raise ValueError("duplicate logical IDs in frozen Stage C manifest")
    expected_grid = {
        (asset, horizon, seed, model)
        for asset in {row["asset"] for row in manifest}
        for horizon in EXPECTED_HORIZONS
        for seed in EXPECTED_SEEDS
        for model in EXPECTED_MODELS
    }
    actual_grid = {
        (row["asset"], int(row["horizon"]), int(row["seed"]), row["model"])
        for row in manifest
    }
    if actual_grid != expected_grid:
        raise ValueError("frozen Stage C manifest differs from the registered comparison grid")

    registry = {row["asset"]: row for row in read_csv(registry_path)}
    assets = {row["asset"] for row in manifest}
    if not assets <= registry.keys():
        raise ValueError("Stage C assets are missing from the frozen dataset registry")

    outcomes: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    checked_files = 0
    for row in manifest:
        run_id = logical_id(row)
        run_dir = stage_root / f"{run_id}__attempt-01"
        receipt_path = run_dir / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("logical_run_id") != run_id:
            raise ValueError(f"{run_id}: receipt logical ID mismatch")
        if receipt.get("status") != "completed_unreviewed":
            raise ValueError(f"{run_id}: receipt is not completed_unreviewed")
        if receipt.get("execution_track") != EXPECTED_TRACK:
            raise ValueError(f"{run_id}: unexpected execution track")
        if receipt.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"{run_id}: receipt manifest hash mismatch")
        if receipt.get("protocol_version") != auth.get("protocol_version"):
            raise ValueError(f"{run_id}: protocol version differs from authorization")
        if "--inverse" in receipt.get("command", "").split():
            raise ValueError(f"{run_id}: inverse scaling was requested; estimand needs review")

        arrays: dict[str, np.ndarray] = {}
        for filename in ("pred.npy", "true.npy", "metrics.npy"):
            path = run_dir / filename
            observed_sha = sha256(path)
            if receipt.get("sha256", {}).get(filename) != observed_sha:
                raise ValueError(f"{run_id}: {filename} SHA-256 differs from receipt")
            array = np.load(path, allow_pickle=False)
            if list(array.shape) != receipt.get("shapes", {}).get(filename):
                raise ValueError(f"{run_id}: {filename} shape differs from receipt")
            arrays[filename] = array
            checked_files += 1

        pred, true, metrics = arrays["pred.npy"], arrays["true.npy"], arrays["metrics.npy"]
        if pred.shape != true.shape or not np.isfinite(pred).all() or not np.isfinite(true).all():
            raise ValueError(f"{run_id}: prediction/target shape or finite-value check failed")
        mse = float(np.mean(np.square(pred - true)))
        # utils.metrics.metric stores [MAE, MSE, RMSE, MAPE, MSPE, R2].
        if len(metrics) < 2 or not np.isclose(mse, float(metrics[1]), rtol=2e-5, atol=1e-7):
            raise ValueError(f"{run_id}: recomputed MSE differs from metrics.npy")

        key = (row["asset"], int(row["horizon"]), int(row["seed"]), row["model"])
        outcomes[key] = {
            "mse": mse,
            "test_values": int(true.size),
            "prediction_keys_sha256": receipt.get("prediction_keys_sha256"),
            "true_sha256": receipt["sha256"]["true.npy"],
            "true": true,
        }

    detail: list[dict[str, Any]] = []
    for asset in sorted(assets):
        cohort = registry[asset]["cohort"]
        for horizon in EXPECTED_HORIZONS:
            for seed in EXPECTED_SEEDS:
                baseline = outcomes[(asset, horizon, seed, "revin-DLinear")]
                derefusion = outcomes[(asset, horizon, seed, "DeReFusion")]
                if (
                    baseline["prediction_keys_sha256"] != derefusion["prediction_keys_sha256"]
                    or baseline["true_sha256"] != derefusion["true_sha256"]
                    or not np.array_equal(baseline["true"], derefusion["true"])
                ):
                    raise ValueError(f"{asset} h{horizon} s{seed}: paired targets/keys differ")
                delta = derefusion["mse"] - baseline["mse"]
                detail.append(
                    {
                        "asset": asset,
                        "cohort": cohort,
                        "horizon": horizon,
                        "seed": seed,
                        "test_values": baseline["test_values"],
                        "mse_revin_dlinear": baseline["mse"],
                        "mse_derefusion": derefusion["mse"],
                        "delta_derefusion_minus_dlinear": delta,
                        "asset_delta_direction": "negative" if delta < 0 else "positive" if delta > 0 else "zero",
                        "prediction_keys_sha256": baseline["prediction_keys_sha256"],
                        "true_sha256": baseline["true_sha256"],
                    }
                )

    cohort_rows: list[dict[str, Any]] = []
    for cohort in ("original-10", "c1-20"):
        cohort_assets = sorted(asset for asset in assets if registry[asset]["cohort"] == cohort)
        expected_count = 10 if cohort == "original-10" else 20
        if len(cohort_assets) != expected_count:
            raise ValueError(f"{cohort}: expected {expected_count} assets, got {len(cohort_assets)}")
        for horizon in EXPECTED_HORIZONS:
            for seed in EXPECTED_SEEDS:
                cell = [
                    row for row in detail
                    if row["cohort"] == cohort and row["horizon"] == horizon and row["seed"] == seed
                ]
                deltas = [row["delta_derefusion_minus_dlinear"] for row in cell]
                cohort_rows.append(
                    {
                        "cohort": cohort,
                        "horizon": horizon,
                        "seed": seed,
                        "assets": len(cell),
                        "mean_normalized_mse_revin_dlinear": float(np.mean([row["mse_revin_dlinear"] for row in cell])),
                        "mean_normalized_mse_derefusion": float(np.mean([row["mse_derefusion"] for row in cell])),
                        "mean_paired_delta_derefusion_minus_dlinear": float(np.mean(deltas)),
                        "assets_delta_negative": sum(value < 0 for value in deltas),
                        "assets_delta_zero": sum(value == 0 for value in deltas),
                        "assets_delta_positive": sum(value > 0 for value in deltas),
                    }
                )

    if len(detail) != 180 or checked_files != 1080:
        raise ValueError("unexpected number of paired outcomes or checked artifact files")
    print(f"validated_settings={len(manifest)} paired_settings={len(detail)} sha_checked_files={checked_files}")
    print("output_scope=descriptive_only bootstrap=not_computed hypothesis_tests=not_computed")
    return detail, cohort_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PHASE / "C_confirmation.manifest.csv")
    parser.add_argument(
        "--authorization",
        type=Path,
        default=PHASE / "local-5060-manifests" / "authorization-local-5060-stage-c-v1.json",
    )
    parser.add_argument("--registry", type=Path, default=ROOT / "reproduction" / "results" / "dataset_registry.csv")
    parser.add_argument("--stage-root", type=Path, default=PHASE / "local-5060-stage-c-v1")
    parser.add_argument(
        "--asset-output",
        type=Path,
        default=ROOT / "reports" / "phase1" / "stage-c-asset-paired-estimates-20260930.csv",
    )
    parser.add_argument(
        "--cohort-output",
        type=Path,
        default=ROOT / "reports" / "phase1" / "stage-c-descriptive-estimates-20260930.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail, cohort = summarize(args)
    write_csv(args.asset_output, detail)
    write_csv(args.cohort_output, cohort)
    print(f"asset_detail={args.asset_output.resolve()}")
    print(f"cohort_summary={args.cohort_output.resolve()}")


if __name__ == "__main__":
    main()
