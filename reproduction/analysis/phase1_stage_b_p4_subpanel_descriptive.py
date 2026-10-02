"""Descriptive, P4-only summary of the complete non-TimesNet Stage B subpanel.

This reads the fixed 240-fit panel (4 models x 30 assets x 2 horizons, seed
2021), validates the stored package hashes and recomputes MAE/MSE/RMSE from
pred.npy and true.npy. It does not read or pool RTX 8000 outcomes, perform
inference, rank/select models, or modify experiment artifacts.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PHASE1 = ROOT / "reproduction" / "results" / "phase1"
ATTEMPTS = PHASE1 / "p4-partial-20260929" / "attempts"
MANIFEST = PHASE1 / "B_screen.manifest.csv"
REGISTRY = ROOT / "reproduction" / "results" / "dataset_registry.csv"
OUT_DIR = ROOT / "reports" / "phase1"
ASSET_OUT = OUT_DIR / "stage-b-p4-subpanel-asset-metrics-20260930.csv"
SUMMARY_OUT = OUT_DIR / "stage-b-p4-subpanel-descriptive-20260930.csv"
MODELS = (
    "revin-DLinear",
    "DeReFusion",
    "revin-PatchTST",
    "revin-iTransformer",
)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_receipt(run_id: str) -> tuple[dict, Path]:
    attempt_id = f"{run_id}__attempt-01"
    folder = ATTEMPTS / attempt_id
    receipt_path = folder / "receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"missing receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("logical_run_id") != run_id:
        raise RuntimeError(f"logical_run_id mismatch: {receipt_path}")
    if receipt.get("attempt_id") != attempt_id:
        raise RuntimeError(f"attempt_id mismatch: {receipt_path}")
    if receipt.get("status") != "completed_unreviewed":
        raise RuntimeError(f"receipt is not completed_unreviewed: {receipt_path}")
    if receipt.get("stage") != "B_screen":
        raise RuntimeError(f"unexpected stage: {receipt_path}")
    return receipt, folder


def verify_artifact(folder: Path, receipt: dict, name: str) -> Path:
    path = folder / name
    if not path.is_file():
        raise RuntimeError(f"missing {name}: {folder}")
    expected = receipt.get("sha256", {}).get(name)
    actual = sha256(path)
    if not expected or actual != expected:
        raise RuntimeError(f"{name} SHA-256 mismatch: {folder}")
    return path


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    registry = pd.read_csv(REGISTRY)
    required = {"asset", "model", "horizon", "seed", "stage"}
    if not required.issubset(manifest.columns):
        raise RuntimeError(f"manifest is missing identity columns: {required - set(manifest.columns)}")
    manifest["logical_run_id"] = manifest.apply(
        lambda row: f"B_screen_{row['asset']}_{row['model']}_h{int(row['horizon'])}_s{int(row['seed'])}",
        axis=1,
    )
    panel = manifest.loc[manifest["model"].isin(MODELS)].copy()
    panel["asset"] = panel["asset"].astype(str)
    panel["model"] = panel["model"].astype(str)
    panel["horizon"] = panel["horizon"].astype(int)
    panel["seed"] = panel["seed"].astype(int)

    if len(manifest) != 300 or manifest["logical_run_id"].nunique() != 300:
        raise RuntimeError("frozen Stage B manifest is not 300 unique settings")
    if len(panel) != 240 or panel["logical_run_id"].nunique() != 240:
        raise RuntimeError("fixed P4 non-TimesNet panel is not 240 unique settings")
    if set(panel["model"]) != set(MODELS):
        raise RuntimeError("the fixed four-model roster is incomplete")
    if set(panel["horizon"].unique()) != {1, 24} or set(panel["seed"].unique()) != {2021}:
        raise RuntimeError("unexpected horizons or seed in the fixed panel")
    counts = panel.groupby(["model", "asset", "horizon"], dropna=False).size()
    if len(counts) != 240 or not (counts == 1).all():
        raise RuntimeError("duplicate or missing model x asset x horizon cells")

    registry = registry.set_index("asset")
    assets = set(panel["asset"])
    if not assets.issubset(registry.index):
        raise RuntimeError("a Stage B asset is absent from the dataset registry")
    if set(registry.loc[list(assets), "cohort"]) != {"original-10", "c1-20"}:
        raise RuntimeError("unexpected Stage B cohort labels")

    asset_rows: list[dict] = []
    for row in panel.itertuples(index=False):
        run_id = str(row.logical_run_id)
        receipt, folder = load_receipt(run_id)
        cohort = str(registry.loc[row.asset, "cohort"])
        if receipt.get("dataset_sha256") != str(registry.loc[row.asset, "sha256"]):
            raise RuntimeError(f"dataset SHA mismatch for {run_id}")
        if not receipt.get("prediction_keys_sha256"):
            raise RuntimeError(f"missing prediction-key hash for {run_id}")
        if int(receipt.get("shapes", {}).get("pred.npy", [-1])[0]) <= 0:
            raise RuntimeError(f"invalid prediction shape for {run_id}")

        pred_path = verify_artifact(folder, receipt, "pred.npy")
        true_path = verify_artifact(folder, receipt, "true.npy")
        metrics_path = verify_artifact(folder, receipt, "metrics.npy")
        pred = np.load(pred_path, allow_pickle=False)
        true = np.load(true_path, allow_pickle=False)
        stored = np.load(metrics_path, allow_pickle=False).reshape(-1)
        if pred.shape != true.shape or list(pred.shape) != receipt["shapes"]["pred.npy"]:
            raise RuntimeError(f"prediction/true shape mismatch for {run_id}")
        if list(true.shape) != receipt["shapes"]["true.npy"]:
            raise RuntimeError(f"true shape does not match receipt for {run_id}")
        if stored.shape != (6,):
            raise RuntimeError(f"unexpected metric vector shape for {run_id}")
        if not np.isfinite(pred).all() or not np.isfinite(true).all():
            raise RuntimeError(f"nonfinite prediction or target for {run_id}")

        residual = pred.astype(np.float64) - true.astype(np.float64)
        mae = float(np.mean(np.abs(residual)))
        mse = float(np.mean(np.square(residual)))
        rmse = float(np.sqrt(mse))
        recomputed = np.asarray([mae, mse, rmse])
        if not np.isfinite(recomputed).all() or not np.allclose(
            recomputed, stored[[0, 1, 2]], rtol=2e-5, atol=2e-7
        ):
            raise RuntimeError(f"recomputed MAE/MSE/RMSE do not match metrics.npy for {run_id}")

        asset_rows.append(
            {
                "logical_run_id": run_id,
                "attempt_id": receipt["attempt_id"],
                "cohort": cohort,
                "asset": row.asset,
                "model": row.model,
                "horizon": int(row.horizon),
                "seed": int(row.seed),
                "test_values": int(residual.size),
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "prediction_keys_sha256": receipt["prediction_keys_sha256"],
                "split_manifest_id": receipt["split_manifest_id"],
                "dataset_sha256": receipt["dataset_sha256"],
                "pred_sha256": sha256(pred_path),
                "true_sha256": sha256(true_path),
                "metrics_sha256": sha256(metrics_path),
            }
        )

    asset_df = pd.DataFrame(asset_rows).sort_values(
        ["cohort", "horizon", "asset", "model"], kind="stable"
    )
    if len(asset_df) != 240 or asset_df["logical_run_id"].nunique() != 240:
        raise RuntimeError("validated output does not contain exactly 240 unique fits")
    paired = asset_df.groupby(["asset", "horizon"], sort=True)
    if paired.ngroups != 60:
        raise RuntimeError("the fixed panel does not contain 60 asset x horizon pairs")
    for key, group in paired:
        if set(group["model"]) != set(MODELS) or len(group) != 4:
            raise RuntimeError(f"incomplete model pair group: {key}")
        for column in (
            "prediction_keys_sha256",
            "split_manifest_id",
            "dataset_sha256",
            "true_sha256",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise RuntimeError(f"{column} mismatch in pair group {key}")
        if group["test_values"].nunique(dropna=False) != 1:
            raise RuntimeError(f"test-array shape/size mismatch in pair group {key}")

    summary_rows: list[dict] = []
    for (cohort, horizon, model), group in asset_df.groupby(
        ["cohort", "horizon", "model"], sort=True
    ):
        vals = group["mse"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "cohort": cohort,
                "horizon": int(horizon),
                "model": model,
                "assets": int(group["asset"].nunique()),
                "mean_asset_mse": float(np.mean(vals)),
                "median_asset_mse": float(np.median(vals)),
                "min_asset_mse": float(np.min(vals)),
                "max_asset_mse": float(np.max(vals)),
                "mean_asset_mae": float(group["mae"].mean()),
                "mean_asset_rmse": float(group["rmse"].mean()),
            }
        )
    for (cohort, horizon), group in asset_df.groupby(["cohort", "horizon"], sort=True):
        wide = group.pivot(index="asset", columns="model", values="mse")
        for model_a, model_b in itertools.combinations(MODELS, 2):
            delta = wide[model_a] - wide[model_b]
            summary_rows.append(
                {
                    "cohort": cohort,
                    "horizon": int(horizon),
                    "model": f"{model_a} minus {model_b}",
                    "assets": int(delta.notna().sum()),
                    "mean_asset_mse": float(delta.mean()),
                    "median_asset_mse": float(delta.median()),
                    "min_asset_mse": float(delta.min()),
                    "max_asset_mse": float(delta.max()),
                    "mean_asset_mae": np.nan,
                    "mean_asset_rmse": np.nan,
                    "assets_a_lower_mse": int((delta < 0).sum()),
                    "assets_tied_mse": int((delta == 0).sum()),
                    "assets_b_lower_mse": int((delta > 0).sum()),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["cohort", "horizon", "model"], kind="stable"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    asset_df.to_csv(ASSET_OUT, index=False, float_format="%.12g")
    summary_df.to_csv(SUMMARY_OUT, index=False, float_format="%.12g")

    print(
        json.dumps(
            {
                "fits_validated": len(asset_df),
                "asset_horizon_groups": int(asset_df.groupby(["asset", "horizon"]).ngroups),
                "cohort_horizon_model_rows": int(
                    asset_df.groupby(["cohort", "horizon", "model"]).ngroups
                ),
                "pairwise_descriptive_rows": len(summary_df) - int(
                    asset_df.groupby(["cohort", "horizon", "model"]).ngroups
                ),
                "pvalues_or_confidence_intervals": "none",
                "selected_or_pooled_tracks": "none",
                "asset_metrics_csv": str(ASSET_OUT),
                "descriptive_csv": str(SUMMARY_OUT),
                "asset_metrics_sha256": sha256(ASSET_OUT),
                "descriptive_sha256": sha256(SUMMARY_OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
