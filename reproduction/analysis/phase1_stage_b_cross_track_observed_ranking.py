"""Build a complete, explicitly mixed-track descriptive Stage B leaderboard.

The fixed 300-setting manifest is joined to its eligible P4 or RTX 8000
attempt. The output ranks observed model+execution-track artifacts by
asset-equal normalized test MSE within the frozen cohort and horizon. It also
reports the exact multiplicative RTX 8000 TimesNet shift that would tie the
best observed P4 model. That shift is a sensitivity threshold, not an
estimated hardware correction or equivalence bound.

No inferential test, confidence interval, track pooling claim, model-only
causal ranking, score-based data selection, or resource decision is produced.
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
MANIFEST = PHASE1 / "B_screen.manifest.csv"
REGISTRY = ROOT / "reproduction" / "results" / "dataset_registry.csv"
P4_ROOT = PHASE1 / "p4-partial-20260929" / "attempts"
RTX8K_ROOT = PHASE1 / "rtx8000-intake-20260929-v2" / "attempts"
OUT_DIR = ROOT / "reports" / "phase1"
ASSET_OUT = OUT_DIR / "stage-b-cross-track-asset-metrics-20260930.csv"
RANK_OUT = OUT_DIR / "stage-b-cross-track-observed-ranking-20260930.csv"
PAIR_OUT = OUT_DIR / "stage-b-cross-track-pairwise-descriptive-20260930.csv"
SENS_OUT = OUT_DIR / "stage-b-cross-track-timesnet-sensitivity-20260930.csv"
MODELS = (
    "revin-DLinear",
    "DeReFusion",
    "revin-PatchTST",
    "revin-iTransformer",
    "revin-TimesNet",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_package(run_id: str, model: str) -> tuple[str, dict, Path]:
    attempt_id = f"{run_id}__attempt-01"
    candidates = [("P4", P4_ROOT / attempt_id), ("RTX8000", RTX8K_ROOT / attempt_id)]
    found = [(track, folder) for track, folder in candidates if (folder / "receipt.json").is_file()]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one selected-track receipt for {run_id}; found {len(found)}")
    track, folder = found[0]
    if model != "revin-TimesNet" and track != "P4":
        raise RuntimeError(f"non-TimesNet setting unexpectedly outside P4: {run_id}")
    receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("logical_run_id") != run_id or receipt.get("attempt_id") != attempt_id:
        raise RuntimeError(f"receipt identity mismatch: {run_id}")
    if receipt.get("status") != "completed_unreviewed" or receipt.get("stage") != "B_screen":
        raise RuntimeError(f"receipt is not an eligible Stage B completed package: {run_id}")
    return track, receipt, folder


def verified_artifact(folder: Path, receipt: dict, name: str) -> Path:
    path = folder / name
    if not path.is_file():
        raise RuntimeError(f"missing {name}: {folder}")
    expected = receipt.get("sha256", {}).get(name)
    if not expected or sha256(path) != expected:
        raise RuntimeError(f"{name} hash mismatch: {folder}")
    return path


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    registry = pd.read_csv(REGISTRY).set_index("asset")
    required = {"asset", "model", "horizon", "seed", "stage"}
    if not required.issubset(manifest.columns):
        raise RuntimeError(f"manifest is missing columns: {required - set(manifest.columns)}")
    manifest["logical_run_id"] = manifest.apply(
        lambda row: f"B_screen_{row['asset']}_{row['model']}_h{int(row['horizon'])}_s{int(row['seed'])}",
        axis=1,
    )
    if len(manifest) != 300 or manifest["logical_run_id"].nunique() != 300:
        raise RuntimeError("expected the frozen 300-setting manifest")
    if set(manifest["model"]) != set(MODELS):
        raise RuntimeError("unexpected or missing model in frozen manifest")
    if set(manifest["horizon"].astype(int)) != {1, 24} or set(manifest["seed"].astype(int)) != {2021}:
        raise RuntimeError("unexpected horizon or seed in frozen manifest")

    rows: list[dict] = []
    for row in manifest.itertuples(index=False):
        run_id = str(row.logical_run_id)
        track, receipt, folder = load_package(run_id, str(row.model))
        if receipt.get("dataset_sha256") != str(registry.loc[row.asset, "sha256"]):
            raise RuntimeError(f"registry dataset hash mismatch: {run_id}")
        keys_hash = receipt.get("prediction_keys_sha256")
        if not keys_hash:
            raise RuntimeError(f"missing prediction-key hash: {run_id}")

        pred_path = verified_artifact(folder, receipt, "pred.npy")
        true_path = verified_artifact(folder, receipt, "true.npy")
        metrics_path = verified_artifact(folder, receipt, "metrics.npy")
        pred = np.load(pred_path, allow_pickle=False)
        true = np.load(true_path, allow_pickle=False)
        stored = np.load(metrics_path, allow_pickle=False).reshape(-1)
        if pred.shape != true.shape or list(pred.shape) != receipt["shapes"].get("pred.npy"):
            raise RuntimeError(f"prediction/target shape mismatch: {run_id}")
        if list(true.shape) != receipt["shapes"].get("true.npy") or stored.shape != (6,):
            raise RuntimeError(f"receipt-bound shape mismatch: {run_id}")
        if not np.isfinite(pred).all() or not np.isfinite(true).all():
            raise RuntimeError(f"nonfinite arrays: {run_id}")
        error = pred.astype(np.float64) - true.astype(np.float64)
        mae = float(np.mean(np.abs(error)))
        mse = float(np.mean(np.square(error)))
        rmse = float(np.sqrt(mse))
        values = np.asarray([mae, mse, rmse])
        if not np.isfinite(values).all() or not np.allclose(
            values, stored[[0, 1, 2]], rtol=2e-5, atol=2e-7
        ):
            raise RuntimeError(f"recomputed MAE/MSE/RMSE mismatch: {run_id}")

        rows.append(
            {
                "logical_run_id": run_id,
                "attempt_id": receipt["attempt_id"],
                "execution_track": track,
                "cohort": str(registry.loc[row.asset, "cohort"]),
                "asset": str(row.asset),
                "model": str(row.model),
                "horizon": int(row.horizon),
                "seed": int(row.seed),
                "test_values": int(error.size),
                "pred_shape": "x".join(str(dim) for dim in pred.shape),
                "true_shape": "x".join(str(dim) for dim in true.shape),
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "prediction_keys_sha256": keys_hash,
                "true_sha256": receipt["sha256"]["true.npy"],
                "dataset_sha256": receipt["dataset_sha256"],
                "split_manifest_id": receipt["split_manifest_id"],
            }
        )

    data = pd.DataFrame(rows).sort_values(
        ["cohort", "horizon", "asset", "model"], kind="stable"
    )
    if len(data) != 300 or data["logical_run_id"].nunique() != 300:
        raise RuntimeError("the as-run panel does not contain exactly 300 unique fits")

    pair_groups = data.groupby(["asset", "horizon"], sort=True)
    if pair_groups.ngroups != 60:
        raise RuntimeError("expected 60 asset x horizon comparison groups")
    for key, group in pair_groups:
        if len(group) != 5 or set(group["model"]) != set(MODELS):
            raise RuntimeError(f"incomplete model group: {key}")
        for column in (
            "prediction_keys_sha256",
            "true_sha256",
            "dataset_sha256",
            "split_manifest_id",
            "test_values",
            "pred_shape",
            "true_shape",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise RuntimeError(f"{column} mismatch across models for {key}")

    rank_rows: list[dict] = []
    for (cohort, horizon), group in data.groupby(["cohort", "horizon"], sort=True):
        summary = group.groupby("model", sort=False).agg(
            assets=("asset", "nunique"),
            mean_asset_mse=("mse", "mean"),
            median_asset_mse=("mse", "median"),
            min_asset_mse=("mse", "min"),
            max_asset_mse=("mse", "max"),
            mean_asset_mae=("mae", "mean"),
            mean_asset_rmse=("rmse", "mean"),
        ).reset_index()
        summary["rank_lowest_mse_is_1"] = summary["mean_asset_mse"].rank(method="min").astype(int)
        for model in MODELS:
            g = group.loc[group["model"] == model]
            by_track = g["execution_track"].value_counts().to_dict()
            item = summary.loc[summary["model"] == model].iloc[0].to_dict()
            item.update(
                {
                    "cohort": cohort,
                    "horizon": int(horizon),
                    "p4_fits": int(by_track.get("P4", 0)),
                    "rtx8000_fits": int(by_track.get("RTX8000", 0)),
                    "interpretation": "observed model+execution-track artifact rank; descriptive only",
                }
            )
            rank_rows.append(item)

    pair_rows: list[dict] = []
    for (cohort, horizon), group in data.groupby(["cohort", "horizon"], sort=True):
        wide = group.pivot(index="asset", columns="model", values="mse")
        for a, b in itertools.combinations(MODELS, 2):
            delta = wide[a] - wide[b]
            pair_rows.append(
                {
                    "cohort": cohort,
                    "horizon": int(horizon),
                    "model_a": a,
                    "model_b": b,
                    "assets": int(delta.notna().sum()),
                    "mean_asset_mse_delta_a_minus_b": float(delta.mean()),
                    "median_asset_mse_delta_a_minus_b": float(delta.median()),
                    "assets_a_lower_mse": int((delta < 0).sum()),
                    "assets_tied_mse": int((delta == 0).sum()),
                    "assets_b_lower_mse": int((delta > 0).sum()),
                    "warning": "TimesNet execution track differs for some cells; descriptive only",
                }
            )

    sensitivity_rows: list[dict] = []
    for (cohort, horizon), group in data.groupby(["cohort", "horizon"], sort=True):
        n = int(group["asset"].nunique())
        expected_n = {"original-10": 10, "c1-20": 20}.get(str(cohort))
        if expected_n is None or n != expected_n:
            raise RuntimeError(f"unexpected asset count for {(cohort, horizon)}: {n}")
        times = group.loc[group["model"] == "revin-TimesNet"]
        p4_values = times.loc[times["execution_track"] == "P4", "mse"]
        rtx_values = times.loc[times["execution_track"] == "RTX8000", "mse"]
        competitor = (
            group.loc[group["model"] != "revin-TimesNet"]
            .groupby("model")["mse"]
            .mean()
            .sort_values()
        )
        best_model = str(competitor.index[0])
        best_mean = float(competitor.iloc[0])
        rtx_sum = float(rtx_values.sum())
        p4_sum = float(p4_values.sum())
        if rtx_sum == 0:
            raise RuntimeError(f"cannot calculate TimesNet sensitivity for {(cohort, horizon)}")
        break_even = (best_mean * n - p4_sum) / rtx_sum - 1.0
        current_mean = float(times["mse"].mean())
        sensitivity_rows.append(
            {
                "cohort": cohort,
                "horizon": int(horizon),
                "assets": n,
                "timesnet_p4_fits": int(len(p4_values)),
                "timesnet_rtx8000_fits": int(len(rtx_values)),
                "observed_timesnet_mean_asset_mse": current_mean,
                "best_observed_other_model": best_model,
                "best_observed_other_mean_asset_mse": best_mean,
                "rtx8000_timesnet_multiplicative_shift_to_tie_best_other": break_even,
                "interpretation": "hypothetical break-even sensitivity; not an estimated hardware correction",
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(ASSET_OUT, index=False, float_format="%.12g")
    pd.DataFrame(rank_rows).sort_values(
        ["cohort", "horizon", "rank_lowest_mse_is_1", "model"], kind="stable"
    ).to_csv(RANK_OUT, index=False, float_format="%.12g")
    pd.DataFrame(pair_rows).to_csv(PAIR_OUT, index=False, float_format="%.12g")
    pd.DataFrame(sensitivity_rows).to_csv(SENS_OUT, index=False, float_format="%.12g")

    print(
        json.dumps(
            {
                "fits_validated": len(data),
                "paired_asset_horizon_groups": pair_groups.ngroups,
                "rank_rows": len(rank_rows),
                "pairwise_descriptive_rows": len(pair_rows),
                "timesnet_sensitivity_rows": len(sensitivity_rows),
                "pvalues_or_confidence_intervals": "none",
                "equivalence_or_hardware_correction_claim": "none",
                "outputs": {
                    "asset_metrics": str(ASSET_OUT),
                    "observed_ranking": str(RANK_OUT),
                    "pairwise_descriptive": str(PAIR_OUT),
                    "timesnet_sensitivity": str(SENS_OUT),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
