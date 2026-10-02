"""Audit every frozen Phase 1 rolling split without starting training."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_provider.data_loader import Dataset_Custom


CONFIG = ROOT / "reproduction/configs/phase1_modern_baselines.json"


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with (ROOT / config["dataset_registry"]).open(encoding="utf-8-sig", newline="") as handle:
        registry = {row["asset"]: row for row in csv.DictReader(handle)}

    errors: list[str] = []
    checks = 0
    for asset in config["sentinel_assets"]:
        row = registry[asset]
        data_path = ROOT / row["file"]
        for test_year in (2023, 2024, 2025):
            boundaries = {
                "train_end": f"{test_year - 1}-01-01",
                "val_end": f"{test_year}-01-01",
                "test_end": f"{test_year + 1}-01-01",
            }
            for horizon in (1, 24):
                args = SimpleNamespace(
                    split_mode="dates",
                    augmentation_ratio=0,
                    **boundaries,
                )
                datasets = {
                    flag: Dataset_Custom(
                        args,
                        str(data_path.parent),
                        flag=flag,
                        size=[96, 48, horizon],
                        features="MS",
                        data_path=data_path.name,
                        target="Close",
                        scale=True,
                        timeenc=0,
                        freq="b",
                    )
                    for flag in ("train", "val", "test")
                }
                checks += 1
                val = datasets["val"]
                test = datasets["test"]
                expected_train_rows = datasets["train"].split_metadata["train_rows"]
                for flag, dataset in datasets.items():
                    fitted_rows = int(dataset.scaler.n_samples_seen_)
                    if fitted_rows != expected_train_rows:
                        errors.append(
                            f"{asset}/{test_year}/h{horizon}/{flag}: scaler saw "
                            f"{fitted_rows} rows, expected {expected_train_rows} training rows"
                        )
                if val.forecast_start_dates.min() < pd.Timestamp(boundaries["train_end"]):
                    errors.append(f"{asset}/{test_year}/h{horizon}: validation label starts early")
                if val.forecast_end_dates.max() >= pd.Timestamp(boundaries["val_end"]):
                    errors.append(f"{asset}/{test_year}/h{horizon}: validation label ends late")
                if test.forecast_start_dates.min() < pd.Timestamp(boundaries["val_end"]):
                    errors.append(f"{asset}/{test_year}/h{horizon}: test label starts early")
                if test.forecast_end_dates.max() >= pd.Timestamp(boundaries["test_end"]):
                    errors.append(f"{asset}/{test_year}/h{horizon}: test label ends late")

    print(f"rolling split combinations checked: {checks}")
    if errors:
        for error in errors:
            print("ERROR:", error)
        return 2
    print("PASS: all forecast labels stay inside their declared validation/test years")
    print("PASS: every scaler was constructed from the corresponding training partition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
