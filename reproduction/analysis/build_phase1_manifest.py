"""Validate the frozen Phase 1 grid and optionally write run manifests.

This script never starts training.  It checks data hashes, model availability,
stage cardinalities and explicit rolling-origin boundaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "reproduction/configs/phase1_modern_baselines.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_assets(stage: dict, config: dict, registry: list[dict]) -> list[str]:
    selector = stage["assets"]
    if selector == "calibration_assets":
        return list(config[selector])
    if selector == "sentinel_assets":
        return list(config[selector])
    if selector == "confirmatory_cohorts":
        cohorts = set(config[selector])
        return sorted(row["asset"] for row in registry if row["cohort"] in cohorts)
    raise ValueError(f"unknown asset selector: {selector}")


def resolve_models(stage: dict, config: dict) -> list[str]:
    selector = stage["models"]
    return list(config[selector]) if isinstance(selector, str) else list(selector)


def model_path(name: str) -> Path:
    if name == "DeReFusion":
        return ROOT / "models/derefusion/DeReFusion.py"
    return ROOT / "models" / f"{name}.py"


def validate_registry(config: dict, registry: list[dict]) -> tuple[dict[str, dict], list[str]]:
    by_asset = {row["asset"]: row for row in registry}
    errors: list[str] = []
    required = set(config["calibration_assets"] + config["sentinel_assets"])
    required.update(
        row["asset"]
        for row in registry
        if row["cohort"] in set(config["confirmatory_cohorts"])
    )
    for asset in sorted(required):
        row = by_asset.get(asset)
        if row is None:
            errors.append(f"missing registry row: {asset}")
            continue
        path = ROOT / row["file"]
        if not path.is_file():
            errors.append(f"missing data file: {row['file']}")
            continue
        actual = sha256(path)
        if actual != row["sha256"]:
            errors.append(f"hash mismatch: {asset} expected={row['sha256']} actual={actual}")
    return by_asset, errors


def build_rows(stage_name: str, stage: dict, config: dict, registry: list[dict]) -> list[dict]:
    assets = resolve_assets(stage, config, registry)
    models = resolve_models(stage, config)
    rows = []
    for asset, model, horizon, seed, origin in product(
        assets, models, stage["horizons"], stage["seeds"], stage["origins"]
    ):
        train_end = val_end = test_end = ""
        if stage.get("split_mode") == "dates":
            test_year = int(origin)
            train_end = f"{test_year - 1}-01-01"
            val_end = f"{test_year}-01-01"
            test_end = f"{test_year + 1}-01-01"
        rows.append(
            {
                "protocol_version": config["protocol_version"],
                "stage": stage_name,
                "asset": asset,
                "model": model,
                "horizon": horizon,
                "seed": seed,
                "origin": origin,
                "split_mode": stage.get("split_mode", "ratio"),
                "train_end": train_end,
                "val_end": val_end,
                "test_end": test_end,
                "status": "planned",
                "block_reason": "",
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--write", action="store_true", help="write deterministic CSV manifests")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    registry_path = ROOT / config["dataset_registry"]
    registry = load_registry(registry_path)
    _, errors = validate_registry(config, registry)

    missing_models = [name for name in config["available_models"] if not model_path(name).is_file()]
    errors.extend(f"missing model implementation: {name}" for name in missing_models)

    all_rows: dict[str, list[dict]] = {}
    for stage_name, stage in config["stages"].items():
        rows = build_rows(stage_name, stage, config, registry)
        all_rows[stage_name] = rows
        expected = int(stage["expected_fits"])
        if len(rows) != expected:
            errors.append(f"{stage_name}: expected {expected} fits, built {len(rows)}")

    print(f"protocol={config['protocol_version']}")
    for stage_name, rows in all_rows.items():
        blocked = sum(row["status"] == "blocked" for row in rows)
        print(f"{stage_name}: fits={len(rows)} blocked={blocked}")
    print("declared gaps=" + ", ".join(config["declared_gaps"]))

    if errors:
        for error in errors:
            print("ERROR:", error)
        return 2

    if args.write:
        output = ROOT / "reproduction/results/phase1"
        for stage_name, rows in all_rows.items():
            write_manifest(output / f"{stage_name}.manifest.csv", rows)
        print(f"wrote manifests to {output.relative_to(ROOT)}")
    else:
        print("dry-run only; pass --write to create manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
