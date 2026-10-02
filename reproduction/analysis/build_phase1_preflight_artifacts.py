"""Build result-free Phase 1 data and prediction-key artifacts.

This utility reads the frozen Phase 1 configuration, dataset registry, and raw
CSV files only. It does not import or execute a model. B/C share one legacy
ratio test-key universe; D gets explicit date-origin keys. Prediction keys are
one row per forecast window and lead, so overlapping windows remain distinct.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "reproduction/configs/phase1_modern_baselines.json"
OUTPUT_NAMES = (
    "phase1_data_contract.csv",
    "phase1_split_manifest.csv",
    "phase1_prediction_keys.csv.gz",
)
CONTRACT_FIELDS = (
    "protocol_id", "dataset_id", "source_path", "sha256", "schema_version",
    "time_column", "frequency", "timezone_status", "series_group_id",
    "cohort_label", "static_strata", "missingness_summary",
    "nonfinite_summary", "ohlc_status", "price_domain_status",
    "allowed_transformations", "status", "status_reason",
)
SPLIT_FIELDS = (
    "protocol_id", "split_manifest_id", "dataset_id", "series_id", "asset",
    "cohort_label", "stage_scope", "split_mode", "train_end", "val_end",
    "test_end", "train_rows", "val_rows", "test_rows", "seq_len",
    "forecast_horizon", "target", "partition", "window_count",
    "prediction_point_count", "prediction_keys_sha256",
)
KEY_FIELDS = (
    "protocol_id", "split_manifest_id", "dataset_id", "series_id", "asset",
    "cohort_label", "stage_scope", "split_mode", "train_end", "val_end",
    "test_end", "seq_len", "forecast_horizon", "origin_timestamp",
    "horizon", "target_timestamp", "target", "partition", "key_hash",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def read_registry(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_market_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    return header, rows


def dataset_contract(
    protocol_id: str, registry_row: dict[str, str], source: Path
) -> tuple[dict[str, str], list[str]]:
    header, rows = read_market_csv(source)
    asset = registry_row["asset"]
    hash_actual = sha256_file(source)
    dates = [row[0] for row in rows]
    errors: list[str] = []
    if hash_actual != registry_row["sha256"]:
        errors.append(f"{asset}: file SHA-256 does not match frozen registry")
    if header != ["date", "Open", "High", "Low", "Close"]:
        errors.append(f"{asset}: OHLC schema mismatch")
    try:
        parsed_dates = [date.fromisoformat(value) for value in dates]
    except ValueError as error:
        errors.append(f"{asset}: invalid date value ({error})")
        parsed_dates = []
    if parsed_dates and any(a >= b for a, b in zip(parsed_dates, parsed_dates[1:])):
        errors.append(f"{asset}: dates must be strictly increasing and unique")
    if len(rows) != int(registry_row["rows"]):
        errors.append(f"{asset}: row count differs from frozen registry")
    if int(registry_row["missing_or_nonfinite_cells"]):
        errors.append(f"{asset}: registry reports missing or non-finite cells")

    high_violations = int(registry_row["high_envelope_violations"])
    low_violations = int(registry_row["low_envelope_violations"])
    nonpositive_rows = int(registry_row["rows_with_nonpositive_price"])
    issues: list[str] = []
    if high_violations or low_violations:
        issues.append(f"OHLC envelope violations retained: high={high_violations}, low={low_violations}")
    if nonpositive_rows:
        issues.append(f"rows with a non-positive OHLC value retained: {nonpositive_rows}")
    if not issues:
        issues.append("no registry-recorded OHLC envelope or positive-price issue")

    reason = (
        "Frozen source bytes and schema verified. Known data conditions are retained without repair; "
        "the confirmatory target is level-space Close with normalized MSE, and no log-price/return "
        "transformation is authorized. " + "; ".join(issues)
    )
    contract = {
        "protocol_id": protocol_id,
        "dataset_id": f"{asset}@{hash_actual[:16]}",
        "source_path": registry_row["file"],
        "sha256": hash_actual,
        "schema_version": "date-open-high-low-close/v1",
        "time_column": "date",
        "frequency": "daily; asset-specific market/session calendar",
        "timezone_status": "date-only session labels; timezone absent; no cross-asset intraday alignment claim",
        "series_group_id": asset,
        "cohort_label": registry_row["cohort"],
        "static_strata": json.dumps([registry_row["cohort"]], separators=(",", ":")),
        "missingness_summary": f"missing_or_nonfinite_cells={registry_row['missing_or_nonfinite_cells']}",
        "nonfinite_summary": f"missing_or_nonfinite_cells={registry_row['missing_or_nonfinite_cells']}",
        "ohlc_status": f"high_envelope_violations={high_violations};low_envelope_violations={low_violations}",
        "price_domain_status": f"rows_with_nonpositive_price={nonpositive_rows}; raw prices retained",
        "allowed_transformations": "train-only StandardScaler; per-window RevIN; no log/offset/absolute-value/row deletion",
        "status": "eligible",
        "status_reason": reason,
    }
    return contract, errors


@dataclass
class SplitSpec:
    values: dict[str, str]
    first_target_index: int
    test_stop: int
    dates: list[str]


def make_split(
    protocol_id: str,
    dataset_id: str,
    registry_row: dict[str, str],
    dates: list[str],
    split_mode: str,
    forecast_horizon: int,
    seq_len: int,
    target: str,
    stage_scope: str,
    date_boundaries: tuple[str, str, str] | None = None,
) -> SplitSpec:
    n_rows = len(dates)
    if split_mode == "ratio":
        num_train = int(n_rows * 0.7)
        num_test = int(n_rows * 0.2)
        num_val = n_rows - num_train - num_test
        train_stop = num_train
        val_stop = num_train + num_val
        test_stop = n_rows
        test_start = n_rows - num_test
        boundaries = ("", "", "")
    elif split_mode == "dates" and date_boundaries:
        train_end, val_end, test_end = date_boundaries
        date_objects = [date.fromisoformat(value) for value in dates]
        train_stop = next((i for i, value in enumerate(date_objects) if value.isoformat() >= train_end), n_rows)
        val_stop = next((i for i, value in enumerate(date_objects) if value.isoformat() >= val_end), n_rows)
        test_stop = next((i for i, value in enumerate(date_objects) if value.isoformat() >= test_end), n_rows)
        num_train = train_stop
        num_val = val_stop - train_stop
        num_test = test_stop - val_stop
        test_start = val_stop
        boundaries = date_boundaries
    else:
        raise ValueError(f"unsupported split specification: {split_mode} {date_boundaries}")

    if num_train < seq_len + forecast_horizon:
        raise ValueError(f"{registry_row['asset']}: insufficient training rows for h={forecast_horizon}")
    if num_val < forecast_horizon or num_test < forecast_horizon:
        raise ValueError(f"{registry_row['asset']}: insufficient validation/test rows for h={forecast_horizon}")

    window_count = num_test - forecast_horizon + 1
    identity = {
        "protocol_id": protocol_id,
        "dataset_id": dataset_id,
        "split_mode": split_mode,
        "ratio_rule": "floor(70% train), floor(20% test), remainder validation" if split_mode == "ratio" else None,
        "date_boundaries": boundaries,
        "train_rows": num_train,
        "val_rows": num_val,
        "test_rows": num_test,
        "seq_len": seq_len,
        "forecast_horizon": forecast_horizon,
        "target": target,
        "partition": "test",
    }
    split_id = canonical_hash(identity)
    split_row = {
        "protocol_id": protocol_id,
        "split_manifest_id": split_id,
        "dataset_id": dataset_id,
        "series_id": registry_row["asset"],
        "asset": registry_row["asset"],
        "cohort_label": registry_row["cohort"],
        "stage_scope": stage_scope,
        "split_mode": split_mode,
        "train_end": boundaries[0],
        "val_end": boundaries[1],
        "test_end": boundaries[2],
        "train_rows": str(num_train),
        "val_rows": str(num_val),
        "test_rows": str(num_test),
        "seq_len": str(seq_len),
        "forecast_horizon": str(forecast_horizon),
        "target": target,
        "partition": "test",
        "window_count": str(window_count),
        "prediction_point_count": str(window_count * forecast_horizon),
        "prediction_keys_sha256": "",
    }
    return SplitSpec(split_row, test_start, test_stop, dates)


def key_rows(spec: SplitSpec) -> Iterable[dict[str, str]]:
    metadata = spec.values
    horizon = int(metadata["forecast_horizon"])
    seq_len = int(metadata["seq_len"])
    for window in range(int(metadata["window_count"])):
        first_target = spec.first_target_index + window
        origin = spec.dates[first_target - 1]
        for lead in range(1, horizon + 1):
            target_index = first_target + lead - 1
            if target_index >= spec.test_stop:
                raise ValueError(f"prediction key crossed test boundary for {metadata['asset']}")
            target_timestamp = spec.dates[target_index]
            identity = {
                "split_manifest_id": metadata["split_manifest_id"],
                "series_id": metadata["series_id"],
                "origin_timestamp": origin,
                "forecast_horizon": horizon,
                "horizon": lead,
                "target_timestamp": target_timestamp,
                "target": metadata["target"],
                "partition": "test",
            }
            yield {
                "protocol_id": metadata["protocol_id"],
                "split_manifest_id": metadata["split_manifest_id"],
                "dataset_id": metadata["dataset_id"],
                "series_id": metadata["series_id"],
                "asset": metadata["asset"],
                "cohort_label": metadata["cohort_label"],
                "stage_scope": metadata["stage_scope"],
                "split_mode": metadata["split_mode"],
                "train_end": metadata["train_end"],
                "val_end": metadata["val_end"],
                "test_end": metadata["test_end"],
                "seq_len": str(seq_len),
                "forecast_horizon": str(horizon),
                "origin_timestamp": origin,
                "horizon": str(lead),
                "target_timestamp": target_timestamp,
                "target": metadata["target"],
                "partition": "test",
                "key_hash": canonical_hash(identity),
            }


def write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(path: Path, fields: tuple[str, ...], specs: list[SplitSpec]) -> tuple[int, dict[str, str]]:
    row_count = 0
    key_digests = {spec.values["split_manifest_id"]: hashlib.sha256() for spec in specs}
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                for spec in specs:
                    split_id = spec.values["split_manifest_id"]
                    digest = key_digests[split_id]
                    observed = 0
                    for row in key_rows(spec):
                        writer.writerow(row)
                        digest.update((row["key_hash"] + "\n").encode("ascii"))
                        row_count += 1
                        observed += 1
                    if observed != int(spec.values["prediction_point_count"]):
                        raise AssertionError(f"key count mismatch for {spec.values['asset']}")
    return row_count, {key: value.hexdigest() for key, value in key_digests.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write immutable preflight artifacts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduction/results/phase1")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol_id = config["protocol_version"]
    registry_path = ROOT / config["dataset_registry"]
    registry = read_registry(registry_path)
    by_asset = {row["asset"]: row for row in registry}
    if len(by_asset) != len(registry):
        raise SystemExit("dataset registry has duplicate asset IDs; no artifacts written")

    confirmatory = set(config["confirmatory_cohorts"])
    assets = sorted(row["asset"] for row in registry if row["cohort"] in confirmatory)
    sentinels = list(config["sentinel_assets"])
    if not set(sentinels).issubset(assets):
        raise SystemExit("sentinel assets are not a subset of the frozen confirmatory cohorts")

    contracts: list[dict[str, str]] = []
    dates_by_asset: dict[str, list[str]] = {}
    errors: list[str] = []
    for asset in assets:
        row = by_asset[asset]
        source = ROOT / row["file"]
        if not source.is_file():
            errors.append(f"{asset}: missing source file")
            continue
        contract, row_errors = dataset_contract(protocol_id, row, source)
        contracts.append(contract)
        errors.extend(row_errors)
        _, data_rows = read_market_csv(source)
        dates_by_asset[asset] = [item[0] for item in data_rows]
    if errors:
        for error in errors:
            print("BLOCK:", error)
        raise SystemExit("identity/schema preflight failed; no prediction keys written")

    seq_len = int(config["common"]["seq_len"])
    target = "Close"
    split_specs: dict[str, SplitSpec] = {}
    stage_names = ("B_screen", "C_confirmation")
    for stage_name in stage_names:
        stage = config["stages"][stage_name]
        if stage.get("split_mode", "ratio") != "ratio" or stage["origins"] != ["legacy-fixed"]:
            raise SystemExit(f"{stage_name}: expected frozen legacy-fixed ratio split")
        for asset in assets:
            row = by_asset[asset]
            dataset_id = f"{asset}@{row['sha256'][:16]}"
            for forecast_horizon in sorted(set(int(x) for x in stage["horizons"])):
                spec = make_split(
                    protocol_id, dataset_id, row, dates_by_asset[asset], "ratio",
                    forecast_horizon, seq_len, target, stage_name,
                )
                existing = split_specs.get(spec.values["split_manifest_id"])
                if existing:
                    scopes = set(existing.values["stage_scope"].split(";"))
                    scopes.add(stage_name)
                    existing.values["stage_scope"] = ";".join(sorted(scopes))
                else:
                    split_specs[spec.values["split_manifest_id"]] = spec

    stage = config["stages"]["D_temporal_robustness"]
    for asset in sentinels:
        row = by_asset[asset]
        dataset_id = f"{asset}@{row['sha256'][:16]}"
        for origin in stage["origins"]:
            year = int(origin)
            boundaries = (f"{year - 1}-01-01", f"{year}-01-01", f"{year + 1}-01-01")
            for forecast_horizon in sorted(set(int(x) for x in stage["horizons"])):
                spec = make_split(
                    protocol_id, dataset_id, row, dates_by_asset[asset], "dates",
                    forecast_horizon, seq_len, target, "D_temporal_robustness", boundaries,
                )
                if spec.values["split_manifest_id"] in split_specs:
                    raise AssertionError("split identity collision")
                split_specs[spec.values["split_manifest_id"]] = spec

    specs = sorted(
        split_specs.values(),
        key=lambda item: (item.values["stage_scope"], item.values["asset"], int(item.values["forecast_horizon"]), item.values["train_end"]),
    )
    if not args.write:
        total_keys = sum(int(spec.values["prediction_point_count"]) for spec in specs)
        print(f"dry-run: datasets={len(contracts)} splits={len(specs)} test prediction points={total_keys}")
        print("B/C use one shared result-free key set per asset and horizon; D keys are date-origin specific")
        print("No files written. Pass --write to create the three deterministic artifacts.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / name for name in OUTPUT_NAMES]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise SystemExit("refusing to overwrite existing preflight artifacts: " + ", ".join(str(p) for p in existing))

    temp_paths: list[Path] = []
    try:
        for target_path in targets:
            fd, temp_name = tempfile.mkstemp(prefix=target_path.name + ".", suffix=".tmp", dir=output_dir)
            os.close(fd)
            temp_paths.append(Path(temp_name))
        write_csv(temp_paths[0], CONTRACT_FIELDS, contracts)
        key_count, key_digests = write_gzip_csv(temp_paths[2], KEY_FIELDS, specs)
        split_rows = []
        for spec in specs:
            row = dict(spec.values)
            row["prediction_keys_sha256"] = key_digests[row["split_manifest_id"]]
            split_rows.append(row)
        write_csv(temp_paths[1], SPLIT_FIELDS, split_rows)
        if sum(int(row["prediction_point_count"]) for row in split_rows) != key_count:
            raise AssertionError("aggregate prediction-key count mismatch")
        for temp, target_path in zip(temp_paths, targets):
            temp.replace(target_path)
    finally:
        for temp in temp_paths:
            if temp.exists():
                temp.unlink()

    print(f"datasets={len(contracts)} splits={len(specs)} test prediction points={key_count}")
    for path in targets:
        print(f"{path.relative_to(ROOT)} sha256={sha256_file(path)} bytes={path.stat().st_size}")
    print("status=eligible for model-independent key comparison; no training was run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
