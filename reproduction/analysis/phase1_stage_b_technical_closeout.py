"""Summarize Stage B receipt coverage and per-track training runtimes.

Reads the frozen manifests, accepted intake reports, receipts, and run logs.
Never reads or aggregates test metric values. P4 and RTX8000 runtimes are
reported in separate groups because they are distinct execution tracks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "reproduction" / "results" / "phase1"
MODELS = ("revin-DLinear", "DeReFusion", "revin-PatchTST", "revin-iTransformer", "revin-TimesNet")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def run_id(row: dict[str, str]) -> str:
    return f"{row['stage']}_{row['asset']}_{row['model']}_h{row['horizon']}_s{row['seed']}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", type=Path, default=PHASE)
    parser.add_argument(
        "--p4-report",
        type=Path,
        default=PHASE / "intake-p4-20260929-v1" / "report.json",
    )
    parser.add_argument(
        "--rtx-report",
        type=Path,
        default=PHASE / "rtx8000-intake-review-20260930-complete" / "report.json",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "reports" / "phase1" / "stage-b-runtime-by-track-20260930.csv",
    )
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=ROOT / "reports" / "phase1" / "stage-b-runtime-setting-ledger-20260930.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase = args.phase_root.resolve()
    b_rows = read_csv(phase / "B_screen.manifest.csv")
    supplement_rows = read_csv(phase / "supplemental-manifests" / "rtx8000-timesnet-remaining-v1.csv")
    p4_rows = read_csv(phase / "intake-p4-20260929-v1" / "selected-manifest.csv")
    rtx_rows = read_csv(phase / "rtx8000-intake-review-20260930-complete" / "selected-manifest.csv")
    identity = lambda rows: {run_id(row) for row in rows}
    b_ids, p4_ids, rtx_ids, supplement_ids = map(identity, (b_rows, p4_rows, rtx_rows, supplement_rows))
    if len(b_rows) != 300 or len(b_ids) != 300:
        raise ValueError("frozen Stage B manifest is not exactly 300 unique settings")
    if len(p4_rows) != 245 or len(p4_ids) != 245:
        raise ValueError("P4 intake manifest is not exactly 245 unique settings")
    if len(rtx_rows) != 55 or len(rtx_ids) != 55 or len(supplement_ids) != 55:
        raise ValueError("RTX8000 complete intake/supplement is not exactly 55 unique settings")
    if p4_ids & rtx_ids or p4_ids | rtx_ids != b_ids or supplement_ids != b_ids - p4_ids or rtx_ids != supplement_ids:
        raise ValueError("P4 and RTX8000 intake identities do not exactly cover the frozen manifest")

    sources = (
        ("P4 main partial execution", args.p4_report.resolve(), p4_ids, "frozen_protocol"),
        ("RTX8000 nonconfirmatory supplement", args.rtx_report.resolve(), rtx_ids, "nonconfirmatory_remote_gpu_supplement"),
    )
    detail: list[dict[str, Any]] = []
    for source, report_path, expected_ids, expected_track in sources:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary = report["summary"]
        if report.get("status") != "completed_descriptive_only" or report.get("issues"):
            raise ValueError(f"{source}: intake is not completed issue-free descriptive intake")
        if summary.get("eligible_packages") != len(expected_ids):
            raise ValueError(f"{source}: intake eligible count does not match expected IDs")
        attempts = report.get("attempts", [])
        seen: set[str] = set()
        for item in attempts:
            rid = item["logical_run_id"]
            if rid in seen:
                raise ValueError(f"{source}: duplicate intake run {rid}")
            seen.add(rid)
            if rid not in expected_ids or item.get("status") != "eligible":
                raise ValueError(f"{source}: unexpected or ineligible intake run {rid}")
            attempt_path = Path(item["attempt_path"])
            receipt = json.loads((attempt_path / "receipt.json").read_text(encoding="utf-8"))
            if receipt.get("logical_run_id") != rid or receipt.get("status") != "completed_unreviewed":
                raise ValueError(f"{source}: receipt status/identity mismatch for {rid}")
            if receipt.get("execution_track", "frozen_protocol") != expected_track:
                raise ValueError(f"{source}: execution track mismatch for {rid}")
            run_log = attempt_path / "run.log"
            log_sha = sha256(run_log)
            declared_sha = receipt.get("sha256", {}).get("run.log")
            if declared_sha is None:
                declared_sha = item.get("identity", {}).get("artifact_sha256", {}).get("run.log")
            if declared_sha != log_sha:
                raise ValueError(f"{source}: run.log SHA-256 differs from intake receipt for {rid}")
            log_text = run_log.read_text(encoding="utf-8", errors="replace")
            matches = re.findall(r"train_time:([0-9]+(?:\.[0-9]+)?)s", log_text)
            if len(matches) != 1:
                raise ValueError(f"{source}: expected one train_time record for {rid}, found {len(matches)}")
            model = next((candidate for candidate in MODELS if f"_{candidate}_h" in rid), None)
            horizon_match = re.search(r"_h(1|24)_s2021$", rid)
            if model is None or horizon_match is None:
                raise ValueError(f"{source}: cannot parse frozen model/horizon from {rid}")
            detail.append(
                {
                    "execution_track": source,
                    "logical_run_id": rid,
                    "model": model,
                    "horizon": int(horizon_match.group(1)),
                    "train_time_seconds": float(matches[0]),
                    "run_log_sha256": log_sha,
                }
            )
        if seen != expected_ids:
            raise ValueError(f"{source}: intake report does not cover every expected setting")

    groups: dict[tuple[str, str, int], list[float]] = {}
    for row in detail:
        key = (row["execution_track"], row["model"], row["horizon"])
        groups.setdefault(key, []).append(row["train_time_seconds"])
    summary_rows: list[dict[str, Any]] = []
    for (track, model, horizon), values in sorted(groups.items()):
        summary_rows.append(
            {
                "execution_track": track,
                "model": model,
                "horizon": horizon,
                "eligible_fits": len(values),
                "median_train_time_seconds": statistics.median(values),
                "p90_train_time_seconds": float(np.percentile(values, 90)),
                "min_train_time_seconds": min(values),
                "max_train_time_seconds": max(values),
            }
        )

    if len(detail) != 300:
        raise ValueError(f"expected 300 accepted run logs, found {len(detail)}")
    write_csv(args.summary_output, summary_rows)
    write_csv(args.detail_output, detail)
    print("frozen_settings=300 p4_eligible=245 rtx8000_eligible=55 identity_overlap=0")
    print("intake_issues=0 run_log_sha_checks=300 runtime_field=training_seconds metric_values_used=0")
    print("Cross-track runtime pooling/comparison=not performed")
    print(f"summary={args.summary_output.resolve()}")
    print(f"detail={args.detail_output.resolve()}")


if __name__ == "__main__":
    main()
