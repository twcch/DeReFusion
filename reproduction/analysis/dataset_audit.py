"""Audit every tracked OHLC CSV and write a machine-readable data registry.

The audit is deliberately descriptive: it never rewrites market data.  It checks
schema, date ordering/uniqueness, missing and non-finite values, non-positive
prices, and OHLC envelope consistency.  Known cohorts are labels for evidence
navigation, not claims about data provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "dataset"
EXPECTED = ["date", "Open", "High", "Low", "Close"]

ORIGINAL = {"BABA", "NVO", "TM", "GSPC", "DJI", "SOX", "EURUSD", "USDJPY", "BTCUSD", "ETHUSD"}
GATE_A = {"BYD", "BOE", "EASTMONEY", "YANGHE"}
C1 = {
    "AAPL", "AMZN", "AUDUSD", "FTSE", "GBPUSD", "GDAXI", "GLD", "GOLD", "HSI", "JPM",
    "META", "MSFT", "N225", "RUT", "TLT", "TSLA", "USDCAD", "WMT", "WTI", "XOM",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cohort(asset: str) -> str:
    labels = []
    if asset in ORIGINAL:
        labels.append("original-10")
    if asset in GATE_A:
        labels.append("gate-a-4")
    if asset in C1:
        labels.append("c1-20")
    return "+".join(labels) if labels else "supporting-unregistered"


def audit(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    asset = path.stem.removesuffix("-2016-2025")
    schema_ok = list(df.columns) == EXPECTED
    dates = pd.to_datetime(df["date"], errors="coerce") if "date" in df else pd.Series(dtype="datetime64[ns]")
    numeric = df[[c for c in EXPECTED[1:] if c in df]].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)) if not numeric.empty else np.empty((0, 0), dtype=bool)
    high_bad = int(((numeric["High"] + 1e-12 < numeric[["Open", "Close", "Low"]].max(axis=1))).sum()) if schema_ok else -1
    low_bad = int(((numeric["Low"] - 1e-12 > numeric[["Open", "Close", "High"]].min(axis=1))).sum()) if schema_ok else -1
    nonpositive = int((numeric <= 0).any(axis=1).sum()) if not numeric.empty else -1
    issues = []
    if not schema_ok:
        issues.append("schema")
    if dates.isna().any():
        issues.append("invalid_date")
    if not dates.is_monotonic_increasing:
        issues.append("date_order")
    if dates.duplicated().any():
        issues.append("duplicate_date")
    if numeric.isna().any().any() or (finite.size and not finite.all()):
        issues.append("missing_or_nonfinite")
    if high_bad > 0 or low_bad > 0:
        issues.append("ohlc_envelope")
    if nonpositive > 0:
        issues.append("nonpositive_price")
    return {
        "asset": asset,
        "file": path.as_posix().removeprefix(REPO.as_posix() + "/"),
        "cohort": cohort(asset),
        "rows": len(df),
        "date_start": "" if dates.empty or dates.isna().all() else dates.min().date().isoformat(),
        "date_end": "" if dates.empty or dates.isna().all() else dates.max().date().isoformat(),
        "sha256": sha256(path),
        "schema_ok": schema_ok,
        "date_sorted": bool(dates.is_monotonic_increasing),
        "duplicate_dates": int(dates.duplicated().sum()),
        "missing_or_nonfinite_cells": int(numeric.isna().sum().sum() + (finite.size - finite.sum() if finite.size else 0)),
        "high_envelope_violations": high_bad,
        "low_envelope_violations": low_bad,
        "rows_with_nonpositive_price": nonpositive,
        "issues": ";".join(issues),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "reproduction" / "results" / "dataset_registry.csv")
    args = parser.parse_args()
    rows = [audit(path) for path in sorted(DATASET.glob("*.csv"))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    issue_files = [row for row in rows if row["issues"]]
    print(f"files={len(rows)} rows={sum(int(row['rows']) for row in rows)} issue_files={len(issue_files)}")
    for row in issue_files:
        print(f"{row['asset']}: {row['issues']}")
    return 1 if any("schema" in str(row["issues"]) or "missing" in str(row["issues"]) for row in issue_files) else 0


if __name__ == "__main__":
    raise SystemExit(main())
