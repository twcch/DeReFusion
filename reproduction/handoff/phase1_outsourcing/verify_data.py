"""Verify every registered Phase 1 dataset by raw-byte SHA-256."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "reproduction/results/dataset_registry.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    failures = []
    with REGISTRY.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = ROOT / row["file"]
        observed = sha256(path) if path.is_file() else "MISSING"
        if observed != row["sha256"]:
            failures.append((row["asset"], str(path), row["sha256"], observed))
    if failures:
        for item in failures:
            print("FAIL", *item)
        return 2
    print(f"PASS {len(rows)} registered datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
