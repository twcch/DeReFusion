"""Read-only intake audit for packaged Phase 1 B/C/D attempts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction.batches.run_phase1_stages import (
    CONFIG, MANIFESTS, OUTPUT, canonical_hash, completed_attempt, expected_truth,
    logical_id, rows, split_for, validate_preflight,
)
from utils.metrics import metric

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=list(MANIFESTS), required=True)
    parser.add_argument("--attempt-root", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    manifest = rows(MANIFESTS[args.stage])
    _, _, preflight_issues = validate_preflight(manifest)
    issues = list(preflight_issues)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    registry = {r["asset"]: r for r in rows(ROOT / "reproduction/results/dataset_registry.csv")}
    inventory = []
    for row in manifest:
        logical = logical_id(row)
        attempts = sorted(args.attempt_root.glob(logical + "__attempt-*"))
        if len(attempts) != 1:
            issues.append(f"{logical}: expected exactly one attempt, found {len(attempts)}")
            continue
        path = attempts[0]
        try:
            receipt = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
            split = split_for(row)
            device = receipt.get("execution_device")
            if device not in {"cpu", "cuda"}:
                raise RuntimeError("receipt lacks a recognized execution_device")
            expected_config = canonical_hash({"row": row, "common": config["common"], "device": device})
            completed_attempt(path, row, split, expected_config)
            registry_row = registry[row["asset"]]
            if receipt.get("dataset_sha256") != registry_row["sha256"]:
                raise RuntimeError("receipt dataset hash differs from frozen registry")
            pred, true, metrics = (np.load(path / name, allow_pickle=False) for name in ("pred.npy", "true.npy", "metrics.npy"))
            expected = expected_truth(row, split, config["common"], ROOT / registry_row["file"])
            if pred.shape != expected.shape or true.shape != expected.shape or metrics.shape != (6,):
                raise RuntimeError(f"shape mismatch: pred={pred.shape}, true={true.shape}, metrics={metrics.shape}, expected={expected.shape}")
            if not all(np.isfinite(a).all() for a in (pred, true, metrics)):
                raise RuntimeError("non-finite artifact value")
            if not np.allclose(true, expected, rtol=2e-5, atol=2e-6):
                raise RuntimeError("true.npy is not aligned with frozen prediction keys and train-only scaling")
            recomputed_metrics = np.asarray(metric(pred, true), dtype=np.float64)
            if not np.allclose(metrics, recomputed_metrics, rtol=1e-5, atol=1e-8, equal_nan=True):
                raise RuntimeError("metrics.npy does not reproduce from packaged predictions and targets")
            inventory.append({"logical_run_id": logical, "attempt_id": path.name, "status": "accepted_for_stage_review"})
        except Exception as error:
            issues.append(f"{logical}: {error}")
    result = {"stage": args.stage, "expected_runs": len(manifest), "accepted_for_stage_review": len(inventory),
              "issues": issues, "inventory": inventory}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        if args.report.exists():
            raise RuntimeError(f"refusing to overwrite report: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not issues and len(inventory) == len(manifest) else 2


if __name__ == "__main__":
    raise SystemExit(main())
