"""Bounded CPU-only consumer of newly completed, immutable fit packages.

Reads finished receipts only. Never loads checkpoints, uses a GPU, trains,
edits production artifacts, or interprets screen outcomes. Each intake pass
and acknowledgement has an exclusive directory and survives consumer restart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from reproduction.analysis.intake_phase1_completed_packages import intake, write_json


def prepare(source: Path, manifest: Path, output: Path, label: str) -> dict[str, str]:
    source, manifest, output = source.resolve(), manifest.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("consumer output must be separate from production source")
    if output.parent != ROOT / "reproduction/results/phase1":
        raise ValueError("consumer root must be an isolated Phase 1 child")
    identity = {"consumer_version": "phase1-cpu-intake-consumer/v1", "source": str(source),
                "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "source_label": label}
    if output.exists():
        marker = output / "consumer.json"
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != identity:
            raise FileExistsError("consumer root belongs to a different queue")
    else:
        output.mkdir()
        write_json(output / "consumer.json", identity)
    consumed = {}
    for marker in sorted(output.glob("pass-*/acknowledgement.json")):
        for attempt, digest in json.loads(marker.read_text(encoding="utf-8"))["receipt_hashes"].items():
            if attempt in consumed and consumed[attempt] != digest:
                raise RuntimeError("consumer history has conflicting receipt identities")
            consumed[attempt] = digest
    return consumed


def candidates(source: Path, consumed: dict[str, str]) -> dict[str, str]:
    ready = {}
    for marker in sorted(source.glob("*__attempt-*/receipt.json")):
        raw = marker.read_bytes()
        receipt = json.loads(raw)
        if receipt.get("status") != "completed_unreviewed":
            continue
        name, digest = marker.parent.name, hashlib.sha256(raw).hexdigest()
        if name in consumed:
            if consumed[name] != digest:
                raise RuntimeError("previously consumed completed receipt changed: " + name)
        else:
            ready[name] = digest
    return ready


def consume_pass(source: Path, manifest: Path, output: Path, label: str,
                 consumed: dict[str, str]) -> dict | None:
    ready = candidates(source, consumed)
    if not ready:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex
    pass_root = output / ("pass-" + stamp)
    pass_root.mkdir()
    report = intake(source, manifest, pass_root / "intake", label, set(ready))
    # A failed pass is recorded, not endlessly reprocessed. Nothing is declared
    # eligible except packages verified by intake; human review remains explicit.
    acknowledgement = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                       "receipt_hashes": ready, "intake_status": report["status"],
                       "summary": report.get("summary"), "error": report.get("error"),
                       "issues": report.get("issues"), "training_performed": False,
                       "gpu_used": False, "report": str(pass_root / "intake/report.json")}
    write_json(pass_root / "acknowledgement.json", acknowledgement)
    consumed.update(ready)
    return acknowledgement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--max-hours", type=float, default=8)
    args = parser.parse_args()
    if args.poll_seconds < 10 or not 0 < args.max_hours <= 8:
        parser.error("poll >= 10 seconds and bounded max-hours in (0,8] required")
    consumed = prepare(args.source_root, args.manifest, args.output_root, args.source_label)
    deadline = time.monotonic() + args.max_hours * 3600
    print(f"CPU intake consumer ready: {args.source_label}; restored {len(consumed)} acknowledgements", flush=True)
    try:
        while time.monotonic() < deadline:
            result = consume_pass(args.source_root, args.manifest, args.output_root, args.source_label, consumed)
            if result:
                print(json.dumps(result), flush=True)
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        print("consumer stopped; all intake reports and original artifacts preserved", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
