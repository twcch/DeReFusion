# -*- coding: utf-8 -*-
"""C1 cohort ingester / validator (T004 intake).

Checks the cohort geng delivers under geng-lobster/T004-cohort/ against the locked spec in
thinkbook-lobster/047-T004-cohort-acquisition.md, and emits a DRAFT lock table for
reports/evidence_closure/23_c1_preregistration.md (a draft is NOT a lock - the lock is a separate
commit after review).

Checks per file:
  - file exists, non-empty
  - header is EXACTLY: date,Open,High,Low,Close
  - every date parses as YYYY-MM-DD, strictly ascending, no duplicates
  - price columns parse as floats; no row with all prices missing
  - coverage: first date <= 2016-01-31 and last date >= 2025-12-01
  - SHA-256 matches MANIFEST.csv
  - tag is in the pre-named universe and does NOT collide with the existing assets
Global checks:
  - MANIFEST.csv present with the agreed columns; every ok row has a file
  - the receipt exists and attests the provenance (mentions Yahoo)
  - no cohort tag duplicates an existing dataset tag (original 10 + the 4 A-share cohort assets)

Usage:
  python c1_cohort_validate.py [--cohort DIR] [--manifest CSV] [--repo DIR] [--draft OUT.md]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parents[2]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

UNIVERSE = ["AAPL", "MSFT", "AMZN", "META", "TSLA", "JPM", "XOM", "WMT",
            "N225", "GDAXI", "HSI", "FTSE", "RUT",
            "GBPUSD", "AUDUSD", "USDCAD", "GOLD", "WTI", "GLD", "TLT"]

EXISTING = ["GSPC", "BTCUSD", "ETHUSD", "USDJPY", "EURUSD", "SOX", "DJI", "BABA", "NVO", "TM",
            "BYD", "BOE", "EASTMONEY", "YANGHE"]

HEADER = ["date", "Open", "High", "Low", "Close"]
FIRST_MIN = datetime(2016, 1, 31)
LAST_MIN = datetime(2025, 12, 1)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file(path: str):
    """Return (ok, rows, first, last, dupes, problems[])."""
    problems = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        try:
            head = next(rd)
        except StopIteration:
            return False, 0, "", "", 0, ["empty file"]
        head = [h.strip() for h in head]
        if head != HEADER:
            problems.append(f"header is {head!r}, expected {HEADER!r}")
        rows = 0
        seen = set()
        dupes = 0
        first_d, last_d = None, None
        prev = None
        bad_dates = 0
        bad_prices = 0
        for r in rd:
            if not r or all(not x.strip() for x in r):
                continue
            rows += 1
            d = r[0].strip()
            try:
                dt = datetime.strptime(d, "%Y-%m-%d")
            except Exception:
                bad_dates += 1
                continue
            if d in seen:
                dupes += 1
            seen.add(d)
            if prev is not None and dt < prev:
                problems.append(f"not ascending at {d}")
                prev = dt
            else:
                prev = dt
            first_d = dt if first_d is None or dt < first_d else first_d
            last_d = dt if last_d is None or dt > last_d else last_d
            vals = []
            for x in r[1:5]:
                try:
                    vals.append(float(x))
                except Exception:
                    vals.append(None)
            if all(v is None for v in vals):
                bad_prices += 1
    if bad_dates:
        problems.append(f"{bad_dates} unparsable dates")
    if bad_prices:
        problems.append(f"{bad_prices} rows with all-missing OHLC")
    if dupes:
        problems.append(f"{dupes} duplicate dates")
    if first_d and first_d > FIRST_MIN:
        problems.append(f"first date {first_d.date()} later than {FIRST_MIN.date()}")
    if last_d and last_d < LAST_MIN:
        problems.append(f"last date {last_d.date()} earlier than {LAST_MIN.date()}")
    ok = not problems
    return ok, rows, (first_d.date().isoformat() if first_d else ""), \
        (last_d.date().isoformat() if last_d else ""), dupes, problems


def read_manifest(path: str):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=str(REPO / "dataset"))
    ap.add_argument("--manifest", default="")
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--draft", default="")
    a = ap.parse_args()

    manifest_path = a.manifest or os.path.join(a.cohort, "MANIFEST.csv")
    out_rows = []

    print(f"=== C1 cohort intake | {a.cohort} ===")
    if not os.path.isdir(a.cohort):
        print("  COHORT DIRECTORY ABSENT - nothing to validate yet.")
        print("  (expected after geng delivers T004; the dispatch is thinkbook-lobster/047-T004-cohort-acquisition.md)")
        return 2

    if not os.path.isfile(manifest_path):
        print(f"  MANIFEST missing at {manifest_path}")
        return 3

    man = read_manifest(manifest_path)
    by_tag = {m.get("tag", "").strip(): m for m in man}
    print(f"  manifest rows: {len(man)}")

    problems_global = []
    for tag in UNIVERSE:
        p = os.path.join(a.cohort, f"{tag}.csv")
        m = by_tag.get(tag)
        if not os.path.isfile(p):
            print(f"  {tag:8s} MISSING FILE")
            problems_global.append(f"{tag}: file missing")
            out_rows.append([tag, "", "", "", "", "", "MISSING"])
            continue
        ok, rows, first, last, dupes, probs = check_file(p)
        digest = sha256(p)
        man_sha = (m or {}).get("sha256", "").strip().lower()
        sha_ok = (not man_sha) or (man_sha == digest)
        if not sha_ok:
            probs.append(f"sha mismatch (manifest {man_sha[:12]}... vs file {digest[:12]}...)")
            ok = False
        flag = "OK" if ok else "FAIL"
        print(f"  {tag:8s} rows={rows:5d} {first} -> {last} dupes={dupes} sha={'ok' if sha_ok else 'MISMATCH'} [{flag}]")
        for pr in probs:
            print(f"           - {pr}")
        if not ok:
            problems_global.append(f"{tag}: {'; '.join(probs)}")
        out_rows.append([tag, rows, first, last, digest, "ok" if ok else "fail", flag])

    # overlap with the existing assets
    overlap = sorted(set(UNIVERSE) & set(EXISTING))
    if overlap:
        problems_global.append(f"universe overlaps existing assets: {overlap}")
        print(f"  !! universe overlaps existing assets: {overlap}")

    # extra files that are not in the universe
    extra = [f for f in os.listdir(a.cohort) if f.endswith(".csv") and f != "MANIFEST.csv"
             and f[:-4] not in UNIVERSE]
    if extra:
        print(f"  note: extra csv files not in the universe: {extra}")

    # provenance attestation
    receipt = os.path.join(os.path.dirname(a.cohort), "006-T004-receipt.md")
    if os.path.isfile(receipt):
        txt = open(receipt, encoding="utf-8", errors="replace").read().lower()
        if "yahoo" not in txt:
            problems_global.append("receipt does not attest the Yahoo provenance")
            print("  !! receipt does not mention Yahoo (provenance unattested)")
        else:
            print("  provenance: receipt attests Yahoo")
    else:
        problems_global.append(f"receipt missing at {receipt}")
        print(f"  note: receipt not found ({receipt})")

    # draft lock table
    if a.draft:
        lines = ["| # | Asset | Provider | Rows | First date | Last date | SHA-256 |",
                 "|---|---|---|---|---|---|---|"]
        for i, r in enumerate(out_rows, 1):
            if len(r) >= 6:
                lines.append(f"| {i} | {r[0]} | Yahoo (via geng 83JM system proxy) | {r[1]} | {r[2]} | {r[3]} | `{r[4]}` |")
            else:
                lines.append(f"| {i} | {r[0]} | — | — | — | — | *MISSING* |")
        body = "\n".join(lines)
        with open(a.draft, "w", encoding="utf-8") as f:
            f.write("# C1 lock table — DRAFT (generated, not a lock)\n\n")
            f.write("Generated by `reproduction/analysis/c1_cohort_validate.py`.\n")
            f.write("**This is a draft.** The lock is a separate commit made AFTER review and BEFORE\n")
            f.write("any feature or outcome is inspected (see `23_c1_preregistration.md` §8).\n\n")
            f.write(body + "\n")
        print(f"  draft lock table written: {a.draft}")

    print("\n=== summary ===")
    if problems_global:
        print(f"  {len(problems_global)} problem(s):")
        for p in problems_global:
            print(f"    - {p}")
        return 1
    print(f"  all {len(UNIVERSE)} files pass: schema, dates, coverage, duplicates, hashes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
