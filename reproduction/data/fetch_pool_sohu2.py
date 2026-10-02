# -*- coding: utf-8 -*-
"""Candidate-pool expansion via Sohu: A-share stocks and more ETFs.

Reuses the verified fetch/write logic from fetch_pool_sohu.py (same column mapping and
validation). Adds liquid SH/SZ names plus ETFs so that the pool has a wide |ACF1| spread,
which is what the pre-registered selection rule needs.
"""
import importlib.util
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fps", _HERE / "fetch_pool_sohu.py")
fps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fps)

ROOT = fps.ROOT
POOL2 = [
    # SH large caps
    ("MOUTAI",   "cn_600519", "equity"), ("CMB",      "cn_600036", "equity"),
    ("PINGAN",   "cn_601318", "equity"), ("CITICSEC", "cn_600030", "equity"),
    ("YILI",     "cn_600887", "equity"), ("CITS",     "cn_601888", "equity"),
    ("HENGRUI",  "cn_600276", "equity"), ("YANGTZE",  "cn_600900", "equity"),
    ("ICBC",     "cn_601398", "equity"), ("SINOPEC",  "cn_600028", "equity"),
    # SZ large caps
    ("VANKE",    "cn_000002", "equity"), ("MIDEA",    "cn_000333", "equity"),
    ("GREE",     "cn_000651", "equity"), ("BYD",      "cn_002594", "equity"),
    ("CATL",     "cn_300750", "equity"), ("WULIANG",  "cn_000858", "equity"),
    ("HIK",      "cn_002415", "equity"), ("EASTMONEY", "cn_300059", "equity"),
    ("BOE",      "cn_000725", "equity"), ("YANGHE",   "cn_002304", "equity"),
    # ETFs
    ("CSI500ETF", "cn_510500", "fund"),  ("SECETF",   "cn_512880", "fund"),
    ("TECHETF",  "cn_515000", "fund"),   ("BOND10Y",  "cn_511260", "fund"),
    ("CHINEXTETF", "cn_159915", "fund"), ("MEDETF",   "cn_512010", "fund"),
]


def main():
    ok, fail = [], []
    for tag, code, kind in POOL2:
        out = ROOT / f"{tag}-2016-2025.csv"
        if out.exists() and out.stat().st_size > 1000:
            print(f"[skip] {tag}")
            ok.append(tag)
            continue
        try:
            rows, err = fps.fetch(code)
            if not rows:
                print(f"[fail] {tag:11s} {code:9s} -> {err}")
                fail.append(tag)
            else:
                with out.open("w", encoding="utf-8", newline="") as f:
                    f.write("date,Open,High,Low,Close\n")
                    for r in rows:
                        f.write(",".join(str(x) for x in r) + "\n")
                print(f"[OK]   {tag:11s} {code:9s} {len(rows):5d} rows  {rows[0][0]} .. {rows[-1][0]}")
                ok.append(tag)
        except Exception as e:
            print(f"[fail] {tag:11s} {code:9s} -> {type(e).__name__} {str(e)[:50]}")
            fail.append(tag)
        time.sleep(1.2)
    print(f"\nfetched {len(ok)}/{len(POOL2)}; failed: {fail or 'none'}")


if __name__ == "__main__":
    main()
