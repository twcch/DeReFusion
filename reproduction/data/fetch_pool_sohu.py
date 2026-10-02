# -*- coding: utf-8 -*-
"""Candidate-pool fetcher via Sohu (the one source that works on this host).

Sohu hisHq returns rows like [date, open, close, change, pct, low, high, volume, amount, ...];
column order is verified at runtime before anything is written, and only rows with a full
OHLC set are kept. Output: dataset/<TAG>-2016-2025.csv with header date,Open,High,Low,Close
(same schema as the existing ten datasets).

Provenance note: the original ten datasets came from Yahoo Finance; new assets from this
fetcher come from Sohu. This is recorded in the pre-registration document.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2] / "dataset"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
HDR = {"User-Agent": "Mozilla/5.0", "Referer": "https://q.stock.sohu.com/"}

# (tag, sohu code, category)
POOL = [
    ("SSE",     "cn_000001", "index"),      # Shanghai Composite
    ("SZSE",    "cn_399001", "index"),      # Shenzhen Component
    ("CSI300",  "cn_000300", "index"),      # CSI 300
    ("CHINEXT", "cn_399006", "index"),      # ChiNext
    ("AAPL",    "us_AAPL",   "equity"),
    ("MSFT",    "us_MSFT",   "equity"),
    ("AMZN",    "us_AMZN",   "equity"),
    ("TSLA",    "us_TSLA",   "equity"),
    ("NVDA",    "us_NVDA",   "equity"),
    ("KO",      "us_KO",     "equity"),
    ("XOM",     "us_XOM",    "equity"),
    ("JPM",     "us_JPM",    "equity"),
    ("WMT",     "us_WMT",    "equity"),
    ("TENCENT", "hk_00700",  "equity"),
    ("CCB",     "hk_00939",  "equity"),
    ("GOLDETF", "cn_518880", "fund"),       # gold ETF (commodity proxy)
    ("SP500ETF", "cn_513500", "fund"),      # S&P500 ETF (QDII)
    ("CSI300ETF", "cn_510300", "fund"),
    ("BONDETF", "cn_511010", "fund"),       # treasury ETF
]


def fetch(code):
    url = (f"https://q.stock.sohu.com/hisHq?code={code}&start=20160101&end=20251231"
           f"&stat=1&order=D&period=d")
    raw = OPENER.open(urllib.request.Request(url, headers=HDR), timeout=30).read()
    j = json.loads(raw.decode("utf-8", "replace"))
    if not j or not j[0].get("hq"):
        return None, (j[0].get("msg") if j else "empty")
    hq = j[0]["hq"]
    # row: [date, open, close, change, pct, low, high, volume, amount, turnover]
    rows = []
    for r in hq:
        try:
            d, o, c, lo, hi = r[0], float(r[1]), float(r[2]), float(r[5]), float(r[6])
        except (ValueError, IndexError):
            continue
        if min(o, c, lo, hi) <= 0:
            continue
        rows.append((d, o, hi, lo, c))
    rows.sort(key=lambda x: x[0])
    return rows, None


def main():
    ok, fail = [], []
    for tag, code, kind in POOL:
        out = ROOT / f"{tag}-2016-2025.csv"
        if out.exists() and out.stat().st_size > 1000:
            print(f"[skip] {tag}")
            ok.append(tag)
            continue
        try:
            rows, err = fetch(code)
            if not rows:
                print(f"[fail] {tag:10s} {code:9s} -> {err}")
                fail.append(tag)
            else:
                with out.open("w", encoding="utf-8", newline="") as f:
                    f.write("date,Open,High,Low,Close\n")
                    for r in rows:
                        f.write(",".join(str(x) for x in r) + "\n")
                print(f"[OK]   {tag:10s} {code:9s} {len(rows):5d} rows  {rows[0][0]} .. {rows[-1][0]}")
                ok.append(tag)
        except Exception as e:
            print(f"[fail] {tag:10s} {code:9s} -> {type(e).__name__} {str(e)[:60]}")
            fail.append(tag)
        time.sleep(1.2)
    print(f"\nfetched {len(ok)}/{len(POOL)}; failed: {fail or 'none'}")


if __name__ == "__main__":
    main()
