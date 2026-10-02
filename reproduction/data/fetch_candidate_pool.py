# -*- coding: utf-8 -*-
"""Structural-validation candidate pool fetcher.

Purpose: fetch a diverse candidate pool (2016-2025 daily OHLC) so that new assets can be
pre-registered by their structural features (|ACF1| extremes) BEFORE any operator experiment
is run. Same data source and the same fetch pattern as fetch_dataset_v2.py.

Writes to dataset/<TAG>-2016-2025.csv (skips any file already present and non-trivial).
"""
import io
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
ROOT.mkdir(exist_ok=True)

# (tag, yahoo ticker encoded, category)
POOL = [
    # indices
    ("N225",   "%5EN225",   "index"),
    ("GDAXI",  "%5EGDAXI",  "index"),
    ("HSI",    "%5EHSI",    "index"),
    ("RUT",    "%5ERUT",    "index"),
    ("FTSE",   "%5EFTSE",   "index"),
    # equities
    ("AAPL",   "AAPL",      "equity"),
    ("MSFT",   "MSFT",      "equity"),
    ("AMZN",   "AMZN",      "equity"),
    ("META",   "META",      "equity"),
    ("TSLA",   "TSLA",      "equity"),
    ("JPM",    "JPM",       "equity"),
    ("XOM",    "XOM",       "equity"),
    ("KO",     "KO",        "equity"),
    ("WMT",    "WMT",       "equity"),
    # fx
    ("GBPUSD", "GBPUSD=X",  "fx"),
    ("AUDUSD", "AUDUSD=X",  "fx"),
    ("USDCAD", "USDCAD=X",  "fx"),
    ("USDCNY", "USDCNY=X",  "fx"),
    # crypto
    ("LTCUSD", "LTC-USD",   "crypto"),
    ("XRPUSD", "XRP-USD",   "crypto"),
    ("BNBUSD", "BNB-USD",   "crypto"),
    # commodities / funds
    ("GOLD",   "GC=F",      "commodity"),
    ("WTI",    "CL=F",      "commodity"),
    ("GLD",    "GLD",       "fund"),
    ("TLT",    "TLT",       "fund"),
]

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1={p1}&period2={p2}&interval=1d"


def _period():
    import datetime
    p1 = int(datetime.datetime(2016, 1, 1).timestamp())
    p2 = int(datetime.datetime(2025, 12, 31).timestamp())
    return p1, p2


def fetch(ticker_encoded):
    p1, p2 = _period()
    url = CHART.format(t=ticker_encoded, p1=p1, p2=p2)
    try:
        from curl_cffi import requests as creq
        text = creq.get(url, impersonate="chrome", timeout=30).text
    except Exception:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        text = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    j = json.loads(text)
    res = j["chart"]["result"][0]
    ts, q = res["timestamp"], res["indicators"]["quote"][0]
    rows = []
    for i, t in enumerate(ts):
        o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        if None in (o, h, l, c):
            continue
        rows.append((time.strftime("%Y-%m-%d", time.gmtime(t)), o, h, l, c))
    return rows


def main():
    ok, fail = [], []
    for tag, tick, kind in POOL:
        out = ROOT / f"{tag}-2016-2025.csv"
        if out.exists() and out.stat().st_size > 1000:
            print(f"[skip] {tag:8s} exists")
            ok.append(tag)
            continue
        got = False
        for attempt in range(4):
            try:
                rows = fetch(tick)
                with out.open("w", encoding="utf-8", newline="") as f:
                    f.write("date,Open,High,Low,Close\n")
                    for r in rows:
                        f.write(",".join(str(x) for x in r) + "\n")
                print(f"[OK]   {tag:8s} {kind:9s} {len(rows):5d} rows  {rows[0][0]} -> {rows[-1][0]}")
                ok.append(tag)
                got = True
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"[retry{attempt+1}] {tag}: {str(e)[:80]} wait {wait}s")
                time.sleep(wait)
        if not got:
            fail.append(tag)
        time.sleep(2)
    print(f"\nfetched {len(ok)}/{len(POOL)}; failed: {fail or 'none'}")


if __name__ == "__main__":
    main()
