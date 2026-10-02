# -*- coding: utf-8 -*-
"""补全 DeReFusion dataset v3：urllib 直连 Yahoo chart API（本机已验证可达，UA 必带）。"""
import sys, json, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "dataset"  # repo root / dataset
ROOT.mkdir(exist_ok=True)

ASSETS = [
    ("BABA-2016-2025.csv",   "BABA",     "股票"),
    ("NVO-2016-2025.csv",    "NVO",      "股票"),
    ("TM-2016-2025.csv",     "TM",       "股票"),
    ("GSPC-2016-2025.csv",   "%5EGSPC",  "指数"),
    ("DJI-2016-2025.csv",    "%5EDJI",   "指数"),
    ("SOX-2016-2025.csv",    "%5ESOX",   "指数"),
    ("EURUSD-2016-2025.csv", "EURUSD=X", "外汇"),
    ("USDJPY-2016-2025.csv", "USDJPY=X", "外汇"),
    ("BTCUSD-2016-2025.csv", "BTC-USD",  "加密"),
    ("ETHUSD-2016-2025.csv", "ETH-USD",  "加密"),
]

def fetch(tick):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{tick}?period1=1451606400&period2=1767139200&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    res = json.loads(urllib.request.urlopen(req, timeout=30).read())["chart"]["result"][0]
    ts = res["timestamp"]; q = res["indicators"]["quote"][0]
    return [(time.strftime("%Y-%m-%d", time.gmtime(t)), q["open"][i], q["high"][i], q["low"][i], q["close"][i])
            for i, t in enumerate(ts)
            if None not in (q["open"][i], q["high"][i], q["low"][i], q["close"][i])]

def main():
    ok, fail = [], []
    for fname, tick, kind in ASSETS:
        out = ROOT / fname
        if out.exists() and out.stat().st_size > 1000:
            print("[skip]", fname); ok.append(fname); continue
        for attempt in range(3):
            try:
                rows = fetch(tick)
                with out.open("w", newline="") as f:
                    f.write("date,Open,High,Low,Close\n")
                    for r in rows:
                        f.write(",".join(str(x) for x in r) + "\n")
                print(f"[OK] {fname:22s} {kind} {len(rows)}行 {rows[0][0]}->{rows[-1][0]}")
                ok.append(fname); break
            except Exception as e:
                print(f"[retry{attempt+1}] {fname}: {type(e).__name__} {str(e)[:80]}")
                time.sleep(5 * (attempt + 1))
        else:
            fail.append(fname)
        time.sleep(2)
    print(f"DONE {len(ok)}/10, fail: {fail or '无'}")

if __name__ == "__main__":
    main()
