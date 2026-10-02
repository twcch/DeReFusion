# -*- coding: utf-8 -*-
"""Probe East Money (eastmoney.com) kline API as the new-asset data source.

The host's system proxy (127.0.0.1:7897) is broken for TLS, so every request here bypasses
it explicitly (ProxyHandler({})). Prints row counts and date ranges for a few asset classes.
"""
import json
import urllib.request

BASE = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
        "?secid={secid}&klt=101&fqt=1&beg=20160101&end=20251231"
        "&fields1=f1,f2,f3,f4,f5&fields2=f51,f52,f53,f54,f55")

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

CANDIDATES = [
    ("SSE-index", "1.000001"),
    ("SZSE-index", "0.399001"),
    ("CSI300", "1.000300"),
    ("US-NASDAQ", "100.NDX"),
    ("US-SPX", "100.SPX"),
    ("US-AAPL", "105.AAPL"),
    ("HK-HSI", "100.HSI"),
    ("FX-EURUSD", "133.EURUSD"),
    ("Crypto-BTC", "120.BTC"),
    ("Futures-gold", "101.GC00Y"),
]


def probe(name, secid):
    url = BASE.format(secid=secid)
    try:
        r = OPENER.open(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=25)
        j = json.loads(r.read().decode("utf-8", "replace"))
        data = j.get("data")
        if not data or not data.get("klines"):
            print(f"  {name:14s} {secid:12s} -> no data ({str(j)[:70]})")
            return
        kl = data["klines"]
        print(f"  {name:14s} {secid:12s} -> {len(kl):5d} rows  {kl[0][:10]} .. {kl[-1][:10]}  "
              f"name={data.get('name','?')}  sample={kl[0][:30]}")
    except Exception as e:
        print(f"  {name:14s} {secid:12s} -> FAIL {type(e).__name__} {str(e)[:70]}")


print("East Money kline probe (proxy bypassed):")
for n, s in CANDIDATES:
    probe(n, s)
