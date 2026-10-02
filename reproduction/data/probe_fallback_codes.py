# -*- coding: utf-8 -*-
"""Probe fallback sources with several symbol-code conventions (US equities / HK index / FX)."""
import json
import sys
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}


def get(url, ref=""):
    h = dict(UA)
    if ref:
        h["Referer"] = ref
    with OP.open(urllib.request.Request(url, headers=h), timeout=30) as r:
        return r.read().decode("utf-8", "replace")


print("=== Tencent gtimg (US / HK / index variants) ===")
for code in ["usAAPL", "usAAPL.OQ", "hkHSI", "r_hkHSI", "jpN225", "usDJI", "usINX"]:
    u = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,"
         f"2016-01-01,2025-12-31,640,qfq")
    try:
        j = json.loads(get(u, "https://gu.qq.com/"))
        d = (j.get("data") or {}).get(code)
        if isinstance(d, dict):
            arr = d.get("qfqday") or d.get("day") or []
            print(f"  {code:10s}: rows={len(arr)}" + (f"  {arr[0][0]}..{arr[-1][0]}" if arr else ""))
        else:
            print(f"  {code:10s}: data[{code}] type={type(d).__name__} raw={str(d)[:80]}")
    except Exception as e:
        print(f"  {code:10s}: {type(e).__name__} {str(e)[:70]}")

print("=== Sohu (US variants) ===")
for code in ["us_aapl", "gb_aapl", "us_AAPL", "gb_AAPL"]:
    u = f"https://q.stock.sohu.com/hisHq?code={code}&start=20160101&end=20251231&stat=1&order=D&period=d"
    try:
        j = json.loads(get(u, "https://q.stock.sohu.com/"))
        msg = (j[0].get("msg") if j else "empty")
        hq = (j[0].get("hq") if j else None) or []
        print(f"  {code:10s}: rows={len(hq)} msg={str(msg)[:60]}")
    except Exception as e:
        print(f"  {code:10s}: {type(e).__name__} {str(e)[:70]}")

print("=== East Money (US 105.AAPL / HK 100.HSI / index) with full headers ===")
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
       "Referer": "https://quote.eastmoney.com/", "Accept": "*/*"}
for secid in ["105.AAPL", "100.HSI", "100.NDX", "1.000001"]:
    u = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=" + secid +
         "&klt=101&fqt=1&beg=20160101&end=20251231&fields1=f1,f2,f3,f4,f5&fields2=f51,f52,f53,f54,f55")
    try:
        with OP.open(urllib.request.Request(u, headers=HDR), timeout=30) as r:
            j = json.loads(r.read().decode("utf-8", "replace"))
        d = j.get("data") or {}
        kl = d.get("klines") or []
        print(f"  {secid:10s}: rows={len(kl)} name={d.get('name','?')}" + (f"  {kl[0][:10]}..{kl[-1][:10]}" if kl else ""))
    except Exception as e:
        print(f"  {secid:10s}: {type(e).__name__} {str(e)[:70]}")
