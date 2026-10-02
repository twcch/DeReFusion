# -*- coding: utf-8 -*-
"""Decide whether Yahoo can be reached from this host at all, and check fallback sources
that serve the SAME asset classes (US equities / indices / FX / commodities)."""
import json
import ssl
import sys
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))          # bypass broken proxy
CJAR = CookieJar()
OPENER_C = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(CJAR))

BROWSER = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
    "Origin": "https://finance.yahoo.com",
}


def probe(label, url, headers=None, timeout=30):
    h = dict(BROWSER)
    if headers:
        h.update(headers)
    try:
        with OPENER.open(urllib.request.Request(url, headers=h), timeout=timeout) as r:
            b = r.read()
        print(f"  {label}: HTTP {r.status} | {r.headers.get('Content-Type','')} | {len(b)} bytes")
        return r.status, b
    except urllib.error.HTTPError as e:
        print(f"  {label}: HTTP {e.code} | {e.headers.get('Content-Type','')} | blocked/panda={b'sad-panda' in e.read() or b'<html' in e.read()[:200]}")
        return e.code, b""
    except Exception as e:
        print(f"  {label}: {type(e).__name__} {str(e)[:90]}")
        return None, b""


print("=== A) Yahoo with full browser headers (query1 / query2) ===")
P = "range=max&interval=1d&events=div,splits&includeAdjustedClose=true"
probe("query1 AAPL", f"https://query1.finance.yahoo.com/v8/finance/chart/AAPL?{P}")
probe("query2 AAPL", f"https://query2.finance.yahoo.com/v8/finance/chart/AAPL?{P}")

print("=== B) Yahoo cookie handshake then chart call ===")
try:
    with OPENER_C.open(urllib.request.Request("https://fc.yahoo.com/", headers=BROWSER), timeout=30) as r:
        print(f"  fc.yahoo.com: HTTP {r.status} | cookies={len(CJAR)}")
except Exception as e:
    print(f"  fc.yahoo.com: {type(e).__name__} {str(e)[:80]}")
try:
    with OPENER_C.open(urllib.request.Request(f"https://query1.finance.yahoo.com/v8/finance/chart/AAPL?{P}", headers=BROWSER), timeout=30) as r:
        b = r.read()
        print(f"  chart with cookies: HTTP {r.status} | {len(b)} bytes | looks_json={b[:1] == b'{'}")
except Exception as e:
    print(f"  chart with cookies: {type(e).__name__} {str(e)[:80]}")

print("=== C) fallback sources for the same assets ===")
# Tencent: US stocks / HK / indices
for label, code in [("tencent US-AAPL", "usAAPL.OQ"), ("tencent HK-HSI", "hkHSI"), ("tencent N225", "jpN225")]:
    u = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,"
         f"2016-01-01,2025-12-31,3000,qfq")
    try:
        with OPENER.open(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0",
                                                            "Referer": "https://gu.qq.com/"}), timeout=30) as r:
            j = json.loads(r.read().decode("utf-8", "replace"))
        d = (j.get("data") or {}).get(code) or {}
        arr = d.get("qfqday") or d.get("day") or []
        print(f"  {label}: rows={len(arr)}" + (f"  {arr[0][0]}..{arr[-1][0]}" if arr else ""))
    except Exception as e:
        print(f"  {label}: {type(e).__name__} {str(e)[:80]}")
# Sohu: US stocks (us_AAPL failed earlier -> confirm), indices
for label, code in [("sohu us_AAPL", "us_AAPL")]:
    u = f"https://q.stock.sohu.com/hisHq?code={code}&start=20160101&end=20251231&stat=1&order=D&period=d"
    try:
        with OPENER.open(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0",
                                                            "Referer": "https://q.stock.sohu.com/"}), timeout=30) as r:
            j = json.loads(r.read().decode("utf-8", "replace"))
        print(f"  {label}: {str(j)[:120]}")
    except Exception as e:
        print(f"  {label}: {type(e).__name__} {str(e)[:80]}")
