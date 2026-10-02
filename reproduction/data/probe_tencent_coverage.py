# -*- coding: utf-8 -*-
"""Coverage test: Tencent gtimg full-history (2016-2025) for the non-A-share list."""
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
      "Referer": "https://gu.qq.com/"}

# (output name, candidate tencent codes)
CAND = [
    ("AAPL", ["usAAPL.OQ", "usAAPL"]),
    ("MSFT", ["usMSFT.OQ", "usMSFT"]),
    ("AMZN", ["usAMZN.OQ", "usAMZN"]),
    ("META", ["usMETA.OQ", "usMETA"]),
    ("TSLA", ["usTSLA.OQ", "usTSLA"]),
    ("JPM",  ["usJPM.N", "usJPM"]),
    ("XOM",  ["usXOM.N", "usXOM"]),
    ("WMT",  ["usWMT.N", "usWMT"]),
    ("HSI",  ["hkHSI"]),
    ("N225", ["jpN225", "hkN225", "usN225"]),
    ("GDAXI", ["deGDAXI", "hkGDAXI"]),
    ("GOLD", ["usGC.CMX", "hkgold", "usGOLD"]),
]

for name, codes in CAND:
    found = False
    for code in codes:
        u = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,"
             f"2016-01-01,2025-12-31,3000,qfq")
        try:
            with OP.open(urllib.request.Request(u, headers=UA), timeout=30) as r:
                j = json.loads(r.read().decode("utf-8", "replace"))
            d = (j.get("data") or {}).get(code)
            if isinstance(d, dict):
                arr = d.get("qfqday") or d.get("day") or []
                if arr:
                    print(f"  {name:6s} <- {code:12s} rows={len(arr):5d}  {arr[0][0]} .. {arr[-1][0]}")
                    found = True
                    break
        except Exception as e:
            print(f"  {name:6s} <- {code:12s} {type(e).__name__} {str(e)[:50]}")
    if not found:
        print(f"  {name:6s} <- NOT FOUND in candidates {codes}")
