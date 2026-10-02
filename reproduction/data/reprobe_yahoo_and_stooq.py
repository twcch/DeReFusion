# -*- coding: utf-8 -*-
"""Re-probe after the operator's proxy change: Yahoo (AAPL, inherit + bypass) and Stooq
(a promising no-auth CSV source that was never tested with the proxy bypassed)."""
import io
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
P = "range=max&interval=1d&events=div,splits&includeAdjustedClose=true"
YAHOO = f"https://query1.finance.yahoo.com/v8/finance/chart/AAPL?{P}"
STOOQ = "https://stooq.com/q/d/l/?s={s}&i=d"


def hit(label, url, bypass, timeout=25):
    op = urllib.request.build_opener(urllib.request.ProxyHandler({} if bypass else None))
    mode = "bypass" if bypass else "inherit-proxy"
    try:
        with op.open(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            b = r.read()
        head = b[:90].decode("utf-8", "replace").replace("\n", " | ")
        print(f"  [{mode}] {label}: HTTP {r.status} | {len(b)}B | {head}")
        return r.status, b
    except urllib.error.HTTPError as e:
        body = e.read(200).decode("utf-8", "replace")
        print(f"  [{mode}] {label}: HTTPError {e.code} | panda={'sad-panda' in body}")
        return e.code, b""
    except Exception as e:
        print(f"  [{mode}] {label}: {type(e).__name__}: {str(e)[:100]}")
        return None, b""


print("=== Yahoo AAPL (did the proxy change help?) ===")
hit("yahoo-AAPL", YAHOO, bypass=False)
hit("yahoo-AAPL", YAHOO, bypass=True)

print("=== Stooq (no auth, direct CSV) ===")
for s in ["aapl.us", "msft.us", "^spx", "^ndq", "^dji", "eurusd", "gbpusd", "gc.f", "cl.f"]:
    st, b = hit(f"stooq {s}", STOOQ.format(s=s), bypass=True, timeout=20)
    if st == 200 and b:
        txt = b.decode("utf-8", "replace")
        lines = txt.strip().splitlines()
        print(f"        -> rows={len(lines) - 1} header={lines[0][:40]!r} first={lines[1][:40]!r}" if len(lines) > 1 else f"        -> {txt[:80]!r}")
