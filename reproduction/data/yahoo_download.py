# -*- coding: utf-8 -*-
"""Yahoo Finance v8 chart-API downloader (per operator spec, 2026-09-13).

- Python requests if available, else urllib (both bypass the broken Windows system proxy:
  every HTTPS request through 127.0.0.1:7897 dies with an SSL handshake timeout).
- query1 -> query2 fallback, timeout 30 s, UA header, events=div,splits, includeAdjustedClose=true.
- Prints ticker / URL / status / content-type / first 500 chars on failure (never just "failed").
- Output schema: Date,Open,High,Low,Close,Adj_Close,Volume  -> data/raw/yahoo/<NAME>.csv
- Report: data/raw/yahoo/download_report.csv

Usage:
    python yahoo_download.py --only AAPL        # spec step 13: test one symbol first
    python yahoo_download.py                    # all symbols
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO, "data", "raw", "yahoo")
UA = "Mozilla/5.0"
TIMEOUT = 30

# (yahoo symbol, output name)
SYMBOLS = [
    ("AAPL", "AAPL"), ("MSFT", "MSFT"), ("AMZN", "AMZN"), ("META", "META"),
    ("TSLA", "TSLA"), ("JPM", "JPM"), ("XOM", "XOM"), ("WMT", "WMT"),
    ("^N225", "N225"), ("^GDAXI", "GDAXI"), ("^HSI", "HSI"), ("^FTSE", "FTSE"),
    ("^RUT", "RUT"),
    ("GBPUSD=X", "GBPUSD"), ("AUDUSD=X", "AUDUSD"), ("USDCAD=X", "USDCAD"),
    ("GC=F", "GOLD"), ("CL=F", "WTI"), ("GLD", "GLD"), ("TLT", "TLT"),
]

PARAMS = {"range": "max", "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"}


def build_url(host, symbol):
    return f"https://{host}/v8/finance/chart/{urllib.parse.quote(symbol)}?" + urllib.parse.urlencode(PARAMS)


def _http_get(url):
    """Return (http_status, content_type, text). Uses requests when available."""
    try:
        import requests  # noqa
        s = requests.Session()
        s.trust_env = False                      # ignore the broken system proxy
        s.proxies = {}
        r = s.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
        return r.status_code, r.headers.get("Content-Type", ""), r.text
    except ImportError:
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # bypass proxy
        try:
            with op.open(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=TIMEOUT) as resp:
                return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read().decode("utf-8", "replace")
    except Exception:
        raise


def fetch(symbol):
    """Try query1 then query2. Returns (rows, meta) with full diagnostics."""
    meta = {"symbol": symbol, "url": "", "http_status": "", "content_type": "", "error": ""}
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = build_url(host, symbol)
        meta["url"] = url
        try:
            status, ctype, text = _http_get(url)
        except Exception as e:
            meta["http_status"] = "EXC"
            meta["content_type"] = ""
            meta["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"  [{symbol}] {url}\n    EXC {meta['error']}")
            time.sleep(1.0)
            continue
        print(f"  [{symbol}] {url}\n    HTTP {status} | {ctype} | {len(text)} bytes")
        if status != 200:
            print(f"    body[:500]: {text[:500]}")
            meta["http_status"] = status
            meta["content_type"] = ctype
            meta["error"] = text[:200]
            time.sleep(1.0)
            continue
        try:
            j = json.loads(text)
        except Exception as e:
            meta["http_status"] = status
            meta["content_type"] = ctype
            meta["error"] = f"json parse: {e}: {text[:200]}"
            print(f"    json parse failed: {text[:300]}")
            continue
        chart = j.get("chart") or {}
        err = chart.get("error")
        res = chart.get("result")
        meta["http_status"] = status
        meta["content_type"] = ctype
        if err:
            meta["error"] = json.dumps(err)[:200]
            print(f"    chart.error = {err}")
            continue
        if not res:
            meta["error"] = "result empty"
            print("    result empty")
            continue
        r0 = res[0]
        ts = r0.get("timestamp") or []
        q = ((r0.get("indicators") or {}).get("quote") or [{}])[0]
        adj = (((r0.get("indicators") or {}).get("adjclose") or [{}])[0]).get("adjclose") or []
        rows = []
        for i, t in enumerate(ts):
            try:
                o, h, l, c = q.get("open", [])[i], q.get("high", [])[i], q.get("low", [])[i], q.get("close", [])[i]
                v = (q.get("volume") or [None] * (i + 1))[i]
            except IndexError:
                continue
            a = adj[i] if i < len(adj) else None
            # spec 10: drop rows with no timestamp or all-empty OHLC
            if t is None or all(x is None for x in (o, h, l, c)):
                continue
            rows.append((time.strftime("%Y-%m-%d", time.gmtime(t)), o, h, l, c, a, v))
        if not rows:
            meta["error"] = "no usable rows"
            print("    no usable rows")
            continue
        rows.sort(key=lambda r: r[0])              # spec 11: ascending
        meta["error"] = ""
        return rows, meta
    return [], meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    todo = [s for s in SYMBOLS if not args.only or s[1] == args.only or s[0] == args.only]

    os.makedirs(OUTDIR, exist_ok=True)
    report = []
    print(f"=== yahoo v8 chart download | {len(todo)} symbol(s) | out={OUTDIR} ===")
    for sym, name in todo:
        rows, meta = fetch(sym)
        out = os.path.join(OUTDIR, f"{name}.csv")
        if rows:
            with open(out, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume"])
                for r in rows:
                    w.writerow(r)
            print(f"  -> saved {out} ({len(rows)} rows: {rows[0][0]} .. {rows[-1][0]})")
            report.append([sym, name, "ok", len(rows), rows[0][0], rows[-1][0], meta["http_status"], ""])
        else:
            print(f"  -> FAILED {sym}: {meta['error'][:160]}")
            report.append([sym, name, "fail", 0, "", "", meta["http_status"], meta["error"][:300]])
        time.sleep(1.2)

    rp = os.path.join(OUTDIR, "download_report.csv")
    with open(rp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "output_name", "status", "rows", "start_date", "end_date", "http_status", "error"])
        w.writerows(report)
    ok = sum(1 for r in report if r[2] == "ok")
    print(f"=== done: {ok} ok / {len(report) - ok} failed | report={rp} ===")
    for r in report:
        print(f"  {r[0]:9s} -> {r[1]:7s} {r[2]:4s} rows={r[3]:>5} {r[4]} .. {r[5]} {r[7][:80]}")


if __name__ == "__main__":
    main()
