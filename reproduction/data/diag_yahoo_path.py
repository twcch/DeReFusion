# -*- coding: utf-8 -*-
"""Yahoo reachability matrix, per operator instruction:
   (A) inherit the system proxy   (B) explicit bypass   (C) bypass + IPv4 forced
   (D) control target through the proxy (is the proxy able to tunnel TLS at all?)
   Plus TLS/socket-level detail recorded per attempt (exception type + message)."""
import json
import socket
import ssl
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = "range=max&interval=1d&events=div,splits&includeAdjustedClose=true"
TARGETS = {
    "yahoo-q1-AAPL": f"https://query1.finance.yahoo.com/v8/finance/chart/AAPL?{P}",
    "yahoo-q2-AAPL": f"https://query2.finance.yahoo.com/v8/finance/chart/AAPL?{P}",
    "control-tuna": "https://pypi.tuna.tsinghua.edu.cn/simple/",
    "control-cloudflare": "https://www.cloudflare.com/cdn-cgi/trace",
}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}


def attempt(label, url, proxy):
    if proxy is None:
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        mode = "bypass"
    elif proxy == "inherit":
        op = urllib.request.build_opener()
        mode = "inherit-system"
    else:
        op = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        mode = f"explicit:{proxy}"
    try:
        with op.open(urllib.request.Request(url, headers=UA), timeout=25) as r:
            b = r.read(2048)
        print(f"  [{mode}] {label}: HTTP {r.status} | {r.headers.get('Content-Type','')[:40]} | {len(b)}B")
        return r.status
    except urllib.error.HTTPError as e:
        body = e.read(300)
        print(f"  [{mode}] {label}: HTTPError {e.code} | panda={'sad-panda' in body.decode('utf-8','replace')}")
        return e.code
    except Exception as e:
        print(f"  [{mode}] {label}: {type(e).__name__}: {str(e)[:120]}")
        return None


print("=== A/B: Yahoo AAPL, inherit system proxy vs explicit bypass ===")
for name, url in [("yahoo-q1-AAPL", TARGETS["yahoo-q1-AAPL"]), ("yahoo-q2-AAPL", TARGETS["yahoo-q2-AAPL"])]:
    attempt(name, url, "inherit")
    attempt(name, url, None)

print("=== D: can the proxy tunnel TLS at all? (controls) ===")
for name in ("control-tuna", "control-cloudflare"):
    attempt(name, TARGETS[name], "inherit")
    attempt(name, TARGETS[name], None)

print("=== C: force IPv4 (DNS returns IPv6 first) ===")
for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
    try:
        infos = socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)   # IPv4 only
        ips = sorted({i[4][0] for i in infos})
        print(f"  {host} A-records: {ips}")
    except Exception as e:
        print(f"  {host}: getaddrinfo(AF_INET) failed: {e}")
        ips = []
    for ip in ips[:2]:
        # raw TLS handshake with correct SNI to bypass any DNS interference
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((ip, 443), timeout=15) as s:
                with ctx.wrap_socket(s, server_hostname=host) as ss:
                    print(f"  TLS to {ip} (SNI={host}): OK  version={ss.version()}  cipher={ss.cipher()[0]}")
        except Exception as e:
            print(f"  TLS to {ip} (SNI={host}): {type(e).__name__}: {str(e)[:100]}")

print("=== DNS comparison (hijack check) ===")
import subprocess
for resolver in (None, "223.5.5.5", "119.29.29.29"):
    cmd = ["nslookup", "query1.finance.yahoo.com"] + ([resolver] if resolver else [])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
        ips = sorted({ln.split()[-1] for ln in out.splitlines() if "Address" in ln and ":" not in ln.split()[-1][0]})
        print(f"  resolver={resolver or 'default'}: {ips or out.strip()[:120]}")
    except Exception as e:
        print(f"  resolver={resolver or 'default'}: {type(e).__name__} {str(e)[:60]}")
