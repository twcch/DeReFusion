# -*- coding: utf-8 -*-
"""Isolate the cause: proxy environment variables vs Yahoo's edge block.

A) report where the proxy env vars exist (current process)
B) child process with ALL proxy env vars removed -> AAPL chart endpoint
C) same child, explicit ProxyHandler({}) (bypass by code)
D) same child, explicit proxy to 127.0.0.1:7897
Each attempt records HTTP status or the full exception type/message.
"""
import json
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PY = sys.executable
PROXY_KEYS = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
              "http_proxy", "https_proxy", "all_proxy", "no_proxy"]
URL = ("https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
       "?range=max&interval=1d&events=div,splits&includeAdjustedClose=true")

print("=== A) proxy env vars in THIS process ===")
present = {k: os.environ[k] for k in PROXY_KEYS if k in os.environ}
print("  present:", json.dumps(present, ensure_ascii=False) if present else "none")

CHILD = r'''
import sys, urllib.request
url = sys.argv[1]
mode = sys.argv[2]
if mode == "bypass":
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
elif mode == "explicit":
    op = urllib.request.build_opener(urllib.request.ProxyHandler({"http":"http://127.0.0.1:7897","https":"http://127.0.0.1:7897"}))
else:
    op = urllib.request.build_opener()
h = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
try:
    with op.open(urllib.request.Request(url, headers=h), timeout=25) as r:
        b = r.read(400)
    print(f"  HTTP {r.status} | {r.headers.get('Content-Type','')[:30]} | {len(b)}B")
except urllib.error.HTTPError as e:
    body = e.read(400).decode("utf-8","replace")
    print(f"  HTTPError {e.code} | panda={'sad-panda' in body} | body[:80]={body[:80]!r}")
except Exception as e:
    print(f"  {type(e).__name__}: {str(e)[:130]}")
'''

env_clean = {k: v for k, v in os.environ.items() if k not in PROXY_KEYS}
env_inherit = dict(os.environ)

print("=== B) child WITHOUT proxy env vars (mode=default opener) ===")
subprocess.run([PY, "-c", CHILD, URL, "default"], env=env_clean)

print("=== C) child WITHOUT proxy env vars (mode=explicit bypass) ===")
subprocess.run([PY, "-c", CHILD, URL, "bypass"], env=env_clean)

print("=== D) child WITHOUT proxy env vars (mode=explicit proxy 7897) ===")
subprocess.run([PY, "-c", CHILD, URL, "explicit"], env=env_clean)

print("=== E) child WITH inherited proxy env vars (control) ===")
subprocess.run([PY, "-c", CHILD, URL, "default"], env=env_inherit)
