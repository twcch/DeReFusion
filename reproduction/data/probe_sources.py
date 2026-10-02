# -*- coding: utf-8 -*-
"""Multi-source probe for new-asset daily OHLC on this host.

All requests bypass the (broken) system proxy explicitly. Each source is tested with the
headers its own web front-end sends. Goal: find ONE working China-accessible source so that
the structural-validation candidate pool can be fetched.
"""
import json
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(url, headers, timeout=20):
    return OPENER.open(urllib.request.Request(url, headers=headers), timeout=timeout).read()


def t_eastmoney_hdr():
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.000001&klt=101&fqt=1"
           "&beg=20160101&end=20251231&fields1=f1,f2,f3,f4,f5&fields2=f51,f52,f53,f54,f55")
    h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
         "Referer": "https://quote.eastmoney.com/", "Accept": "*/*"}
    j = json.loads(get(url, h).decode("utf-8", "replace"))
    kl = (j.get("data") or {}).get("klines") or []
    print(f"  eastmoney+hdr : {len(kl)} rows" + (f"  {kl[0][:10]}..{kl[-1][:10]}" if kl else ""))


def t_tencent():
    url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000001,day,"
           "2016-01-01,2025-12-31,3000,qfq")
    h = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}
    j = json.loads(get(url, h).decode("utf-8", "replace"))
    d = j.get("data", {}).get("sh000001", {})
    kl = d.get("qfqday") or d.get("day") or []
    print(f"  tencent       : {len(kl)} rows" + (f"  {kl[0][0]}..{kl[-1][0]}" if kl else ""))


def t_sina():
    url = "https://hq.sinajs.cn/list=sh000001"
    h = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}
    txt = get(url, h).decode("gbk", "replace")
    print(f"  sina quote    : {txt[:60]}")


def t_sohu():
    url = "https://q.stock.sohu.com/hisHq?code=cn_000001&start=20160101&end=20251231&stat=1&order=D&period=d"
    h = {"User-Agent": "Mozilla/5.0", "Referer": "https://q.stock.sohu.com/"}
    j = json.loads(get(url, h).decode("utf-8", "replace"))
    rows = (j[0].get("hq") if j else []) or []
    print(f"  sohu          : {len(rows)} rows" + (f"  {rows[0][0]}..{rows[-1][0]}" if rows else ""))


print("Multi-source probe (proxy bypassed):")
for name, fn in [("eastmoney+hdr", t_eastmoney_hdr), ("tencent", t_tencent),
                 ("sina", t_sina), ("sohu", t_sohu)]:
    try:
        fn()
    except Exception as e:
        print(f"  {name:14s}: FAIL {type(e).__name__} {str(e)[:90]}")
