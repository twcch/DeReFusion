# -*- coding: utf-8 -*-
"""补全 DeReFusion dataset v2：应对 Yahoo 限流。
策略：curl_cffi 浏览器指纹（若可安装）→ yfinance 冷却重试兜底；单资产成功即落盘，断点续跑。
"""
import sys, io, time, json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2] / "dataset"  # repo root / dataset
ROOT.mkdir(exist_ok=True)

ASSETS = [
    ("BABA-2016-2025.csv",   "BABA",    "股票"),
    ("NVO-2016-2025.csv",    "NVO",     "股票"),
    ("TM-2016-2025.csv",     "TM",      "股票"),
    ("GSPC-2016-2025.csv",   "%5EGSPC", "指数"),
    ("DJI-2016-2025.csv",    "%5EDJI",  "指数"),
    ("SOX-2016-2025.csv",    "%5ESOX",  "指数"),
    ("EURUSD-2016-2025.csv", "EURUSD=X","外汇"),
    ("USDJPY-2016-2025.csv", "USDJPY=X","外汇"),
    ("BTCUSD-2016-2025.csv", "BTC-USD", "加密"),
    ("ETHUSD-2016-2025.csv", "ETH-USD", "加密"),
]

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1={p1}&period2={p2}&interval=1d"

def yahoo_chart_curlcffi(ticker_encoded):
    """用 curl_cffi 模拟浏览器 TLS 指纹直连 Yahoo chart API（绕限流最有效）。"""
    from curl_cffi import requests as creq
    import datetime
    p1 = int(datetime.datetime(2016, 1, 1).timestamp())
    p2 = int(datetime.datetime(2025, 12, 31).timestamp())
    url = CHART.format(t=ticker_encoded, p1=p1, p2=p2)
    r = creq.get(url, impersonate="chrome", timeout=30)
    j = json.loads(r.text)
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    rows = []
    for i, t in enumerate(ts):
        o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        if None in (o, h, l, c):
            continue
        rows.append((time.strftime("%Y-%m-%d", time.gmtime(t)), o, h, l, c))
    return rows

def yahoo_direct(ticker_encoded):
    """兜底：urllib 直连 chart API（需 UA）。"""
    import datetime, urllib.request
    p1 = int(datetime.datetime(2016, 1, 1).timestamp())
    p2 = int(datetime.datetime(2025, 12, 31).timestamp())
    url = CHART.format(t=ticker_encoded, p1=p1, p2=p2)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    j = json.loads(urllib.request.urlopen(req, timeout=30).read())
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    rows = []
    for i, t in enumerate(ts):
        o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        if None in (o, h, l, c):
            continue
        rows.append((time.strftime("%Y-%m-%d", time.gmtime(t)), o, h, l, c))
    return rows

def main():
    try:
        import curl_cffi  # noqa
        have_curlcffi = True
        print("[i] curl_cffi 可用（浏览器指纹模式）")
    except ImportError:
        print("[i] curl_cffi 不可用，使用 urllib 直连")
    ok, fail = [], []
    for fname, tick_enc, kind in ASSETS:
        out = ROOT / fname
        if out.exists() and out.stat().st_size > 1000:
            print(f"[skip] {fname} 已存在"); ok.append(fname); continue
        got = False
        for attempt in range(4):
            try:
                if "curl_cffi" in sys.modules:
                    rows = yahoo_chart_curlcffi(tick_enc)
                else:
                    rows = yahoo_direct(tick_enc)
                with out.open("w", encoding="utf-8", newline="") as f:
                    f.write("date,Open,High,Low,Close\n")
                    for r in rows:
                        f.write(",".join(str(x) for x in r) + "\n")
                print(f"[OK]   {fname:22s} {kind}  {len(rows)} 行  {rows[0][0]} → {rows[-1][0]}")
                ok.append(fname); got = True; break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"[retry{attempt+1}] {fname}: {str(e)[:90]}  等{wait}s")
                time.sleep(wait)
        if not got:
            fail.append(fname)
        time.sleep(2)  # 资产间隔，降低限流概率
    print(f"\n完成 {len(ok)}/{len(ASSETS)}，失败: {fail or '无'}")

if __name__ == "__main__":
    main()
