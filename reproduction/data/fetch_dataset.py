# -*- coding: utf-8 -*-
"""补全 DeReFusion dataset：按论文 4.1 节规格从 Yahoo Finance 拉取 10 个资产日线 OHLC。
输出格式与论文一致：date,Open,High,Low,Close（date=YYYY-MM-DD，无 Volume 列）。
区间：2016-01-01 → 2025-12-31。
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "dataset"  # repo root / dataset
ROOT.mkdir(exist_ok=True)

# (文件名, yfinance ticker, 类型) —— 论文 Table 2 资产清单
ASSETS = [
    ("BABA-2016-2025.csv",   "BABA",    "股票"),
    ("NVO-2016-2025.csv",    "NVO",     "股票"),
    ("TM-2016-2025.csv",     "TM",      "股票"),
    ("GSPC-2016-2025.csv",   "^GSPC",   "指数"),
    ("DJI-2016-2025.csv",    "^DJI",    "指数"),
    ("SOX-2016-2025.csv",    "^SOX",    "指数"),
    ("EURUSD-2016-2025.csv", "EURUSD=X","外汇"),
    ("USDJPY-2016-2025.csv", "USDJPY=X","外汇"),
    ("BTCUSD-2016-2025.csv", "BTC-USD", "加密"),
    ("ETHUSD-2016-2025.csv", "ETH-USD", "加密"),
]

def fetch_yf(ticker, start="2016-01-01", end="2025-12-31"):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("empty dataframe")
    # 多级列（yf 新版）压平
    if hasattr(df.columns, "get_level_values") and df.columns.nlevels > 1:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    # 统一列名
    ren = {}
    for c in df.columns:
        cl = str(c).lower()
        if c.lower() == "date": ren[c] = "date"
        elif c.lower() == "open": ren[c] = "Open"
        elif c.lower() == "high": ren[c] = "High"
        elif c.lower() == "low": ren[c] = "Low"
        elif c.lower() == "close": ren[c] = "Close"
    df = df.rename(columns=ren)[["date", "Open", "High", "Low", "Close"]]
    df["date"] = df["date"].astype(str).str[:10]
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    return df

def main():
    # 延迟导入 + 代理探测：yfinance 内部请求也需要网络；直连被墙时尝试常见端口
    import os
    proxies = None
    for p in ("7897", "7890", "10809"):
        import socket
        s = socket.socket(); s.settimeout(1)
        if s.connect_ex(("127.0.0.1", int(p))) == 0:
            proxies = {"http": f"http://127.0.0.1:{p}", "https": f"http://127.0.0.1:7897"}
            s.close(); break
        s.close()
    if proxies:
        os.environ["HTTP_PROXY"] = proxies["http"]
        os.environ["HTTPS_PROXY"] = proxies["https"]
        print(f"[i] 使用本地代理 {proxies['https']}")
    else:
        print("[i] 无本地代理，直连尝试")

    import yfinance as yf
    print("[i] yfinance ready")

if __name__ == "__main__":
    ok, fail = [], []
    for fname, ticker, kind in ASSETS:
        out = ROOT / fname
        if out.exists() and out.stat().st_size > 1000:
            print(f"[skip] {fname} 已存在")
            ok.append(fname); continue
        for attempt in range(3):
            try:
                df = fetch_yf(ticker)
                df.to_csv(out, index=False)
                print(f"[OK]   {fname:22s} {kind}  {len(df)} 行  {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
                ok.append(fname); break
            except Exception as e:
                print(f"[retry{attempt+1}] {fname}: {e}")
                time.sleep(3 * (attempt + 1))
        else:
            fail.append(fname)
    print(f"\n完成 {len(ok)}/{len(ASSETS)}，失败: {fail or '无'}")
