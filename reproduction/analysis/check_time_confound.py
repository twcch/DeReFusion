# -*- coding: utf-8 -*-
"""检查波动率分层是否与时间混淆：rv 与样本序号的相关性 + 各分位均值日期。"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
SEQ, T = 96, 24

for tag, csv in [("GSPC", "GSPC-2016-2025.csv"), ("BTCUSD", "BTCUSD-2016-2025.csv")]:
    df = pd.read_csv(os.path.join(REPO, "dataset", csv))
    n = len(df)
    b1 = n - int(n * 0.2) - SEQ
    ns = n - b1 - SEQ - T + 1
    close = df["Close"].to_numpy(float)
    dates = pd.to_datetime(df["date"])
    rv = np.array([np.std(np.diff(np.log(close[b1 + i:b1 + i + SEQ]))) for i in range(ns)])
    idx = np.arange(ns)
    corr = float(np.corrcoef(rv, idx)[0, 1])
    print(f"[{tag}] n={ns} corr(rv, 时间序号)={corr:+.3f}")
    if tag == "GSPC":
        qs = np.percentile(rv, [0, 20, 40, 60, 80, 100])
        for k in range(5):
            m = (rv >= qs[k]) & (rv < qs[k + 1]) if k < 4 else (rv >= qs[k])
            ii = np.where(m)[0]
            # 每个样本的目标窗口结束日期
            ds = dates.to_numpy()[b1 + ii + SEQ + T - 1]
            print(f"  Q{k+1}: n={ii.size}, 目标日期中位 {np.sort(ds)[ii.size//2]}, 时间序号中位 {np.median(ii):.0f}")
