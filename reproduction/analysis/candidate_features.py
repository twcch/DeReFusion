# -*- coding: utf-8 -*-
"""Pre-registration input: structural features of the candidate pool.

Uses the SAME definition as the existing analysis (imports asset_features from
final_diagnosis.py): test-window features with SEQ_LEN=96, border1 = n - int(n*0.2) - 96,
medians over windows of |ACF1|, realized volatility, jump ratio (c=3 sigma), trend
persistence (|t|), sign persistence, skew, kurtosis.

Selection rule (fixed BEFORE looking at any operator result):
  from the NEW candidates with full coverage (>= 2300 rows), take the 2 with the LOWEST
  |ACF1| and the 2 with the HIGHEST |ACF1|.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("fd", REPO / "reproduction" / "analysis" / "final_diagnosis.py")
fd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fd)

EXISTING = {"GSPC", "BTCUSD", "ETHUSD", "USDJPY", "EURUSD", "SOX", "DJI", "BABA", "NVO", "TM"}

rows = []
for p in sorted((REPO / "dataset").glob("*-2016-2025.csv")):
    tag = p.name.split("-")[0]
    n = sum(1 for _ in p.open(encoding="utf-8")) - 1
    try:
        f = fd.asset_features(str(p))
    except Exception as e:
        print(f"  {tag}: feature computation failed ({type(e).__name__})")
        continue
    rows.append({"asset": tag, "n_rows": n, "is_new": tag not in EXISTING, **f})

df = pd.DataFrame(rows)
df.to_csv(REPO / "reproduction" / "results" / "candidate_pool_features.csv", index=False)

new = df[(df.is_new) & (df.n_rows >= 2300)].sort_values("acf1_abs")
print("=== NEW candidates with full coverage (>=2300 rows), sorted by |ACF1| ===")
print(new[["asset", "n_rows", "acf1_abs", "realized_vol", "jump_ratio",
           "trend_persistence", "sign_persistence", "skew", "kurtosis"]].round(4).to_string(index=False))
print("\n=== existing ten (reference positions) ===")
print(df[~df.is_new][["asset", "n_rows", "acf1_abs", "realized_vol"]].sort_values("acf1_abs").round(4).to_string(index=False))
print("\n=== short-history new candidates (excluded from selection) ===")
print(df[(df.is_new) & (df.n_rows < 2300)][["asset", "n_rows", "acf1_abs"]].round(4).to_string(index=False))
