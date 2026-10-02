# -*- coding: utf-8 -*-
"""C1 prospective predictor.

Computes the C1 predictor(s) for the locked cohort using ONLY data strictly before the decision
point, per reports/evidence_closure/23_c1_preregistration.md §4 and §8.

Decision point (cutoff) = border1_test, the start index of the first test window, using the SAME
convention as the frozen pipeline (structure_routing_experiment.load_asset):
    num_train     = int(n * 0.7)
    border1_test  = n - int(n * 0.2) - SEQ_LEN

A feature window i covers close[i : i+SEQ_LEN]. It is usable only if the whole window lies before
the decision point:  i + SEQ_LEN <= border1_test  <=>  i <= border1_test - SEQ_LEN.
So the predictor aggregates the window features over i in [0, border1_test - SEQ_LEN] (the
training+validation windows). Nothing at or after the cutoff is used for the primary predictor.

Outputs (repo):
  reproduction/results/c1_predictor.csv        machine-readable
  reproduction/results/c1_predictor.md         human-readable table + leakage check
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import structure_routing_experiment as S  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAGS = ["AAPL", "MSFT", "AMZN", "META", "TSLA", "JPM", "XOM", "WMT",
        "N225", "GDAXI", "HSI", "FTSE", "RUT",
        "GBPUSD", "AUDUSD", "USDCAD", "GOLD", "WTI", "GLD", "TLT"]

rows = []
for tag in TAGS:
    df, raw, close, num_train, border1_test = S.load_asset(tag)
    n = len(df)
    # last window that lies entirely before the decision point
    last_ok = border1_test - S.SEQ_LEN
    idx = list(range(0, last_ok + 1))
    feats = [S.window_features(close, i) for i in idx]
    f = pd.DataFrame(feats)

    max_index_used = (idx[-1] + S.SEQ_LEN - 1) if idx else -1     # last close index touched
    leak_ok = max_index_used < border1_test

    # contemporaneous analogue (test-window features) - reported for contrast only, never primary
    te_idx = list(range(border1_test, n - S.SEQ_LEN + 1))
    te = pd.DataFrame([S.window_features(close, i) for i in te_idx]) if te_idx else pd.DataFrame()

    rows.append({
        "tag": tag,
        "n_rows": n,
        "num_train": num_train,
        "cutoff_index": border1_test,
        "cutoff_date": str(df["date"].iloc[border1_test]) if border1_test < n else "",
        "windows_used": len(idx),
        "last_close_index_used": max_index_used,
        "leak_check_ok": bool(leak_ok),
        # PRIMARY, prospective: aggregate over pre-cutoff windows only
        "acf1_abs_pre": float(np.nanmedian(np.abs(f["acf1"].to_numpy()))),
        "rv_pre": float(np.nanmedian(f["rv"].to_numpy())),
        # CONTRAST ONLY: the contemporaneous analogue (test-window medians)
        "acf1_abs_test": float(np.nanmedian(np.abs(te["acf1"].to_numpy()))) if len(te) else float("nan"),
        "rv_test": float(np.nanmedian(te["rv"].to_numpy())) if len(te) else float("nan"),
    })

out = pd.DataFrame(rows).sort_values("acf1_abs_pre")
csv_path = os.path.join(REPO, "reproduction", "results", "c1_predictor.csv")
out.to_csv(csv_path, index=False)

all_ok = bool(out["leak_check_ok"].all())
md = []
md.append("# C1 predictor — pre-cutoff (prospective) values, LOCKED cohort\n")
md.append(f"Leakage check: **{'ALL PASS' if all_ok else 'FAIL'}** "
          f"(every predictor window ends strictly before `border1_test`).\n")
md.append("`acf1_abs_pre` / `rv_pre` = median over the pre-cutoff windows — the PRIMARY predictor.\n"
          "`acf1_abs_test` / `rv_test` = the contemporaneous analogue over the test windows — "
          "reported for contrast only, never used as the primary.\n")
cols = ["tag", "n_rows", "cutoff_index", "cutoff_date", "windows_used",
        "last_close_index_used", "leak_check_ok", "acf1_abs_pre", "rv_pre", "acf1_abs_test", "rv_test"]

def _row(vals):
    return "| " + " | ".join(str(v) for v in vals) + " |"

_d = out[cols].round(5)
md.append(_row(cols))
md.append("|" + "---|" * len(cols))
for _, _r in _d.iterrows():
    md.append(_row([_r[c] for c in cols]))
md_path = os.path.join(REPO, "reproduction", "results", "c1_predictor.md")
with open(md_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(md) + "\n")

print(out[["tag", "cutoff_index", "windows_used", "last_close_index_used", "leak_check_ok",
           "acf1_abs_pre", "rv_pre", "acf1_abs_test"]].round(5).to_string(index=False))
print(f"\nleakage check: {'ALL PASS' if all_ok else 'FAIL'}")
print(f"saved: {csv_path}")
print(f"saved: {md_path}")
sys.exit(0 if all_ok else 1)
