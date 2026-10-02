# -*- coding: utf-8 -*-
"""Protocol section 9: is rho(|ACF1|, interaction) = +0.733
   (A) a genuine cross-asset relation, (B) leverage by a few assets, or (C) a capacity artefact?

Uses ONLY existing data (no new assets): N=10 asset-level interaction effects + the
capacity CSVs for the three assets that have them. No definition is changed.
"""
import json
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(REPO, "reproduction", "results", "asset-dependence-summary.csv")
CAP = os.path.join(REPO, "reproduction", "results", "operator-regime-capacity.csv")


def sp(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


d = pd.read_csv(P)
print("=== (1) full sample ===")
print(f"  N={len(d)}  rho(|ACF1|, dint)={sp(d.acf1_abs, d.interaction):+.3f}   "
      f"rho(RV, dint)={sp(d.realized_vol, d.interaction):+.3f}")

print("\n=== (2) leave-one-asset-out (top |ACF1| assets first) ===")
for a in ["NVO", "SOX", "BTCUSD", "GSPC", "ETHUSD"]:
    s = d[d.asset != a]
    print(f"  drop {a:7s} (|ACF1|={d[d.asset==a].acf1_abs.iloc[0]:.3f}): "
          f"rho(|ACF1|)={sp(s.acf1_abs, s.interaction):+.3f}  rho(RV)={sp(s.realized_vol, s.interaction):+.3f}")

print("\n=== (3) exclude GSPC (the only 2-seed asset) ===")
s = d[d.asset != "GSPC"]
print(f"  rho(|ACF1|)={sp(s.acf1_abs, s.interaction):+.3f}  rho(RV)={sp(s.realized_vol, s.interaction):+.3f}")

print("\n=== (4) leverage ranking: drop each single asset, show range ===")
rows = []
for a in d.asset:
    s = d[d.asset != a]
    rows.append((a, sp(s.acf1_abs, s.interaction), sp(s.realized_vol, s.interaction)))
lo = pd.DataFrame(rows, columns=["dropped", "rho_acf1", "rho_rv"])
print(f"  rho(|ACF1|) across drops: min={lo.rho_acf1.min():+.3f} max={lo.rho_acf1.max():+.3f} "
      f"flips_sign={bool((lo.rho_acf1.min() * lo.rho_acf1.max()) < 0)}")
print(f"  rho(RV)    across drops: min={lo.rho_rv.min():+.3f} max={lo.rho_rv.max():+.3f} "
      f"flips_sign={bool((lo.rho_rv.min() * lo.rho_rv.max()) < 0)}")

print("\n=== (5) capacity view (only the 3 assets with capacity data) ===")
c = pd.read_csv(CAP)
g = c.groupby(["width", "asset"])["dMSE"].mean().unstack()
for w in (64, 128):
    if w in g.index:
        pref = g.loc[w]
        joined = d.set_index("asset").join(pref.rename("dMSE_w"))
        sub = joined[joined.dMSE_w.notna()]
        print(f"  width={w}: assets={list(sub.index)}  dMSE={[round(v,4) for v in sub.dMSE_w]}  "
              f"rho(|ACF1|, dMSE_w)={sp(sub.acf1_abs, sub.dMSE_w):+.3f}  (n={len(sub)})")

print("\n=== (6) direction-of-preference check against the association's prediction ===")
pred = {"ETHUSD": "low |ACF1| -> nonlinear-favoured expected",
        "BTCUSD": "high |ACF1| -> linear-favoured expected",
        "GSPC": "mid |ACF1| -> near parity expected"}
for a, txt in pred.items():
    row = d[d.asset == a].iloc[0]
    w128 = g.loc[128, a]
    print(f"  {a:7s} |ACF1|={row.acf1_abs:.3f}  dint={row.interaction:+.5f}  "
          f"dMSE(128)={w128:+.4f}  -> {txt}")
