# -*- coding: utf-8 -*-
"""GSPC seed-merge sensitivity: recompute the two candidate associations with
GSPC taken as s2021 only / s2022 only / merged (mean), per protocol section 5."""
import json
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(REPO, "reproduction", "results", "asset-dependence-summary.csv")
R = os.path.join(REPO, "reproduction", "results")


def sp(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


d = pd.read_csv(P)
vals = {}
for s in (2021, 2022):
    f = f"{R}/volatility_stratification_GSPC_relative_s{s}.json"
    Rj = json.load(open(f, encoding="utf-8"))
    vals[s] = Rj["horizons"]["T24"]["comparison"]["interaction_high50_minus_low50"]["value"]

rows = []
for s, v in vals.items():
    dd = d.copy()
    dd.loc[dd.asset == "GSPC", "interaction"] = v
    rows.append((f"GSPC={s} only", v, sp(dd.realized_vol, dd.interaction), sp(dd.acf1_abs, dd.interaction)))
dd = d.copy()
rows.append(("merged (mean)", float(dd[dd.asset == "GSPC"].interaction.iloc[0]),
             sp(dd.realized_vol, dd.interaction), sp(dd.acf1_abs, dd.interaction)))
rows.append(("GSPC dropped", None, sp(dd[dd.asset != "GSPC"].realized_vol, dd[dd.asset != "GSPC"].interaction),
             sp(dd[dd.asset != "GSPC"].acf1_abs, dd[dd.asset != "GSPC"].interaction)))

print(f"{'variant':16s} {'GSPC dint':>10s} {'rho(RV)':>9s} {'rho(|ACF1|)':>12s}")
for name, v, r1, r2 in rows:
    vs = f"{v:+.5f}" if v is not None else "n/a"
    print(f"{name:16s} {vs:>10s} {r1:>+9.3f} {r2:>+12.3f}")
