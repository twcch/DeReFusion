# -*- coding: utf-8 -*-
"""Per-asset capacity results for the four pre-registered structural-validation assets,
compared against the pre-registered expectations."""
import pandas as pd

CSV = "reproduction/results/operator-regime-capacity_sv.csv"

PREREG = {  # asset: (|ACF1|, expected interaction direction, expected preference)
    "BYD":       (0.0324, "more negative", "nonlinear-favoured"),
    "BOE":       (0.0408, "more negative", "nonlinear-favoured"),
    "EASTMONEY": (0.1296, "more positive", "linear-favoured"),
    "YANGHE":    (0.1307, "more positive", "linear-favoured"),
}

d = pd.read_csv(CSV)
rows = []
for (w, a), g in d.groupby(["width", "asset"]):
    rows.append({
        "width": int(w), "asset": a, "n_obs": len(g),
        "mean_dMSE": round(g.dMSE.mean(), 5),
        "prefer_nonlin": f"{int((g.dMSE < 0).sum())}/{len(g)}",
        "sig_prefer_nonlin": int(((g.dMSE < 0) & (g.sig == "sig")).sum()),
        "mean_win_N": round(g.winN.mean(), 3),
    })
t = pd.DataFrame(rows).sort_values(["asset", "width"])
print("=== per asset x width ===")
print(t.to_string(index=False))

print("\n=== observed preference vs pre-registration ===")
for a in ["BYD", "BOE", "EASTMONEY", "YANGHE"]:
    sub = t[t.asset == a]
    m64 = sub[sub.width == 64].mean_dMSE.iloc[0]
    m128 = sub[sub.width == 128].mean_dMSE.iloc[0]
    obs = ("nonlinear-favoured" if max(m64, m128) < 0 else
           "linear-favoured" if min(m64, m128) > 0 else "mixed/tie")
    acf, dirn, exp = PREREG[a]
    print(f"  {a:10s} |ACF1|={acf:.4f}  mean dMSE: w64={m64:+.5f} w128={m128:+.5f}  "
          f"observed={obs:19s} expected={exp:19s} -> {'MATCH' if obs == exp else 'MISMATCH'}")

print("\n=== per-seed consistency (sign of mean dMSE) ===")
for a in ["BYD", "BOE", "EASTMONEY", "YANGHE"]:
    for w in (64, 128):
        sub = d[(d.asset == a) & (d.width == w)]
        signs = [("+" if v > 0 else "-") for v in sub.groupby("seed").dMSE.mean()]
        print(f"  {a:10s} w={w:3d}: per-seed mean-dMSE signs = {''.join(signs)}")
