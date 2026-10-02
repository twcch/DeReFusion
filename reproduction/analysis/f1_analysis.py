# -*- coding: utf-8 -*-
"""F1 analysis, strictly per the locked rules in reports/evidence_closure/20_f1_preregistration.md.

Rules:
  - per (asset, width): mean dMSE over states and seeds; negative = nonlinear-favoured
  - seed-stable at a width: per-seed mean dMSE has the same sign in ALL THREE seeds
  - width-stable: seed-stable at BOTH width 64 and width 128 with the SAME sign
  - stable asset = width-stable
  - success: >= 2 additional stable assets AND at least one NEW opposite-sign pair among them
"""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

F1 = "reproduction/results/operator-regime-capacity_f1.csv"
EXISTING = "reproduction/results/operator-regime-capacity.csv"

d = pd.read_csv(F1)
print(f"=== raw: {len(d)} rows | assets={sorted(d.asset.unique())} | widths={sorted(d.width.unique())} | seeds={sorted(d.seed.unique())} ===")

# per-seed means
per_seed = d.groupby(["asset", "width", "seed"]).dMSE.mean().reset_index()
rows = []
for (a, w), g in per_seed.groupby(["asset", "width"]):
    signs = set(g.dMSE.apply(lambda v: "neg" if v < 0 else "pos"))
    rows.append({
        "asset": a, "width": int(w),
        "mean_dMSE": round(g.dMSE.mean(), 5),
        "per_seed": " ".join(f"{v:+.4f}" for v in g.sort_values("seed").dMSE),
        "seed_stable": len(signs) == 1,
        "class": "nonlinear" if g.dMSE.mean() < 0 else "linear",
    })
t = pd.DataFrame(rows).sort_values(["asset", "width"])
print("\n=== per asset x width ===")
print(t.to_string(index=False))

# width stability at 64 and 128, same sign
print("\n=== width-stability check (64 vs 128, same sign + seed-stable at both) ===")
stable = []
for a, g in t[t.width.isin([64, 128])].groupby("asset"):
    g = g.sort_values("width")
    if len(g) != 2:
        print(f"  {a}: missing a width")
        continue
    s64, s128 = g.iloc[0], g.iloc[1]
    same_sign = (s64.mean_dMSE < 0) == (s128.mean_dMSE < 0)
    ok = same_sign and bool(s64.seed_stable) and bool(s128.seed_stable)
    print(f"  {a:8s} w64={s64.mean_dMSE:+.5f} (seed-stable={bool(s64.seed_stable)})  "
          f"w128={s128.mean_dMSE:+.5f} (seed-stable={bool(s128.seed_stable)})  "
          f"same_sign={same_sign}  -> STABLE={ok}  class={g.iloc[1]['class']}")
    if ok:
        stable.append((a, g.iloc[1]["class"]))

print(f"\n=== stable assets: {stable if stable else 'NONE'} ===")
nl = [a for a, c in stable if c == "nonlinear"]
lin = [a for a, c in stable if c == "linear"]
print(f"  nonlinear-favoured stable: {nl}")
print(f"  linear-favoured   stable: {lin}")
n_pairs = len(nl) * len(lin)
print(f"  new opposite-sign pairs among F1 assets: {n_pairs}")

print("\n=== locked success criterion ===")
print(f"  >=2 additional stable assets : {len(stable) >= 2} (have {len(stable)})")
print(f"  >=1 NEW opposite-sign pair   : {n_pairs >= 1}")
print(f"  ==> F1 {'SUCCESS' if (len(stable) >= 2 and n_pairs >= 1) else 'FAIL'}")

# context: the three existing assets, same computation, for reference only
try:
    e = pd.read_csv(EXISTING)
    pe = e.groupby(["asset", "width", "seed"]).dMSE.mean().reset_index()
    print("\n=== reference (already-existing three assets) ===")
    for (a, w), g in pe.groupby(["asset", "width"]):
        print(f"  {a:8s} w{int(w):3d}  mean={g.dMSE.mean():+.5f}  per-seed={' '.join(f'{v:+.4f}' for v in g.sort_values('seed').dMSE)}")
except Exception as ex:
    print("  (existing-file reference unavailable:", ex, ")")
