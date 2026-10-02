# -*- coding: utf-8 -*-
"""Gate A finalisation for the structural-validation protocol.

Builds the N=14 asset-level table (ten existing assets + the four pre-registered new ones),
recomputes the exploratory Spearman associations with leave-one-asset-out, evaluates the five
required views, compares every new asset against the direction locked in the pre-registration
(reports/evidence_closure/11_structural_validation_preregistration.md), and prints the Gate A
verdict plus the protocol's output block.

Nothing here changes any existing definition, threshold, seed or metric, and it never writes
over the N=10 artifacts (its outputs carry the `_sv` suffix).
"""
import importlib.util
import json
import os
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INT = os.path.join(REPO, "reproduction", "results")

NEW = ["BYD", "BOE", "EASTMONEY", "YANGHE"]
PREREG = {  # asset: (|ACF1|, expected interaction direction, expected operator preference)
    "BYD": (0.0324, "negative", "nonlinear"),
    "BOE": (0.0408, "negative", "nonlinear"),
    "EASTMONEY": (0.1296, "positive", "linear"),
    "YANGHE": (0.1307, "positive", "linear"),
}


def sp(x, y):
    """Spearman rho and exploratory p-value (scipy when available)."""
    x, y = pd.Series(list(x), dtype=float), pd.Series(list(y), dtype=float)
    ok = (~x.isna()) & (~y.isna())
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), float("nan"), len(x)
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return float(r), float(p), len(x)
    except Exception:
        r = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
        return r, float("nan"), len(x)


def strat_interaction(tag, seed=2021):
    p = os.path.join(REPO, "reproduction", "results",
                     f"volatility_stratification_{tag}_relative_s{seed}.json")
    if not os.path.exists(p):
        return None
    R = json.load(open(p, encoding="utf-8"))
    c = R["horizons"]["T24"]["comparison"]
    it = c.get("interaction_high50_minus_low50") or {}
    return {"interaction": it.get("value"),
            "ci_lo": (it.get("ci95") or [None])[0], "ci_hi": (it.get("ci95") or [None, None])[1],
            "significant": it.get("significant")}


def main():
    old = pd.read_csv(os.path.join(INT, "asset-dependence-summary.csv"))
    feats = pd.read_csv(os.path.join(REPO, "reproduction", "results", "candidate_pool_features.csv"))
    cap = pd.read_csv(os.path.join(REPO, "reproduction", "results", "operator-regime-capacity_sv.csv"))
    cap0 = pd.read_csv(os.path.join(REPO, "reproduction", "results", "operator-regime-capacity.csv"))

    rows = []
    for _, r in old.iterrows():
        rows.append({"asset": r.asset, "set": "existing", "interaction": r.interaction,
                     "acf1_abs": r.acf1_abs, "realized_vol": r.realized_vol,
                     "significant": r.significant})
    missing = []
    for a in NEW:
        s = strat_interaction(a)
        f = feats[feats.asset == a]
        if s is None or not len(f):
            missing.append(a)
            continue
        rows.append({"asset": a, "set": "new", "interaction": s["interaction"],
                     "acf1_abs": float(f.acf1_abs.iloc[0]), "realized_vol": float(f.realized_vol.iloc[0]),
                     "significant": s["significant"]})
    if missing:
        print(f"[wait] new-asset stratifications not ready yet: {missing}")
        print("       (the framework batch writes volatility_stratification_<ASSET>_relative_s2021.json)")
        if len(rows) <= 10:
            return
    d = pd.DataFrame(rows)
    print(f"=== N = {len(d)} asset-level table ===")
    print(d.round(5).to_string(index=False))

    r_acf, p_acf, n = sp(d.acf1_abs, d.interaction)
    r_rv, p_rv, _ = sp(d.realized_vol, d.interaction)
    print(f"\n=== associations (N={n}) ===")
    print(f"  rho(|ACF1|, d_interaction) = {r_acf:+.3f} (p={p_acf:.3f})   [N=10 was +0.733, p=0.016]")
    print(f"  rho(RV,    d_interaction) = {r_rv:+.3f} (p={p_rv:.3f})   [N=10 was +0.661, p=0.038]")

    print("\n=== LOO (drop each asset) ===")
    loo = []
    for a in d.asset:
        s = d[d.asset != a]
        r1, _, _ = sp(s.acf1_abs, s.interaction)
        r2, _, _ = sp(s.realized_vol, s.interaction)
        loo.append((a, r1, r2))
    L = pd.DataFrame(loo, columns=["dropped", "rho_acf1", "rho_rv"])
    flip = bool((L.rho_acf1.min() * L.rho_acf1.max()) < 0)
    print(f"  rho(|ACF1|) range [{L.rho_acf1.min():+.3f}, {L.rho_acf1.max():+.3f}]  sign_flip={flip}")
    print(f"  rho(RV)     range [{L.rho_rv.min():+.3f}, {L.rho_rv.max():+.3f}]  "
          f"sign_flip={bool((L.rho_rv.min() * L.rho_rv.max()) < 0)}")
    worst = L.reindex(L.rho_acf1.abs().sort_values().index).head(3)
    print(f"  most influential drops for |ACF1|: {', '.join(worst.dropped)}")

    print("\n=== view: exclude GSPC ===")
    s = d[d.asset != "GSPC"]
    r1, p1, n1 = sp(s.acf1_abs, s.interaction)
    print(f"  rho(|ACF1|)={r1:+.3f} (p={p1:.3f}, N={n1})")

    print("\n=== view: capacity consistency (sign of mean dMSE per asset, widths 64 / 128) ===")
    cap_all = pd.concat([cap0, cap], ignore_index=True)
    contradictions, pref = [], {}
    for a in d.asset:
        sub = cap_all[cap_all.asset == a]
        if not len(sub):
            continue
        m = sub.groupby("width").dMSE.mean()
        if 64 in m.index and 128 in m.index:
            pref[a] = (m[64], m[128])
            if (m[64] > 0) != (m[128] > 0):
                contradictions.append(a)
            print(f"  {a:10s} w64={m[64]:+.5f}  w128={m[128]:+.5f}")
    print(f"  capacity contradictions (sign differs 64 vs 128): {contradictions or 'none'}")

    print("\n=== pre-registration comparison (new assets) ===")
    match_pref, match_dirn = 0, 0
    for a in NEW:
        if a not in pref:
            print(f"  {a:10s} no capacity data")
            continue
        m64, m128 = pref[a]
        obs_pref = ("nonlinear" if min(m64, m128) < 0 else "linear" if max(m64, m128) > 0 else "tie")
        row = d[d.asset == a]
        obs_dirn = "negative" if float(row.interaction.iloc[0]) < 0 else "positive"
        acf, exp_dirn, exp_pref = PREREG[a]
        ok_p = obs_pref == exp_pref
        ok_d = obs_dirn == exp_dirn
        match_pref += int(ok_p)
        match_dirn += int(ok_d)
        print(f"  {a:10s} |ACF1|={acf:.3f}  pref: obs={obs_pref:9s} exp={exp_pref:9s} "
              f"{'MATCH' if ok_p else 'MISMATCH'} | dint: obs={obs_dirn:8s} exp={exp_dirn:8s} "
              f"{'MATCH' if ok_d else 'MISMATCH'}")

    # ---- new-cohort-only descriptive check (N=4). NOT part of the frozen Gate-A rule ----
    print("\n=== new-cohort-only descriptive check (N=4) -- NOT part of the frozen Gate-A rule ===")
    nc = d[d.set == "new"].sort_values("acf1_abs")
    if len(nc) >= 2:
        r_nc, p_nc, n_nc = sp(nc.acf1_abs, nc.interaction)
        for i, (_, r) in enumerate(nc.iterrows(), 1):
            print(f"  rank {i}: {r.asset:10s} |ACF1|={r.acf1_abs:.4f}  "
                  f"d_interaction={r.interaction:+.5f}")
        print(f"  Spearman rho (new cohort only, N={n_nc}) = {r_nc:+.3f} (p={p_nc:.3f}) -- "
              f"descriptive only, no confirmatory claim at this sample size")
    print("  NOTE: descriptive association, not a predictive selection rule; the Gate A verdict "
          "uses the pooled N=14 analysis and the frozen rule, not this block.")

    # ---------------------------------------------------------------- verdict
    rho_flip = (r_acf * 0.733) < 0
    sig = (p_acf == p_acf) and (p_acf < 0.05)
    if rho_flip or flip or contradictions or match_dirn <= 1:
        gate = "FAIL"
    elif sig and abs(r_acf) >= 0.5 and match_pref >= 3 and match_dirn >= 3:
        gate = "PASS"
    else:
        gate = "CONDITIONAL"

    print("\n" + "=" * 30)
    print("STRUCTURAL VALIDATION")
    print("=" * 30)
    print(f"New assets: {len(NEW)} ({', '.join(NEW)})")
    print(f"rho(|ACF1|, interaction): {r_acf:+.3f} (p={p_acf:.3f})   [N=10: +0.733]")
    print(f"LOO: range [{L.rho_acf1.min():+.3f}, {L.rho_acf1.max():+.3f}] sign_flip={flip}")
    print(f"Capacity contradiction: {'YES' if contradictions else 'no'}")
    print(f"Pre-registered match: preference {match_pref}/{len(NEW)}, interaction direction "
          f"{match_dirn}/{len(NEW)}")
    print(f"Gate A: {gate}")
    if gate == "FAIL":
        rec = ("Treat the candidate asset-level structural regularity as not externally "
               "validated; re-frame rho(|ACF1|) as sample-specific rather than proceed.")
    elif gate == "CONDITIONAL":
        rec = ("Keep the regularity as a candidate and require more seeds/assets (and same-source "
               "data) before any structured-operator benchmark is considered.")
    else:
        rec = ("Eligibility for a capacity-matched structured nonolinear operator benchmark holds; "
               "no NS module is implemented at this stage.")
    print(f"Recommended next action: {rec}")
    print("=" * 30)


if __name__ == "__main__":
    main()
