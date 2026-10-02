# -*- coding: utf-8 -*-
"""C1 final analysis — primary test on the NEW cohort only.

Inputs (both produced before any outcome is joined with the predictor):
  reproduction/results/c1_predictor.csv     PRIMARY predictor: acf1_abs_pre (pre-cutoff only)
  reproduction/results/c1_interactions.csv  per-asset interaction from the frozen pipeline
                                            (inter_high50_low50 = the variant used throughout the
                                             frozen evidence chain; inter_q45_q12 reported too)

Analysis, per reports/evidence_closure/23_c1_preregistration.md §6:
  - per-asset interaction = mean over the three seeds
  - Spearman rho(acf1_abs_pre, interaction) with an exploratory p-value
  - leave-one-asset-out rho range (a sign flip or a range crossing zero is a failure signal)
  - single-asset influence: the largest |delta rho| produced by dropping one asset
  - the frozen operational criteria are then applied verbatim:
        sign reversal | |rho| < 0.30 | p > 0.10 | LOO range crossing zero | single-asset driven
  - the OLD ten / pooled N=14 may only be shown as labelled context, never as the primary

Exit code 0 when the association replicates, 1 when it fails.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "reproduction", "results")
EXPECTED_SIGN = +1          # discovery chain: rho(|ACF1|, interaction) was positive
RHO_MIN, P_MAX = 0.30, 0.10


def spearman(x, y):
    """Spearman rho and a permutation-free two-sided p via the t approximation."""
    x = pd.Series(x).rank().to_numpy(float)
    y = pd.Series(y).rank().to_numpy(float)
    n = len(x)
    if n < 4:
        return float("nan"), float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    rho = float((xc * yc).sum() / denom) if denom > 0 else float("nan")
    if not np.isfinite(rho) or abs(rho) >= 1:
        return rho, 0.0
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 2))
    except Exception:
        # normal approximation if scipy is unavailable
        p = float(2 * (1 - 0.5 * (1 + np.math.erf(abs(t) / np.sqrt(2)))))
    return rho, p


def main():
    pp = os.path.join(RES, "c1_predictor.csv")
    ip = os.path.join(RES, "c1_interactions.csv")
    if not (os.path.exists(pp) and os.path.exists(ip)):
        print("INPUTS MISSING:")
        print(f"  predictor : {pp} {'ok' if os.path.exists(pp) else 'MISSING'}")
        print(f"  interactions: {ip} {'ok' if os.path.exists(ip) else 'MISSING (awaiting geng T005)'}")
        return 2

    pred = pd.read_csv(pp)
    inter = pd.read_csv(ip)
    for c in ("inter_high50_low50", "inter_q45_q12"):
        if c not in inter.columns:
            print(f"missing column {c} in interactions")
            return 3

    per = inter.groupby("tag").agg(
        interaction=("inter_high50_low50", "mean"),
        interaction_q45=("inter_q45_q12", "mean"),
        seeds=("seed", "count"),
    ).reset_index()

    df = pred[["tag", "acf1_abs_pre", "rv_pre"]].merge(per, on="tag", how="inner")
    n = len(df)
    if n == 0:
        print("no overlapping tags between predictor and interactions")
        return 4

    rho, p = spearman(df["acf1_abs_pre"], df["interaction"])
    loo = []
    for i in range(n):
        sub = df.drop(df.index[i])
        r, _ = spearman(sub["acf1_abs_pre"], sub["interaction"])
        loo.append(r)
    loo = np.array(loo, dtype=float)
    loo_min, loo_max = float(np.nanmin(loo)), float(np.nanmax(loo))
    driver = df["tag"].iloc[int(np.nanargmax(np.abs(loo - rho)))] if np.isfinite(loo).any() else ""
    max_drop = float(np.nanmax(np.abs(loo - rho))) if np.isfinite(loo).any() else float("nan")

    rho_a, p_a = spearman(df["rv_pre"], df["interaction"])   # secondary predictor

    print(f"=== C1 primary analysis | NEW COHORT ONLY | N = {n} ===")
    print(df.round(5).to_string(index=False))
    print(f"\nrho(|ACF1|_pre, interaction) = {rho:+.4f}   p = {p:.4f}   (expected sign: {'+' if EXPECTED_SIGN>0 else '-'})")
    print(f"LOO rho range              = [{loo_min:+.4f}, {loo_max:+.4f}]  "
          f"(crosses zero: {loo_min * loo_max <= 0})")
    print(f"most influential asset     = {driver}  (max |delta rho| = {max_drop:.4f})")
    print(f"secondary: rho(rv_pre, interaction) = {rho_a:+.4f}  p = {p_a:.4f}")

    fails = []
    if np.sign(rho) != EXPECTED_SIGN and np.isfinite(rho):
        fails.append("sign reversal vs the expected direction")
    if abs(rho) < RHO_MIN:
        fails.append(f"|rho| < {RHO_MIN}")
    if p > P_MAX:
        fails.append(f"p > {P_MAX}")
    if loo_min * loo_max <= 0:
        fails.append("LOO range crosses zero (sign not robust)")
    if max_drop > 0.5 * abs(rho):
        fails.append(f"single-asset driven (dropping {driver} moves rho by {max_drop:.3f})")

    print("\n=== frozen operational criteria ===")
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        print("\nVERDICT: the candidate association did NOT replicate on the new cohort.")
        print("Consequence (per 23 §6): |ACF1| is closed permanently as the structural bridge.")
        return 1
    print("  none of the failure conditions triggered")
    print("\nVERDICT: the candidate association replicated prospectively on the new cohort.")
    print("Scope: portability only. No mechanism claim, no asset-selection rule, no NS eligibility.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
