# -*- coding: utf-8 -*-
"""C1 interaction collector — thin driver around the FROZEN stratification script.

It does not reimplement anything: for each (tag, seed) it invokes
    reproduction/analysis/analyze_volatility_regimes.py --csv dataset/<TAG>-2016-2025.csv
        --tag <TAG> --seed <SEED> --models DeReFusion,revin-DLinear --rv-mode relative
and then reads the per-horizon comparison block out of the JSON that script writes
(horizons.T24.comparison.*). Reusing the frozen script means zero definition drift: the strata
(Q1+Q2 low, Q4+Q5 high), the causal rv_rel window (120) and the bootstrap seeds stay exactly as
frozen in E4/E5.

Regression test mode (--tags GSPC,BTCUSD,ETHUSD) is meant to reproduce values already recorded in
the evidence chain before C1 artefacts exist.

Outputs: reproduction/results/c1_interactions.csv and .md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
ANALYZE = os.path.join(REPO, "reproduction", "analysis", "analyze_volatility_regimes.py")
RES = os.path.join(REPO, "reproduction", "results")

COHORT = ["AAPL", "MSFT", "AMZN", "META", "TSLA", "JPM", "XOM", "WMT",
          "N225", "GDAXI", "HSI", "FTSE", "RUT",
          "GBPUSD", "AUDUSD", "USDCAD", "GOLD", "WTI", "GLD", "TLT"]


def one(tag: str, seed: int, allow_overwrite: bool = False) -> dict | None:
    csv = os.path.join(REPO, "dataset", f"{tag}-2016-2025.csv")
    if not os.path.exists(csv):
        print(f"  {tag}: dataset csv missing -> skip")
        return None
    jp = os.path.join(RES, f"volatility_stratification_{tag}_relative_s{seed}.json")
    # GUARD: never silently overwrite a stratification file that is already tracked as evidence
    # (an earlier regression run did exactly that: the values matched but the file content changed
    # because a different arm set had produced it).
    if os.path.exists(jp) and not allow_overwrite:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch",
                                  os.path.relpath(jp, REPO)], cwd=REPO,
                                 capture_output=True, text=True).returncode == 0
        if tracked:
            print(f"  {tag} s{seed}: REFUSING to overwrite tracked {os.path.basename(jp)} "
                  f"(pass --allow-overwrite to force)")
            return None
    cmd = [PY, "-W", "ignore", ANALYZE, "--csv", csv, "--tag", tag, "--seed", str(seed),
           "--models", "DeReFusion,revin-DLinear", "--rv-mode", "relative"]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=REPO, env=env)
    jp = os.path.join(RES, f"volatility_stratification_{tag}_relative_s{seed}.json")
    if r.returncode != 0 or not os.path.exists(jp):
        print(f"  {tag} s{seed}: FAILED rc={r.returncode} {(r.stderr or '')[-160:]}")
        return None
    d = json.load(open(jp, encoding="utf-8"))
    hz = d.get("horizons", {}).get("T24")
    if not hz:
        print(f"  {tag} s{seed}: no T24 horizon in json")
        return None
    c = hz.get("comparison", {})
    q = c.get("interaction_Q45_minus_Q12", {}) or {}
    h = c.get("interaction_high50_minus_low50", {}) or {}
    low = c.get("Q1+Q2 (low)", {}) or {}
    high = c.get("Q4+Q5 (high)", {}) or {}
    return {
        "tag": tag, "seed": seed, "n_samples": hz.get("n_samples"),
        "inter_q45_q12": q.get("value"), "ci_lo_q45": (q.get("ci95") or [None, None])[0],
        "ci_hi_q45": (q.get("ci95") or [None, None])[1], "sig_q45": q.get("significant"),
        "inter_high50_low50": h.get("value"), "ci_lo_h50": (h.get("ci95") or [None, None])[0],
        "ci_hi_h50": (h.get("ci95") or [None, None])[1], "sig_h50": h.get("significant"),
        "low_mse_delta": low.get("delta_mse"), "high_mse_delta": high.get("delta_mse"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    ap.add_argument("--seeds", default="2021,2022,2023")
    ap.add_argument("--out", default="c1_interactions")
    ap.add_argument("--allow-overwrite", action="store_true",
                    help="permit regenerating a stratification file that is already tracked as evidence")
    a = ap.parse_args()
    tags = [t.strip() for t in a.tags.split(",") if t.strip()] or COHORT
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    rows = []
    print(f"=== interaction collection | {len(tags)} tags x {len(seeds)} seeds ===")
    for tag in tags:
        for seed in seeds:
            r = one(tag, seed, allow_overwrite=a.allow_overwrite)
            if r:
                rows.append(r)
                print(f"  {tag:8s} s{seed}: Q45-Q12 {r['inter_q45_q12']:+.5f} "
                      f"[{r['ci_lo_q45']:+.5f},{r['ci_hi_q45']:+.5f}] | "
                      f"h50-l50 {r['inter_high50_low50']:+.5f} "
                      f"[{r['ci_lo_h50']:+.5f},{r['ci_hi_h50']:+.5f}]")
    if not rows:
        print("  nothing collected")
        return 2
    df = pd.DataFrame(rows)
    cp = os.path.join(RES, f"{a.out}.csv")
    df.to_csv(cp, index=False)
    md = ["# Interaction collection (frozen stratification script)\n",
          "`inter_high50_low50` = ΔMSE(high50) − ΔMSE(low50); negative = nonlinear-favoured in high vol.",
          ""]
    cols = list(df.columns)
    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "---|" * len(cols))
    for _, r in df.round(6).iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    mp = os.path.join(RES, f"{a.out}.md")
    open(mp, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"\nsaved: {cp}\nsaved: {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
