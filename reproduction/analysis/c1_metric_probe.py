# -*- coding: utf-8 -*-
"""Metric-consistency probe: are metrics.npy reproducible from pred.npy/true.npy?

Uses the PIPELINE'S OWN metric functions (utils.metrics) so no convention is assumed, and compares
(a) the two delivered C1 runs against (b) an existing run from this project's own campaign. If the
existing run shows the same relation, the behaviour is a pipeline property; if only the delivered
runs deviate, it is a provenance finding.
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from utils.metrics import MAE, MSE, RMSE, MAPE, MSPE, R2  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load(d):
    p = np.load(os.path.join(d, "pred.npy")).astype(np.float64)
    t = np.load(os.path.join(d, "true.npy")).astype(np.float64)
    m = np.load(os.path.join(d, "metrics.npy")).astype(np.float64)
    return p, t, m


def report(label, d):
    if not os.path.isdir(d):
        print(f"{label}: MISSING {d}")
        return
    p, t, m = load(d)
    vals = [MAE(p, t), MSE(p, t), RMSE(p, t), MAPE(p, t), MSPE(p, t), R2(p, t)]
    names = ["mae", "mse", "rmse", "mape", "mspe", "r2"]
    print(f"{label}: shapes pred{p.shape} true{t.shape}")
    print(f"   {'metric':6s} {'reported':>14s} {'recomputed':>14s} {'abs diff':>11s} {'rel diff':>10s}")
    for n, rep, calc in zip(names, m, vals):
        rd = abs(rep - calc) / max(1e-12, abs(rep))
        print(f"   {n:6s} {rep:14.8f} {calc:14.8f} {abs(rep-calc):11.2e} {rd:10.2e}")
    # variant: metrics over the flattened arrays vs over per-sample means
    print(f"   variant  MSE(flat)={MSE(p.reshape(-1), t.reshape(-1)):.8f}  "
          f"MSE(per-sample-mean)={np.mean((p-t).reshape(p.shape[0], -1).mean(1)**2):.8f}")


print("=== delivered C1 runs ===")
for tag in ("AAPL", "HSI"):
    cands = [x for x in os.listdir(os.path.join(REPO, "results"))
             if x.startswith(f"long_term_forecast_{tag}_96_24_DeReFusion")]
    if cands:
        report(f"{tag} (delivered)", os.path.join(REPO, "results", cands[0]))

print("\n=== existing project runs (reference) ===")
for tag in ("GSPC", "BTCUSD"):
    cands = [x for x in os.listdir(os.path.join(REPO, "results"))
             if x.startswith(f"long_term_forecast_{tag}_96_24_DeReFusion") and "seed2021" in x]
    if cands:
        report(f"{tag} (existing)", os.path.join(REPO, "results", cands[0]))
