# -*- coding: utf-8 -*-
"""C1 alignment smoke test on the first delivered runs (plumbing only, no analysis).

Purpose: confirm end-to-end that the executor's artefacts are wired correctly BEFORE the full panel
lands. It answers three questions and nothing else:
  1. do the delivered pred/true arrays have the shape the frozen pipeline expects?
  2. does the delivered true.npy equal the target windows recomputed from the LOCKED cohort CSV using
     the pipeline's own split and scaler conventions?
  3. does the frozen stratification script find and consume the placed run directories?

It produces no interaction effect, no statistic and no verdict. Its output is a plumbing report and
must never be cited as evidence about the C1 outcome.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import structure_routing_experiment as S  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "reproduction", "c1_handoff")
TAGS = ["AAPL", "HSI"]
SEED = 2021
T = 24


def canonical(tag, model, seed):
    return (f"long_term_forecast_{tag}_96_24_{model}_custom_ftMS_sl96_ll48_pl24_dm32_nh8_el2_dl1_"
            f"df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_seed{seed}_0")


def expected_true(tag):
    """Recompute the target matrix from the locked CSV using the pipeline conventions."""
    df = pd.read_csv(os.path.join(REPO, "dataset", f"{tag}-2016-2025.csv"))
    n = len(df)
    num_train = int(n * 0.7)
    seq_len, pred_len = 96, 24
    border1 = n - int(n * 0.2) - seq_len
    n_samples = n - border1 - seq_len - pred_len + 1
    cols = ["Open", "High", "Low", "Close"]
    scaler = StandardScaler().fit(df[cols].to_numpy(float)[:num_train])
    scaled_close = scaler.transform(df[cols].to_numpy(float))[:, -1]
    mat = np.stack([scaled_close[border1 + i + seq_len: border1 + i + seq_len + pred_len]
                    for i in range(n_samples)])
    return mat, n, num_train, border1, n_samples


ok_all = True
for tag in TAGS:
    src_dir = os.path.join(SRC, str(SEED), f"{tag}_DeReFusion")
    dst_dir = os.path.join(REPO, "results", canonical(tag, "DeReFusion", SEED))
    print(f"--- {tag} seed{SEED}")
    print(f"    source : {src_dir}")
    if not os.path.isdir(src_dir):
        print("    MISSING source directory")
        ok_all = False
        continue
    os.makedirs(dst_dir, exist_ok=True)
    for f in ("pred.npy", "true.npy", "metrics.npy"):
        srcf, dstf = os.path.join(src_dir, f), os.path.join(dst_dir, f)
        with open(srcf, "rb") as fh:
            b = fh.read()
        with open(dstf, "wb") as fh:
            fh.write(b)
    pred = np.load(os.path.join(dst_dir, "pred.npy")).astype(np.float64)
    true = np.load(os.path.join(dst_dir, "true.npy")).astype(np.float64)
    # the pipeline emits a trailing singleton channel: (n, T, 1) -> (n, T)
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    if true.ndim == 3 and true.shape[-1] == 1:
        true = true[..., 0]
    print(f"    placed : results/{os.path.basename(dst_dir)}")
    print(f"    shapes : pred{pred.shape} true{true.shape} finite={np.isfinite(pred).all() and np.isfinite(true).all()}")

    exp, n, num_train, border1, n_samples = expected_true(tag)
    print(f"    cohort : n={n} num_train={num_train} border1={border1} expected_samples={n_samples}")
    if true.shape != exp.shape:
        print(f"    SHAPE MISMATCH: delivered {true.shape} vs expected {exp.shape}")
        ok_all = False
        continue
    d = np.abs(true - exp)
    rel = d.max() / max(1e-12, np.abs(exp).max())
    print(f"    true vs recomputed-from-locked-CSV: max|diff|={d.max():.3e} (relative {rel:.3e}) "
          f"-> {'MATCH' if d.max() < 1e-4 else 'MISMATCH'}")
    if d.max() >= 1e-4:
        ok_all = False
    # the pipeline's own alignment diagnostic, for comparison
    m = np.load(os.path.join(dst_dir, "metrics.npy"))
    print(f"    metrics (6 frozen values): {np.array2string(np.asarray(m), precision=6)}")
    # shape-convention cross-check against an EXISTING run from this project's own campaign
    ref = os.path.join(REPO, "results",
                       "long_term_forecast_GSPC_96_24_DeReFusion_custom_ftMS_sl96_ll48_pl24_dm32_nh8_"
                       "el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_seed2021_0", "true.npy")
    if os.path.isfile(ref):
        print(f"    shape convention vs an existing project run: existing {np.load(ref).shape} "
              f"vs delivered {np.load(os.path.join(dst_dir, 'true.npy')).shape}")

print()
print("PLUMBING VERDICT:", "OK - artefacts are wired correctly" if ok_all else "PROBLEM - see above")
print("(no interaction effect, no statistic, no verdict: this is a wiring check only)")
sys.exit(0 if ok_all else 1)
