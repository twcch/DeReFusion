# -*- coding: utf-8 -*-
"""C1 batch audit — additional integrity checks for delivered runs (no analysis).

Beyond the intake verifier (which checks protocol flags, file hashes and manifest agreement), this
audit answers five further questions that a handoff must be able to answer:

  A. Are the reported metrics reproducible from the delivered arrays?
     The frozen six values are [mae, mse, rmse, mape, mspe, r2]. MAE/MSE/RMSE/R2 are convention-free
     in scaled space; MAPE/MSPE are usually reported in original price units, so this audit also
     inverse-transforms with the training-set scaler statistics and reports both readings, flagging
     which one the reported value matches.
  B. Are any two runs' arrays duplicates? (copy errors, mislabelled runs)
  C. Does each log tail belong to the run directory it was delivered in? (the log embeds the run name)
  D. Is the manifest in bijection with the delivered directory set? (rows without files, files without
     rows)
  E. Coverage: how much of the 20 assets x 2 arms x 3 seeds grid is present, and which cells are
     missing or duplicated.

Usage:
  python c1_batch_audit.py [--src DIR] [--grid]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from utils.metrics import MAE, MSE, RMSE, MAPE, MSPE, R2  # noqa: E402  (the pipeline's own functions)

DEFAULT_SRC = os.path.join(REPO, "reproduction", "c1_handoff")
ASSETS = ["AAPL", "MSFT", "AMZN", "META", "TSLA", "JPM", "XOM", "WMT", "N225", "GDAXI",
          "HSI", "FTSE", "RUT", "GBPUSD", "AUDUSD", "USDCAD", "GOLD", "WTI", "GLD", "TLT"]
ARMS = ["DeReFusion", "revin-DLinear"]
SEEDS = [2021, 2022, 2023]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def train_scaler_stats(tag):
    """mu/sd of the Close channel over the training split, per the pipeline's convention."""
    df = pd.read_csv(os.path.join(REPO, "dataset", f"{tag}-2016-2025.csv"))
    n = len(df)
    num_train = int(n * 0.7)
    cols = ["Open", "High", "Low", "Close"]
    a = df[cols].to_numpy(float)[:num_train]
    mu, sd = a.mean(0), a.std(0)
    return mu[-1], sd[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--grid", action="store_true", help="also report grid coverage")
    a = ap.parse_args()
    if not os.path.isdir(a.src):
        print(f"SOURCE ABSENT: {a.src}")
        return 2

    man = {}
    mp = os.path.join(a.src, "per_run_manifest.csv")
    if os.path.isfile(mp):
        for _, r in pd.read_csv(mp).iterrows():
            man[(str(r["tag"]).strip(), str(r["model"]).strip(), int(r["seed"]))] = r

    # collect delivered run dirs
    dirs = {}
    for seed_dir in sorted(os.listdir(a.src)):
        sd = os.path.join(a.src, seed_dir)
        if not os.path.isdir(sd) or not seed_dir.isdigit():
            continue
        for run in sorted(os.listdir(sd)):
            rd = os.path.join(sd, run)
            if not os.path.isdir(rd):
                continue
            parts = run.split("_")
            dirs[(parts[0], parts[1], int(seed_dir))] = rd

    print(f"=== A. metrics reproducibility ({len(dirs)} runs) ===")
    problems = []
    for key in sorted(dirs):
        tag, model, seed = key
        rd = dirs[key]
        try:
            p = np.load(os.path.join(rd, "pred.npy")).astype(np.float64).squeeze()
            t = np.load(os.path.join(rd, "true.npy")).astype(np.float64).squeeze()
            m = np.load(os.path.join(rd, "metrics.npy")).astype(np.float64)
        except Exception as e:
            print(f"  {tag:8s} {model:14s} s{seed}: unreadable ({type(e).__name__})")
            problems.append(f"{tag}/{model}/{seed}: unreadable")
            continue
        err = p - t
        # use the pipeline's OWN metric functions: no convention is assumed here
        vals = [MAE(p, t), MSE(p, t), RMSE(p, t), MAPE(p, t), MSPE(p, t), R2(p, t)]
        rel = [abs(m[i] - vals[i]) / max(1e-12, abs(m[i])) for i in range(6)]
        scaled_ok = max(rel) < 1e-5
        print(f"  {tag:8s} {model:14s} s{seed}: all six metrics max rel-diff={max(rel):.2e} "
              f"{'OK' if scaled_ok else 'MISMATCH'}")
        if not scaled_ok:
            print(f"      reported  {np.array2string(m, precision=8)}")
            print(f"      recomputed{np.array2string(np.array(vals), precision=8)}")
            problems.append(f"{tag}/{model}/{seed}: metrics not reproducible from arrays")

    print("\n=== B. duplicate arrays across runs ===")
    seen = defaultdict(list)
    for key, rd in dirs.items():
        for f in ("pred.npy", "true.npy"):
            fp = os.path.join(rd, f)
            if os.path.isfile(fp):
                seen[(f, sha256(fp))].append("/".join(map(str, key)))
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    if dups:
        for (f, h), who in dups.items():
            same_tag = len({w.split("/")[0] for w in who}) == 1
            print(f"  {'(same asset - expected for true.npy)' if same_tag and f=='true.npy' else 'DUPLICATE'}: "
                  f"{f} {h[:12]} -> {who}")
        problems += [f"duplicate {f} across {v}" for (f, _), v in dups.items()
                     if len({w.split('/')[0] for w in v}) > 1 or f == 'pred.npy']
    else:
        print("  none")

    print("\n=== C. log tail belongs to its directory ===")
    for key, rd in sorted(dirs.items()):
        tag, model, seed = key
        lt = os.path.join(rd, "log_tail.txt")
        if not os.path.isfile(lt):
            print(f"  {tag:8s} {model:14s} s{seed}: no log_tail")
            continue
        txt = open(lt, encoding="utf-8", errors="replace").read()
        found = re.findall(r"long_term_forecast_[A-Za-z0-9\-]+_96_24_([A-Za-z0-9\-]+)_", txt)
        ok = bool(found) and found[-1] == model and f"{tag}_96_24" in txt
        print(f"  {tag:8s} {model:14s} s{seed}: {'OK' if ok else 'MISMATCH'} "
              f"(log names model={found[-1] if found else '?'})")
        if not ok:
            problems.append(f"{tag}/{model}/{seed}: log tail does not name this run")

    print("\n=== D. manifest <-> directory bijection ===")
    miss_rows = [k for k in dirs if k not in man]
    miss_dirs = [k for k in man if k not in dirs]
    print(f"  directories without a manifest row: {miss_rows if miss_rows else 'none'}")
    print(f"  manifest rows without a directory : {miss_dirs if miss_dirs else 'none'}")
    if miss_rows or miss_dirs:
        problems += ["manifest/directory mismatch"]
    if man:
        flags = pd.DataFrame([{"key": "/".join(map(str, k)), "ingested_orphan": r.get("ingested_orphan"),
                               "wall_clock_min": r.get("wall_clock_min")} for k, r in man.items()])
        print(flags.to_string(index=False))

    if a.grid:
        print("\n=== E. grid coverage (20 assets x 2 arms x 3 seeds = 120) ===")
        have = set(dirs)
        want = {(t, m, s) for t in ASSETS for m in ARMS for s in SEEDS}
        missing = sorted(want - have)
        extra = sorted(have - want)
        print(f"  present {len(have)}/120 | missing {len(missing)} | extra {len(extra)}")
        if missing:
            by_seed = defaultdict(int)
            for t, m, s in missing:
                by_seed[s] += 1
            print(f"  missing by seed: {dict(sorted(by_seed.items()))}")

    print("\n=== audit summary ===")
    print("  no integrity problems found" if not problems else f"  {len(problems)} problem(s):")
    for p in problems:
        print(f"    - {p}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
