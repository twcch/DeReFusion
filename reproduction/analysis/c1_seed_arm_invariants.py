# -*- coding: utf-8 -*-
"""C1 seed/arm invariants over the complete 120-run panel (verification only, no analysis).

Two invariants should hold by construction, and neither had been checked once the full panel landed:

  I1  For a given asset, `true.npy` must be IDENTICAL across all six runs (2 arms x 3 seeds).
      The ground-truth target windows depend only on the asset's split, not on the seed or the model;
      any difference would mean a different split or a different dataset was used.

  I2  For a given asset and arm, `pred.npy` must be pairwise DISTINCT across the three seeds, and for
      a given asset and seed it must differ between the two arms. Identical predictions would mean the
      seed (or the arm) had no effect - a silent seeding or wiring failure.

Report only: counts of satisfied/violated invariants with the offending runs named.
"""
from __future__ import annotations

import hashlib
import os
import sys
from itertools import combinations

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = r"C:\Users\26843\Desktop\project\repos\DeReFusion"
TAGS = ["AAPL", "MSFT", "AMZN", "META", "TSLA", "JPM", "XOM", "WMT", "N225", "GDAXI",
        "HSI", "FTSE", "RUT", "GBPUSD", "AUDUSD", "USDCAD", "GOLD", "WTI", "GLD", "TLT"]
ARMS = ["DeReFusion", "revin-DLinear"]
SEEDS = [2021, 2022, 2023]


def run_dir(tag, arm, seed):
    return os.path.join(REPO, "results",
                        f"long_term_forecast_{tag}_96_24_{arm}_custom_ftMS_sl96_ll48_pl24_dm32_"
                        f"nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_seed{seed}_0")


def h(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


i1_ok = i1_bad = 0
i2_ok = i2_bad = 0
problems = []

print("=== I1: true.npy identical across the 6 runs of each asset ===")
for tag in TAGS:
    hs = {}
    for arm in ARMS:
        for seed in SEEDS:
            p = os.path.join(run_dir(tag, arm, seed), "true.npy")
            if not os.path.isfile(p):
                problems.append(f"{tag} {arm} {seed}: true.npy missing")
                continue
            hs[(arm, seed)] = h(p)
    if len(set(hs.values())) == 1 and len(hs) == 6:
        i1_ok += 1
    else:
        i1_bad += 1
        groups = {}
        for k, v in hs.items():
            groups.setdefault(v, []).append(k)
        print(f"  {tag}: {len(groups)} distinct true.npy across {len(hs)} runs -> VIOLATION")
        for v, ks in groups.items():
            print(f"     {v[:12]} {ks}")
        problems.append(f"{tag}: true.npy not unique across arms/seeds")
print(f"  I1 satisfied for {i1_ok}/{len(TAGS)} assets, violated for {i1_bad}")

print("\n=== I2a: pred.npy distinct across the 3 seeds (per asset, per arm) ===")
for tag in TAGS:
    for arm in ARMS:
        hs = {}
        for seed in SEEDS:
            p = os.path.join(run_dir(tag, arm, seed), "pred.npy")
            if os.path.isfile(p):
                hs[seed] = h(p)
        if len(hs) < 2:
            continue
        if len(set(hs.values())) == len(hs):
            i2_ok += 1
        else:
            i2_bad += 1
            seen = {}
            for s, v in hs.items():
                seen.setdefault(v, []).append(s)
            same = {v: ss for v, ss in seen.items() if len(ss) > 1}
            print(f"  {tag} {arm}: identical pred.npy across seeds {list(same.values())} -> VIOLATION")
            problems.append(f"{tag} {arm}: identical predictions across seeds")
print(f"  I2a satisfied for {i2_ok} (asset, arm) pairs, violated for {i2_bad}")

print("\n=== I2b: pred.npy differs between arms (per asset, per seed) ===")
i2b_ok = i2b_bad = 0
for tag in TAGS:
    for seed in SEEDS:
        a = os.path.join(run_dir(tag, "DeReFusion", seed), "pred.npy")
        b = os.path.join(run_dir(tag, "revin-DLinear", seed), "pred.npy")
        if os.path.isfile(a) and os.path.isfile(b):
            if h(a) != h(b):
                i2b_ok += 1
            else:
                i2b_bad += 1
                print(f"  {tag} seed{seed}: both arms produced identical pred.npy -> VIOLATION")
                problems.append(f"{tag} seed{seed}: arms indistinguishable")
print(f"  I2b satisfied for {i2b_ok}/60, violated for {i2b_bad}")

print("\n=== summary ===")
if problems:
    print(f"  {len(problems)} problem(s):")
    for p in problems[:20]:
        print(f"    - {p}")
else:
    print("  all invariants hold: true arrays are asset-determined; predictions vary with seed and arm")
sys.exit(0 if not problems else 1)
