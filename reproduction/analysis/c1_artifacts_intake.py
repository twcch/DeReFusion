# -*- coding: utf-8 -*-
"""T005 artifact intake: verify geng's returned C1 runs against the frozen protocol, then (optionally)
place them into this repository's results/ directory under the canonical run names so the frozen
stratification pipeline can consume them.

Layout expected (as declared in geng-lobster/008-T005-status.md):
    <src>/<seed>/<TAG>_<MODEL>/           metrics.npy, pred.npy, true.npy, a stored command line, log tail
    <src>/per_run_manifest.csv            tag,model,seed,status,wall_clock_min,metrics_*,pred_sha256,true_sha256,dir

Checks per run
  - metrics.npy / pred.npy / true.npy present
  - a command line is stored and contains EVERY frozen protocol value
  - pred.npy / true.npy SHA-256 match the manifest row
  - metrics.npy parses as the frozen six-value order [mae, mse, rmse, mape, mspe, r2]

Default is a DRY RUN (verify only). Pass --copy to place verified runs into results/.

Canonical destination name:
  long_term_forecast_<TAG>_96_24_<MODEL>_custom_ftMS_sl96_ll48_pl24_dm32_nh8_el2_dl1_df2048_
  expand2_dc4_fc1_ebtimeF_dtTrue_test_seed<SEED>_0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Resolve from this tracked script rather than from a machine-specific checkout.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SRC = os.path.join(REPO, "reproduction", "c1_handoff")

# frozen protocol: every one of these must appear in the stored command line
FROZEN_FLAGS = [
    "--pred_len 24", "--seq_len 96", "--label_len 48", "--d_model 32", "--moving_avg 25",
    "--train_epochs 30", "--batch_size 32", "--learning_rate 0.0001", "--patience 5",
    "--lradj cosine", "--features MS", "--target Close", "--freq b", "--no_use_gpu",
    "--enc_in 4", "--dec_in 4", "--c_out 1",
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def canonical(tag, model, seed):
    return (f"long_term_forecast_{tag}_96_24_{model}_custom_ftMS_sl96_ll48_pl24_dm32_nh8_el2_dl1_"
            f"df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_seed{seed}_0")


def find_cmdline(d):
    for fn in os.listdir(d):
        if fn.lower().endswith((".txt", ".cmd", ".ps1", ".log")):
            p = os.path.join(d, fn)
            try:
                t = open(p, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            if "run.py" in t:
                return p, t
    return None, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dest", default=os.path.join(REPO, "results"))
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--inventory", default="",
                    help="write a per-file SHA-256 inventory (CSV) of everything received")
    ap.add_argument("--exec-commit", default="",
                    help="the executor's reported commit hash of the frozen script it ran")
    ap.add_argument("--report", default="",
                    help="optional path for the machine-readable intake report; dry runs write no report by default")
    a = ap.parse_args()

    if a.inventory and not a.exec_commit:
        print("FAIL: --inventory requires --exec-commit so the executed frozen revision is recorded")
        return 2

    if not os.path.isdir(a.src):
        print(f"SOURCE ABSENT: {a.src}\n(awaiting geng's first push)")
        return 2

    man = {}
    mp = os.path.join(a.src, "per_run_manifest.csv")
    if not os.path.isfile(mp):
        print(f"FAIL: required per_run_manifest.csv absent at {mp}")
        return 2
    for _, r in pd.read_csv(mp).iterrows():
        key = (str(r.get("tag")).strip(), str(r.get("model")).strip(), int(r.get("seed")))
        if key in man:
            print(f"FAIL: duplicate manifest row for {key}")
            return 2
        man[key] = r
    print(f"manifest rows: {len(man)}")

    rows = []
    seen = set()
    for seed_dir in sorted(os.listdir(a.src)):
        sd = os.path.join(a.src, seed_dir)
        if not os.path.isdir(sd) or not seed_dir.isdigit():
            continue
        seed = int(seed_dir)
        for run in sorted(os.listdir(sd)):
            rd = os.path.join(sd, run)
            if not os.path.isdir(rd):
                continue
            tag, model = (run.split("_", 1) + ["?"])[:2]
            probs, hashes = [], {}
            for f in ("metrics.npy", "pred.npy", "true.npy"):
                if not os.path.isfile(os.path.join(rd, f)):
                    probs.append(f"missing {f}")
            if "missing pred.npy" not in probs and "missing true.npy" not in probs:
                for f in ("pred.npy", "true.npy"):
                    hashes[f] = sha256(os.path.join(rd, f))
            cp, ctext = find_cmdline(rd)
            if not cp:
                probs.append("no stored command line")
            else:
                for flag in FROZEN_FLAGS:
                    if flag not in ctext:
                        probs.append(f"cmdline missing '{flag}'")
                if f"--rand_seed {seed}" not in ctext:
                    probs.append(f"cmdline seed mismatch (expected {seed})")
                if f"--model {model}" not in ctext:
                    probs.append(f"cmdline model mismatch (expected {model})")
                if f"--data_path {tag}-2016-2025.csv" not in ctext:
                    probs.append(f"cmdline data_path mismatch (expected {tag}-2016-2025.csv)")
            m = man.get((tag, model, seed))
            key = (tag, model, seed)
            seen.add(key)
            if m is None:
                probs.append("no manifest row")
            elif hashes:
                for f, col in (("pred.npy", "pred_sha256"), ("true.npy", "true_sha256")):
                    want = str(m.get(col, "")).strip().lower()
                    if not want:
                        probs.append(f"manifest missing {col}")
                    elif want != hashes[f]:
                        probs.append(f"{f} sha mismatch vs manifest")
            if os.path.isfile(os.path.join(rd, "metrics.npy")):
                try:
                    v = np.load(os.path.join(rd, "metrics.npy"))
                    if v.size != 6:
                        probs.append(f"metrics has {v.size} values, expected 6")
                    elif not np.isfinite(v).all():
                        probs.append("metrics contains non-finite value")
                except Exception as e:
                    probs.append(f"metrics unreadable: {type(e).__name__}")
            rows.append({"seed": seed, "tag": tag, "model": model, "ok": not probs,
                         "problems": "; ".join(probs)})
            print(f"  {seed} {tag:8s} {model:14s} {'OK' if not probs else 'FAIL'} "
                  f"{'' if not probs else '- ' + rows[-1]['problems'][:90]}")

    if not rows:
        print("no run directories found under the source")
        return 3

    # A manifest entry without a matching run directory is also an integrity failure.
    for key in sorted(set(man) - seen):
        rows.append({"seed": key[2], "tag": key[0], "model": key[1], "ok": False,
                     "problems": "manifest row has no matching run directory"})
        print(f"  {key[2]} {key[0]:8s} {key[1]:14s} FAIL - manifest row has no matching run directory")

    n_ok = sum(1 for r in rows if r["ok"])
    print(f"\n=== {n_ok}/{len(rows)} runs pass protocol+hash checks ===")

    if a.copy and n_ok != len(rows):
        print("REFUSE COPY: this batch has failed checks; no partial batch is placed into results/")
    elif a.copy:
        copied = 0
        for r in rows:
            if not r["ok"]:
                continue
            src_dir = os.path.join(a.src, str(r["seed"]), f"{r['tag']}_{r['model']}")
            dst = os.path.join(a.dest, canonical(r["tag"], r["model"], r["seed"]))
            os.makedirs(dst, exist_ok=True)
            for f in ("metrics.npy", "pred.npy", "true.npy"):
                shutil.copyfile(os.path.join(src_dir, f), os.path.join(dst, f))
            # re-verify after copy
            bad = [f for f in ("pred.npy", "true.npy")
                   if sha256(os.path.join(src_dir, f)) != sha256(os.path.join(dst, f))]
            if bad:
                print(f"  !! copy verification failed for {dst}: {bad}")
            else:
                copied += 1
        print(f"copied {copied} verified runs into {a.dest}")
    else:
        print("(dry run: nothing copied; pass --copy to place verified runs into results/)")

    if a.report:
        pd.DataFrame(rows).to_csv(a.report, index=False)
        print(f"intake report written: {a.report}")

    # own role per the operator's de-duplication notice: per-file SHA-256 inventory of the handoff
    if a.inventory:
        inv = []
        for dirpath, _, files in os.walk(a.src):
            for fn in sorted(files):
                fp = os.path.join(dirpath, fn)
                rel = os.path.relpath(fp, a.src)
                inv.append({"relpath": rel, "bytes": os.path.getsize(fp), "sha256": sha256(fp)})
        pd.DataFrame(inv).to_csv(a.inventory, index=False)
        print(f"inventory written: {a.inventory} ({len(inv)} files)")
        receipt = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": os.path.abspath(a.src),
            "inventory": os.path.abspath(a.inventory),
            "executor_reported_commit": a.exec_commit,
            "runs_seen": len(rows),
            "runs_passing": n_ok,
        }
        receipt_path = a.inventory + ".receipt.json"
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"handoff receipt written: {receipt_path}")

    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
