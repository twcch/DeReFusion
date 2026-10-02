# 00 · Repository & Experiment Audit

**Task:** DeReFusion evidence closure (protocol: 25-section evidence-closure spec).
**Date:** 2026-09-13 · **Auditor:** agent（希伯来 / ThinkBook）
**Repo:** `Alastair-Jiang/DeReFusion` (public) · local: `C:\Users\26843\Desktop\project\repos\DeReFusion`

## A. Version state

| Item | Value |
|---|---|
| Branch | `main` |
| HEAD at audit start | `16d97b8` (docs/latex: IEEEtran twins of the stage report) |
| Working tree | **dirty — 9 untracked entries**, all run artifacts (no prototype code, no modified sources) |
| Untracked (at audit) | `asset_sweep_log.txt`, `capacity_check_log.txt`, `remaining_batch_console.txt`, `remaining_batch_log.txt`, `structure_experiment_log.txt`, `structure_features_{GSPC,BTCUSD,ETHUSD}_test.csv`, `sweep_runs/` |
| Resolution | the raw artifacts above were **committed in this task** so that no raw result exists only on one machine (protocol: "只保留 summary 而删除 raw result" is forbidden) |

Recent research-relevant commits (newest first):

```
16d97b8  docs/latex: IEEEtran twins of the stage report (EN + ZH), N=10 final data
0da9823  Final diagnosis at N=10 (SOX included): PATH 1 unchanged
aa41ef0  docs/latex: stage report (EN + ZH) summarizing completed work up to 2026-09-12
c7e1a28  7-asset sweep results + final diagnosis (PATH 1, N=9 assets; SOX linear rerun pending)
24ebb8a  Asset sweep v2: two-way parallelism, per-run log files, fix model_id interpolation
99ab4e8  Final phase protocol compliance: GSPC seed merging, High/Low CIs, direction-vs-significance
563906c…9d9287a  volatility-regime stratification framework, batch runners, docs/ROADMAP
```

## B. Experiment directory structure

```
reproduction/                      ← all reproduction assets (upstream code untouched)
  analysis/                        ← analysis scripts (8)
    analyze_volatility_regimes.py  ← regime stratification (causal RV, bootstrap CIs)
    structure_routing_experiment.py← structural-state × operator proxy experiment
    structure_routing_capacity_check.py ← capacity sweep (23/64/128)
    final_diagnosis.py             ← asset-level heterogeneity + PATH gate + report generator
    asset_dependence_analysis.py   ← exploratory Spearman + LOO
    make_summary.py                ← digest builder
    check_time_confound.py         ← RV-vs-time confound check
    check_gspc_seed_sensitivity.py ← seed-merge sensitivity (added in this task)
  batches/                         ← runnable batch scripts (7) — the reproducible commands
  data/                            ← fetch/derived data helpers (3)
  logs/                            ← (empty; batch logs live at repo root, now committed)
  results/                         ← versioned per-setting JSON/TXT results (28 files)
docs/
  ROADMAP.md                       ← state of record, pitfalls, next steps
  latex/                           ← elsarticle + IEEEtran stage reports (EN/ZH) + PDFs
reports/evidence_closure/          ← THIS audit + closure reports
```

Mapping of the required experiment families to artifacts:

| Family | Location |
|---|---|
| baseline | `batches/run_batch_repro*.ps1`, `batches/run_btc_repro.ps1`, `results/` (run dirs) |
| volatility-conditioned | `analysis/analyze_volatility_regimes.py` → `results/volatility_stratification_*` |
| capacity sensitivity | `analysis/structure_routing_capacity_check.py` → `results/operator-regime-capacity.csv` |
| asset-level heterogeneity | `analysis/final_diagnosis.py` → `asset-dependence-summary.csv`, `final-diagnosis.md` |
| structural-state analysis | `analysis/structure_routing_experiment.py` → `results/structure_routing_raw.json` |
| final diagnosis | `analysis/final_diagnosis.py` → `final-diagnosis.md` |
| SOX | `results/volatility_stratification_SOX_relative_s2021.json` |
| N=10 asset analysis | `asset-dependence-summary.csv` (10 rows), `asset-dependence-exploration.md` |
| bootstrap | inside every analysis script (4000 resamples, percentile CI) |
| LOO | `asset_dependence_analysis.py` (per-feature leave-one-asset-out) |
| raw JSONL / raw prediction files | **not versioned** — see gap below |
| summary tables | `reproduction/results/*.csv`, `05_research_intelligence/*.csv` |
| configs | embedded in `batches/*.ps1` (no separate config files) |
| scripts | `reproduction/analysis/*.py`, `reproduction/batches/*.ps1` |
| logs | repo-root `*_log.txt` (committed in this task), `sweep_runs/*.log` |

## C. Reproducibility check

| Requirement | Status |
|---|---|
| explicit config | ✅ all flags in `batches/*.ps1` (model_id, model, data, T, seq_len, d_model, epochs, batch, lr, patience, seed, device) |
| dataset identified | ✅ `dataset/<ASSET>-2016-2025.csv` (row counts documented in ROADMAP) |
| train/val/test split | ✅ 70/10/20 chronological, encoded in the shared data provider |
| seed stated | ✅ `--rand_seed` in every command |
| horizon stated | ✅ `--pred_len` |
| input window stated | ✅ `--seq_len 96 --label_len 48` |
| optimizer / lr / batch | ✅ Adam / 1e-4 / 32 (in command) |
| epochs / early stopping | ✅ 30 / patience 5 |
| raw output present | ⚠️ **local only** — `results/` (27 run dirs, 2.3 MB) is git-ignored (`.gitignore:166`) |
| reproducible command | ✅ committed batch scripts reproduce each setting |

### `REPRODUCIBILITY_GAP` register

1. **`RAW_NOT_VERSIONED`** — per-run training outputs (`results/<run_dir>/`, incl. `pred.npy`, `true.npy`,
   `metrics.npy`, figures) are git-ignored and therefore not in the repository. *Mitigation:* every run is
   fully specified by a committed command in `batches/*.ps1`, and all **derived** per-setting results
   (stratification JSON/TXT, capacity CSV, structural-state CSVs, feature CSVs) **are** versioned.
   *Residual risk:* the raw tensors cannot be re-inspected without re-running (CPU cost ≈ 30–40 min/run).
2. **`SEED_COVERAGE_PARTIAL`** — the asset-level analysis is one seed per asset (GSPC two: 2021, 2022;
   the third planned seed 2023 was never run). The capacity/routing proxy experiments do have 3 seeds.
3. No other gaps: no missing config, dataset, split, seed, horizon or command for any reported number.

**No config was inferred or reconstructed in this audit** — every item above was read from the repository.
