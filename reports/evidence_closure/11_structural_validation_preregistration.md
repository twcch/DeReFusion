# 11 · Structural Validation — Pre-registration (locked before any operator run)

**Date:** 2026-09-13 · **Status:** LOCKED — written and committed **before** any operator
experiment was run on the new assets. No expected direction below may be edited after results
are seen.

## 1. Why new assets, and where they come from

The protocol requires external validation of the asset-level structural association
(ρ(|ACF1|, Δ_interaction) = +0.733, ρ(RV, Δ) = +0.661) on **new** assets chosen *before*
seeing any operator result.

**Data-acquisition record.** The originally preferred source (Yahoo Finance) is unreachable from
this host: the Windows system proxy `127.0.0.1:7897` is enabled but its TLS forwarding is broken,
so every HTTPS request through the system proxy dies with `SSL handshake timeout`. All four
previously tried channels failed (Yahoo direct, `curl_cffi` — no Python-3.13 wheel on the
configured mirror, Stooq, `web_fetch`). Requests that **explicitly bypass the proxy** reach the
network; the one working market-data source found is **Sohu (`q.stock.sohu.com`)**.

**Provenance (must be quoted in any write-up):**
- the original ten assets were fetched from **Yahoo Finance** (2016–2025 daily OHLC);
- the new assets are fetched from **Sohu** with the same schema `date,Open,High,Low,Close`;
- the new pool is **A-share stocks and ETFs**, so the extended set mixes markets.
This is a documented deviation from "same source", not a silent one. It affects levels
(market microstructure differs) but not the pre-registered *rank* association that is being
validated.

## 2. Candidate pool actually fetched (31 new datasets)

`reproduction/data/fetch_pool_sohu.py` (indices/ETFs) + `fetch_pool_sohu2.py` (A-share large
caps + ETFs). Features computed with the **unchanged** definition (test-window medians,
`SEQ_LEN = 96`, `border1 = n - int(n·0.2) - 96`) by
`reproduction/analysis/candidate_features.py` → `reproduction/results/candidate_pool_features.csv`.

Candidates with **short history (< 2300 rows) were excluded from selection** by rule, because
they do not cover the same sample period: BOND10Y (2006), CATL (1815), GREE (2251), SECETF
(2262), TECHETF (1422), VANKE (2283).

## 3. Selection rule (fixed before any result)

> From the new candidates with full coverage (≥ 2300 rows), take the **2 with the lowest |ACF1|**
> and the **2 with the highest |ACF1|**.

Result of applying the rule:

| Group | Asset | Source | \|ACF1\| | realized vol | jump ratio | trend pers. | sign pers. | skew | kurtosis |
|---|---|---|---|---|---|---|---|---|---|
| **low** | **BYD** | Sohu `cn_002594` | **0.0324** | 0.0233 | 0.0105 | 0.9827 | 0.0526 | 0.2970 | 2.3610 |
| **low** | **BOE** | Sohu `cn_000725` | **0.0408** | 0.0154 | 0.0211 | 0.5637 | 0.0421 | 0.0563 | 3.4428 |
| **high** | **EASTMONEY** | Sohu `cn_300059` | **0.1296** | 0.0236 | 0.0211 | 0.9636 | 0.0632 | 0.9447 | 4.3909 |
| **high** | **YANGHE** | Sohu `cn_002304` | **0.1307** | 0.0151 | 0.0211 | 0.8758 | 0.1158 | 0.3199 | 3.8573 |

Reference positions of the existing ten: |ACF1| ranges 0.0533 (ETHUSD) … 0.1562 (NVO);
the new low pair extends the range **downward** (0.0324, 0.0408 — below every existing asset);
the new high pair sits at 0.13, **between SOX (0.1145) and NVO (0.1562)**. *Limitation recorded:*
the pool contains no new asset above 0.131, so the "high" side is mid-high, not extreme.

## 4. Pre-registered expected directions (do not edit after results)

Per protocol §8, both expected directions are recorded per new asset, plus the conflict flag.

| Asset | Expected **|ACF1|** direction | Expected **RV** direction | Conflict? |
|---|---|---|---|
| BYD (low |ACF1| 0.032) | interaction **more negative**; operator preference → **nonlinear-favoured** | RV is high (0.0233) → interaction expected **more positive** | ⚠️ **conflict**: |ACF1| and RV predict opposite signs |
| BOE (low |ACF1| 0.041) | interaction **more negative**; → **nonlinear-favoured** | RV mid (0.0154) → mildly positive | partial conflict (magnitudes differ) |
| EASTMONEY (high |ACF1| 0.130) | interaction **more positive**; → **linear-favoured** | RV high (0.0236) → **more positive** | ✅ **aligned** (both predict positive) |
| YANGHE (high |ACF1| 0.131) | interaction **more positive**; → **linear-favoured** | RV mid (0.0151) → mildly positive | ✅ aligned |

**Pre-registered decision table for Gate A** (from the protocol):

- **PASS** if (1) the new assets' operator preference is broadly consistent with the association,
  (2) the |ACF1| Spearman direction does not flip, (3) no fatal LOO sign flip, (4) the result is
  not driven by one new asset, (5) widths 64 and 128 do not contradict each other,
  (6) no protocol/leakage/selection problem.
- **CONDITIONAL** if the direction holds but statistical support weakens markedly.
- **FAIL** on sign reversal, strong single-asset dependence, capacity contradiction, or
  disappearance of the association.

## 5. Frozen protocol for the new runs (nothing changed)

- Framework level (interaction effect): `DeReFusion` and `revin-DLinear`, `T = 24`, `seq_len 96`,
  `label_len 48`, `d_model 32`, 30 epochs, patience 5, lr 1e-4, batch 32, cosine, `seed 2021`,
  `--no_use_gpu` — identical to the existing ten assets.
- Capacity-controlled operator preference: the proxy experiment's operators (linear 9,240 params;
  MLP) and its 12 pre-registered structural states, at **width 64 and width 128**, **3 seeds**
  (2021/2022/2023), everything else identical to E4/E5.
- Metrics kept per (asset, width, seed): operator MSEs, ΔMSE, paired win rate, 95% bootstrap CI;
  raw outputs written to CSV/JSON under `reproduction/results/`.
- The existing ten assets, their features, thresholds and statistics are **not** recomputed or
  altered; the updated analysis simply appends four assets and recomputes ρ, p and LOO on N = 14.

## 6. What this pre-registration does *not* claim

It does not claim the association is causal, does not re-frame the volatility conclusion
(still **INCONSISTENT**), and does not authorise any NS / gating / router work.
