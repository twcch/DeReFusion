# 20 · F1 pre-registration — capacity-controlled breadth panel (LOCKED before execution)

**Locked:** 2026-09-14 12:36 (Asia/Shanghai) · **Base commit at lock time:** `bc6ae5c`
**Authority:** `reports/evidence_closure/16_post_gate_experiment_program.md` §3 (branch F, experiment
F1), with the criterion tightened in `16a` §4 and the shared position in `19`.
**Rule change:** none. This file freezes *what will be run and how it will be judged*; it alters no
definition, threshold, seed, metric, split, horizon, window or capacity setting.

## 1. Purpose

Gate A is `FAIL` (`14_*`). Under `16` §3 the structural-bridge direction closes, and the question
becomes the more basic one: **is asset-level operator preference itself reproducible** once capacity
and seeds are controlled — or was the three-asset result (ETHUSD nonlinear 36/36, BTCUSD linear
36/36, GSPC near parity) too narrow to be a stable property?

F1 is an **ordinary-operator stability** experiment. It does **not** test `|ACF1|`, does **not**
test any NS-inspired operator, and does **not** change Gate A.

## 2. Prerequisites (from `16` §3) — status at lock time

| Prerequisite | Status |
|---|---|
| `14_*` records `FAIL` under the unchanged rule | ✅ `14_structural_validation_result.md` (Gate A = FAIL, capacity contradiction BOE/EASTMONEY) |
| The asset list and file hashes are locked **before** execution | ✅ this file (§3) |
| Dry inspection confirms identical preprocessing / optimiser / stopping to E4–E5 | ✅ same script, same defaults; the only CLI differences are the asset set and the output suffix (§5) |
| Storage naming is idempotent and cannot overwrite existing raw results | ✅ `_f1` suffix (§5) |

## 3. Locked asset list (all seven original assets lacking capacity-controlled results)

| Asset | Rows | SHA-256 (dataset/<TAG>-2016-2025.csv) |
|---|---|---|
| USDJPY | 2602 | `5415bff559ba61d4c303714d43046c643231bceb835d5a3eaa9efabe57a1b3bf` |
| EURUSD | 2602 | `ace2afcfbdc609568362124c2e2e111def10a33b6b4a8658afb54dfd3a9bb14c` |
| SOX | 2513 | `3cbbe90dc25756dd7a4d31a88ccaa5717dfd45e08f30420b2960514496e95d9d` |
| DJI | 2513 | `80fb63a14d07e6d40d6903daa8f1a21fefbeedb6b4dc2bdd3b712c87ac3201cd` |
| BABA | 2513 | `c22d599215866e76bff9952a237c4d7484210f724d13c76afd02c05b30042a38` |
| NVO | 2513 | `9c45986782c38e4ad19538f3533d06fe393a242f8a8ae965ea18c6d1d2dea1aa` |
| TM | 2513 | `63b66333d7aae23a56f50e1cef644b016e7cdb23b7b4d325ceca337b4b16ccb0` |

**No selection by `|ACF1|` or by any other feature** — all seven are included, as `16` §3 requires.

## 4. Frozen settings (identical to E4/E5)

Linear map vs MLP proxy operators; widths **23, 64, 128**; seeds **2021, 2022, 2023**; the twelve
pre-registered structural states used as **evaluation strata** (not training multipliers); the same
preprocessing, optimiser, learning rate, batch size and early-stopping as E4/E5; paired bootstrap
per state. Cost: `7 assets × 3 widths × 3 seeds × 2 operators = 126 fits` (≈5–7 min).

## 5. Output naming (cannot overwrite anything)

`--suffix _f1` →
`reproduction/results/operator-regime-capacity_f1.csv`,
`05_research_intelligence/operator-regime-capacity-summary_f1.csv`,
`05_research_intelligence/capacity-sign-reversal_f1.md`.
The existing artifacts (`operator-regime-capacity.csv` for the three assets, `_sv` for the
structural-validation cohort) are untouched. The only code change made for F1 is the addition of an
explicit `--suffix` argument — **output naming only**.

## 6. Pre-specified analysis and judgement rules

For each asset and each width: mean ΔMSE over states and seeds (negative = nonlinear-favoured).

- **Seed-stable** at a width: the per-seed mean ΔMSE has the same sign in **all three seeds**.
- **Width-stable**: seed-stable at **both** width 64 and width 128 **with the same sign**.
- **Stable asset**: width-stable (that is the strictest reading, and it is the one `16` §3 defines:
  *"an individual asset is not considered stable if its mean ΔMSE sign differs between widths 64 and
  128 or across seeds"*).

**Success criterion (tightened in `16a` §4 — the ETH/BTC pair already exists, so re-finding it
proves nothing):**

> F1 **succeeds** only if **at least two additional assets** are stable *and* at least one **new**
> opposite-sign pair exists among the F1 assets (i.e. breadth beyond the three assets that generated
> the claim).

**F1 fails** if fewer than two additional assets are stable, or if no new opposite-sign pair appears.

## 7. Stop rule

If F1 fails, the asset-heterogeneity claim is recorded as **unreproduced**, and structural-candidate
work and NS-inspired operator work stop (`16` §3). If F1 succeeds, the only licensed upgrade is
*"asset-level heterogeneity replicated on the existing source"* — **no feature-based explanation
follows automatically**, and `|ACF1|` stays sample-specific until an independent prospective cohort
replicates it.

## 8. Explicit non-claims

- F1 does not re-open Gate A, does not restore `NS_BENCHMARK_ELIGIBLE`, and does not authorise the
  dormant runner on `draft/ns-benchmark-runner`.
- F1 results are reported in full, including per-seed and per-state detail, and failures/tie cases
  are reported with the same prominence as successes.
