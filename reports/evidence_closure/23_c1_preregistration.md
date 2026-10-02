# 23 · C1 pre-registration skeleton — independent prospective same-source cohort

**Status:** **COHORT LOCKED** (2026-09-14). Rules were locked at drafting time; the cohort slots are
now filled from the delivered data (see §8). This file still authorises no run by itself: C1 may
only execute after the owner authorises it.
**Owner instruction:** 2026-09-14 13:06 ("都办了") — draft this skeleton and dispatch the two geng
tasks. Base commit at drafting time: `0a48245`.

**Lineage:** `16` §4 (branch C, experiment C1) with the scope corrections of `19` (shared position
with the external agent) and `22` (adopted F1 framing and task allocation).

---

## 1. What C1 tests — and what it must never be used for

**Tests:** whether the candidate structural association (`|ACF1|` ↔ interaction sign) survives in a
genuinely independent, single-source, **prospective** cohort — i.e. whether it is *portable* rather
than sample-specific.

**Never used for:** re-opening Gate A; restoring `NS_BENCHMARK_ELIGIBLE`; creating an
asset-selection rule; reviving the dormant operator runner; routing or gating. `|ACF1|` stays
sample-specific until this test passes — and even then, only portability of the candidate is
upgraded, never a mechanism claim.

## 2. Data requirements (must all hold before the lock commit)

1. **One provider only.** No composite cohorts — a single source for the whole cohort.
2. **Complete daily OHLC, 2016-01-01 → 2025-12-31**, one row per trading day, per asset.
3. **Identical adjustment policy** across every asset in the cohort (documented; e.g. all raw
   closes, or all adjusted — not mixed).
4. **Provenance declared.** The discovery set came from Yahoo; if this cohort comes from another
   provider or market, that deviation is written next to every result (the reviewer's requirement).
5. **No source substitution after outcomes are seen.** If a provider fails mid-acquisition, the
   cohort is re-locked from scratch with a new commit — assets are never swapped in.

## 3. Universe and exclusion rules (mechanical; locked before inspection)

- **Target size:** 15–20 assets.
- **Universe:** every asset in a **pre-named** candidate list (the list is written at lock time and
  never edited afterwards).
- **No selection by `|ACF1|`, volatility or any other feature.** All eligible assets are included —
  the earlier design's "2 lowest + 2 highest" rule belongs to the *discovery* phase and is **not**
  reused here.
- **Mechanical exclusions only** (thresholds fixed at lock time, before data): history not covering
  the window; missing rows beyond a fixed fraction; non-positive or duplicated prices; duplicated
  dates. Every exclusion is reported by name with its reason.

## 4. Predictor (prospective — this is the point of the exercise)

- **Primary:** `|ACF1|`, computed from returns available **strictly before the decision point** (the
  validation cutoff), using the unchanged estimator (one-day returns, single lag, absolute value;
  window fixed at lock time). Features computed over the test segment are **not** usable here — that
  is exactly the contemporaneous-descriptive limitation recorded in `12b` §2.5.
- **Secondary:** realized volatility over the same pre-cutoff window. Reported separately.
- **The six rejected features** (trend persistence, skewness, kurtosis, jump ratio, sign
  persistence, relative volatility as a *predictor*) are **not rescanned**.
- A **leakage check line must be printed** by the analysis: every predictor value derives from
  indices `< cutoff`; any overlap invalidates the run.

## 5. Outcome (unchanged protocol)

The frozen interaction measure: DeReFusion vs `revin-DLinear`, `T=24`, `seq_len 96`, `label_len 48`,
70/10/20 chronological split, seeds 2021/2022/2023, paired bootstrap 4000 resamples; High = Q4+Q5,
Low = Q1+Q2 of the pre-cutoff-free relative volatility definition already frozen in E4/E5.

## 6. Analysis and decision rules (reused verbatim from the frozen set)

- Spearman ρ(`|ACF1|`, interaction) with an exploratory p-value, plus leave-one-asset-out.
- **Primary analysis runs on the new cohort only**; the old ten / pooled N=14 may be shown **only as
  labelled context**, never as the primary test.
- Reused operational criteria: sign reversal, `|ρ| < 0.30`, `p > 0.10`, LOO range crossing zero,
  single-asset dependence, any integrity finding.
- **Success** = the association replicates prospectively under those criteria.
- **Failure** = `|ACF1|` is closed **permanently** as this project's structural bridge, and the
  result is recorded as such.

## 7. Resource envelope

15–20 assets × 2 framework arms × 3 seeds = **90–120 training runs** ≈ **25–35 h** wall-clock at
2-way parallelism on this host, plus a restart budget (detached batches were killed silently three
times on 2026-09-13). Long campaigns must run under the idempotent self-healing launcher and be
verified by **process CPU time**, not by log output.

## 8. Lock table — **LOCKED 2026-09-14**

Delivered by geng as `T004` (`geng-lobster/T004-cohort/`, commit `78fd365`), independently validated
on intake by `reproduction/analysis/c1_cohort_validate.py` before this lock was written: 20/20 files
pass schema, date parsing, ascending order, duplicate check, coverage and SHA-256; provenance
attested as a single source (Yahoo chart API) in `geng-lobster/006-T004-receipt.md`.

| # | Asset | Provider | Rows | First date | Last date | SHA-256 |
|---|---|---|---|---|---|---|
| 1 | AAPL | Yahoo | 2514 | 2016-01-04 | 2025-12-31 | `2d0ec4a90463b9a6bae915a3038d871c061b3d18d813d6572d50d75f4b185916` |
| 2 | MSFT | Yahoo | 2514 | 2016-01-04 | 2025-12-31 | `48139f47b21a71466a60f7db045a20c85c3a75dcaba7afc20d175646f8b0d931` |
| 3 | AMZN | Yahoo | 2514 | 2016-01-04 | 2025-12-31 | `28f348100dc6d056b7bd70aa472ff29923d3776b3eaf9a1385254e08b01e6771` |
| 4 | META | Yahoo | 2514 | 2016-01-04 | 2025-12-31 | `fe0effbe8a541895f60e19c664e7f06791408fd46de31db0396f61b2590fa19d` |
| 5 | TSLA | Yahoo | 2514 | 2016-01-04 | 2025-12-31 | `4993de1e6a0997785a6c40f5397374dcbe9254b5f5bd3f0633edf92c951e5b9d` |
| 6 | JPM | Yahoo | 2514 | 2016-01-04 | 2025-12-31 | `450e3fec5302e7464e1a773e18a4d7f0132b58a926cac014f0e09e2083b1dc5e` |
| 7 | XOM | Yahoo | 2514 | 2016-01-04 | 2025-12-31 | `74ecd45a77d1edb9692cadc61b853a381de1b028f50a290df4057f37f9b9c928` |
| 8 | WMT | Yahoo | 2514 | 2016-01-04 | 2025-12-31 | `927d268a64b42742e6ae0b29c60a178beadf98506c0e6e1bc778c2e08f885481` |
| 9 | N225 | Yahoo | 2444 | 2016-01-04 | 2025-12-30 | `7250ac1596365ef894b61bb257789e23fe56a386a90e378901ce651fce4924a3` |
| 10 | GDAXI | Yahoo | 2537 | 2016-01-04 | 2025-12-30 | `6c8cc51c65fa002dc3129fd12b2278c05ae8b9a3e1d33ce3e3bcb076f57dbee5` |
| 11 | HSI | Yahoo | 2459 | 2016-01-04 | 2025-12-31 | `2a04256095a96d541142a426d289c5f173946b287ceefbd0ee2887c10cc25cec` |
| 12 | FTSE | Yahoo | 2525 | 2016-01-04 | 2025-12-31 | `7c69f5dd2b1d6e8f78c228cd607646c1289a688e83e3c6eeef5735e2f276d13e` |
| 13 | RUT | Yahoo | 2514 | 2016-01-04 | 2025-12-31 | `27f21f002ef041815e4a1db4acb8c6a0b2492832b43dfd2e79e0ad016bf1f718` |
| 14 | GBPUSD | Yahoo | 2602 | 2016-01-01 | 2025-12-31 | `90e00d04d481138da937e5316ac8274728d622b30a117e755e33e9057c833c7d` |
| 15 | AUDUSD | Yahoo | 2602 | 2016-01-01 | 2025-12-31 | `cd12448f0624128676d89c3bc7c0d6a12fc34a8b37309e689834804fa15d2a9a` |
| 16 | USDCAD | Yahoo | 2602 | 2016-01-01 | 2025-12-31 | `4f346a1849e213ec83446c6d7cdcd450e4e13f15532ce9787cb26a05e09727ee` |
| 17 | GOLD | Yahoo | 2513 | 2016-01-04 | 2025-12-31 | `76e42c67bcf264a2ce36a12e162a5f61f8807bde3954823ed074e23b77f8f4e5` |
| 18 | WTI | Yahoo | 2514 | 2016-01-04 | 2025-12-31 | `0b7361bb11333d1f6fb889429d8e4869789f260eb6d5c457c1ecec6b936527f1` |
| 19 | GLD | Yahoo | 2514 | 2016-01-04 | 2025-12-31 | `0b77dfa686fce94a851cf19ad1f0f23a45028a85423a5a9f218594b12f69b231` |
| 20 | TLT | Yahoo | 2514 | 2016-01-04 | 2025-12-31 | `4da4b28f0adc0b50a763a4eba39f6e1f0893df85b9f3bac2769576cfb7671da1` |

The same table was emitted mechanically by the validator into `23_lock_table_draft.md` (kept for
the generation trail).

- **Lock commit:** the commit that introduces this section, message `lock: C1 cohort Yahoo 20 assets`.
- **Provider + adjustment policy:** Yahoo chart API (`query1`, fallback `query2`) — **samethe
  provider as the discovery set**; `Close` = unadjusted close; **no `Adj Close` column**; identical
  policy across all 20 files. Acquired through geng's 83JM system proxy (this host cannot reach
  Yahoo: 403 direct, TLS timeouts through its own broken proxy).
- **Cutoff (decision point):** the first index of each series' test split (last 20 %, i.e. the
  frozen 70/10/20 boundary). The predictor (`|ACF1|`) is computed **only** from closes before that
  boundary, and the analysis must print the leakage check line (all predictor indices `< cutoff`).
- **Exclusion thresholds applied:** none triggered — all 20 pre-named assets were delivered and
  validated; no asset was dropped.
- **Disclosure (must travel with every result):** the discovery set and this cohort share a provider
  (Yahoo) but not a market (this cohort excludes the original ten and the four A-share assets);
  rows were dropped only where Yahoo returned all-null OHLC rows (documented per file in the
  provider's `MANIFEST.csv`), never interpolated.

### Predictor frozen (2026-09-14, before any outcome exists)

`reproduction/analysis/c1_predictor.py` computes the primary predictor from **pre-cutoff windows
only**: a feature window `close[i:i+96]` is usable only if `i + 96 <= border1_test`, so the last
close index touched is `border1_test - 1`. Output `reproduction/results/c1_predictor.csv` (and `.md`)
carries, per asset, `num_train`, the cutoff index and date, the number of windows used, the last
close index used, a boolean leak check, and both the **primary** pre-cutoff medians
(`acf1_abs_pre`, `rv_pre`) and the **contrast-only** contemporaneous test-window medians
(`acf1_abs_test`, `rv_test`).

Result at freeze time: `leakage check: ALL PASS` (20/20). Pre-cutoff `acf1_abs_pre` spans 0.0470
(META) to 0.1257 (WMT) — a usable spread for the correlation. Note the two versions differ
materially for several assets (META 0.0470 vs 0.0769 contemporaneous; WMT 0.1257 vs 0.0566), which
is precisely why the prospective version, not the contemporaneous one, is the primary.

**The predictor is now frozen. It must not be recomputed with a different window, estimator or
cutoff after any outcome is seen.**

## 9. Explicit prohibitions

No running before data + lock + owner authorisation; no rule change after outcomes are seen; no
adding or dropping assets; no metric shopping; no reinterpreting a failure as success; no
outsourcing the analysis to a party that has seen the outcome first.

## 10. Companion documents (bind this pre-registration)

- `24_c1_precommitted_interpretation.md` — the two outcome branches and the authorisation boundary,
  **committed before the first C1 outcome existed**; it fixes what may and may not be said in each
  branch and states that C1 never overrides Gate A.
- `24a_c1_blind_recomputation_spec.md` — the pre-specified blind recomputation of the headline
  statistic by a party that has not seen the analysis, including the integrity rule that a
  discrepancy is a finding rather than something to tune away.
