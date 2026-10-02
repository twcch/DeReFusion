# 14 · Structural Validation — Gate A Result (N = 14)

**Date:** 2026-09-13, ~20:00 (Asia/Shanghai)
**Status:** FINAL — the decisive number now exists; no rule was changed to produce this file.
**Repository head at time of writing:** `2181089` (the N=14 result was produced after this head,
by `reproduction/analysis/sv_gate_a.py`, whose thresholds were frozen in `12a` / `12b`).
**Verdict:**

```
Gate A: FAIL
```

**Nothing was recomputed, re-thresholded or re-labelled to reach this verdict.** Every number below
is the verbatim output of `reproduction/analysis/sv_gate_a.py`, whose thresholds were locked in
`12a_gate_a_decision_rule_addendum.md` (17:32) and left untouched by the two external reviews
(`13`, `13b` → `12b`). `PASS` was already unreachable before the number existed (`12a` §3); the
remaining verdict is decided by the frozen FAIL conditions.

## 0. Provenance of the new data used here

| Item | Value |
|---|---|
| Batch | `reproduction/batches/run_structural_validation.ps1` (self-healing guard `run_sv_batch_guard.ps1`) |
| Batch window | started 18:48:11, final pair ended 19:55:53, stratifications 19:55:53–19:55:59 |
| Framework runs | 8 = 4 new assets × {`DeReFusion`, `revin-DLinear`}, `T=24`, `seed 2021`, `--no_use_gpu`, flags identical to the existing ten |
| Result dirs | 8/8 present under `results/` (`_seed2021_0`) |
| Stratifications | 4/4 present: `volatility_stratification_{BYD,BOE,EASTMONEY,YANGHE}_relative_s2021.{json,txt}` |
| Analysis | `.venv\Scripts\python.exe -W ignore reproduction/analysis/sv_gate_a.py` |
| Per-run logs | `sweep_runs/sv_<ASSET>_<MODEL>.log` (raw, kept) |

## 1. The N = 14 asset-level table

`interaction` = Δ_interaction (ΔMSE_high50 − ΔMSE_low50) at `T=24`; `acf1_abs` = |ACF1|;
`realized_vol` = test-window realized volatility. Definitions are unchanged from the N=10 analysis.

| asset | set | interaction | \|ACF1\| | realized_vol | significant |
|---|---|---|---|---|---|
| GSPC | existing | −0.01726 | 0.07598 | 0.00799 | True |
| BTCUSD | existing | +0.01502 | 0.08446 | 0.02542 | True |
| ETHUSD | existing | −0.00943 | 0.05335 | 0.03583 | False |
| USDJPY | existing | −0.01218 | 0.07192 | 0.00615 | False |
| EURUSD | existing | −0.13710 | 0.06582 | 0.00422 | True |
| SOX | existing | +0.00439 | 0.11447 | 0.02109 | False |
| DJI | existing | −0.00835 | 0.07389 | 0.00749 | False |
| BABA | existing | +0.04045 | 0.08212 | 0.02546 | True |
| NVO | existing | +0.09758 | 0.15619 | 0.02204 | True |
| TM | existing | −0.00805 | 0.06884 | 0.01737 | False |
| BYD | **new** | −0.03221 | 0.03243 | 0.02327 | True |
| BOE | **new** | −0.00038 | 0.04083 | 0.01539 | False |
| EASTMONEY | **new** | −0.00300 | 0.12958 | 0.02363 | False |
| YANGHE | **new** | +0.00002 | 0.13072 | 0.01511 | False |

## 2. Associations and leave-one-asset-out

| Quantity | N = 14 (frozen run) | N = 10 (discovery) |
|---|---|---|
| ρ(\|ACF1\|, Δ_interaction) | **+0.688** (p = 0.007) | +0.733 (p = 0.016) |
| ρ(realized vol, Δ_interaction) | **+0.481** (p = 0.081) | +0.661 (p = 0.038) |

**LOO (drop one asset at a time):**

- ρ(\|ACF1\|) range **[+0.610, +0.786]**, `sign_flip = False`
- ρ(realized vol) range **[+0.352, +0.637]**, `sign_flip = False`
- most influential single drops for \|ACF1\|: **NVO, BYD, YANGHE**

So the Layer-A direction is preserved and no single asset drives it to a sign flip.

> **Reading constraint (12b §2.3, mandatory next to any ρ quote):** ten of these fourteen assets
> belong to the discovery set, so N = 14 is a **discovery-set augmentation**, not independent
> replication. A favourable ρ here cannot upgrade the relation from **CANDIDATE** to **SUPPORTED**,
> and the verdict is capped at CONDITIONAL *regardless of how clean the number looks*.

## 3. The five robustness views (all / exclude GSPC / LOO range / width 64 / width 128)

| # | View | Result |
|---|---|---|
| 1 | **All assets** | ρ(\|ACF1\|) = **+0.688** (p = 0.007); ρ(RV) = +0.481 (p = 0.081) |
| 2 | **Exclude GSPC** (the only 2-seed asset) | ρ(\|ACF1\|) = **+0.709** (p = 0.007, N = 13) |
| 3 | **LOO range** | \|ACF1\| **[+0.610, +0.786]**, no sign flip; RV [+0.352, +0.637], no sign flip |
| 4 | **Width 64** (mean ΔMSE per asset) | GSPC +0.02503 · BTCUSD +0.18884 · ETHUSD −0.02807 · BYD −0.06603 · BOE **+0.00869** · EASTMONEY **+0.00379** · YANGHE −0.00582 |
| 5 | **Width 128** (mean ΔMSE per asset) | GSPC +0.00606 · BTCUSD +0.29983 · ETHUSD −0.03205 · BYD −0.08467 · BOE **−0.00040** · EASTMONEY **−0.00485** · YANGHE −0.01265 |

**Capacity sign-contradiction check (width 64 vs 128):** contradictions found for
**BOE** (+0.00869 → −0.00040) and **EASTMONEY** (+0.00379 → −0.00485).

## 4. Per-new-asset comparison against the pre-registration (`11`, commit `669c80e`)

`pref` = operator preference implied by the sign of the mean ΔMSE at widths 64/128
(nonlinear if either width is negative, else linear); `dint` = sign of Δ_interaction.

| Asset | \|ACF1\| | pref obs | pref exp | pref | dint obs | dint exp | dint |
|---|---|---|---|---|---|---|---|
| BYD | 0.032 | nonlinear | nonlinear | MATCH | negative | negative | MATCH |
| BOE | 0.041 | nonlinear | nonlinear | MATCH | negative | negative | MATCH |
| EASTMONEY | 0.130 | nonlinear | linear | **MISMATCH** | negative | positive | **MISMATCH** |
| YANGHE | 0.131 | nonlinear | linear | **MISMATCH** | positive | positive | MATCH |

**Mechanical match count: preference 2/4, interaction direction 3/4.**

*Note on the count vs `12a`.* `12a` §3 recorded the interim Layer-B reading as *1 of 4* because BOE
was then classified "mixed/tie". The mechanical rule in `sv_gate_a.py` classifies BOE as
nonlinear-favoured at width 64 and (marginally) at width 128, so it now counts as a preference
match — while simultaneously being one of the two **capacity contradictions**. Both statements are
reported; no criterion was redefined to reconcile them.

## 5. Gate A verdict

**The frozen FAIL conditions were applied verbatim:**

| FAIL condition (`12a` §3) | State | Triggered? |
|---|---|---|
| 1. Layer A flips or collapses (ρ sign flip, ρ < 0.30, p > 0.10, LOO crosses zero) | ρ = +0.688, p = 0.007, LOO [+0.610, +0.786] | **no** |
| 2. Layer B systematic: ≥3 of 4 new assets contradict the pre-registered preference with per-seed-consistent signs at both widths | 2 of 4 contradict (EASTMONEY, YANGHE); BOE/EASTMONEY also capacity-contradict | **no** (needs ≥3) |
| 3. Capacity contradiction: mean ΔMSE sign differs between width 64 and 128 for any new asset | **BOE** (+ → −), **EASTMONEY** (+ → −) | **YES** |
| 4. Single-asset dependence (drop-one sign flip) | LOO no sign flip | no |
| 5. Integrity finding (protocol inconsistency / leakage / post-hoc selection) | none found in this run | no |

```
==============================
STRUCTURAL VALIDATION
==============================
New assets: 4 (BYD, BOE, EASTMONEY, YANGHE)
rho(|ACF1|, interaction): +0.688 (p=0.007)   [N=10: +0.733]
LOO: range [+0.610, +0.786] sign_flip=False
Capacity contradiction: YES
Pre-registered match: preference 2/4, interaction direction 3/4
Gate A: FAIL
Recommended next action: Treat the candidate asset-level structural regularity as not externally
validated; re-frame rho(|ACF1|) as sample-specific rather than proceed.
==============================
```

**PASS was unreachable** before the number existed (`12a` §3: pre-registered PASS condition (1)
requires the new assets' preference to be broadly consistent; even the most generous reading is
2/4 < 3/4). The verdict is therefore **FAIL**, decided by FAIL condition **3 (capacity
contradiction on BOE and EASTMONEY)**.

**Honest asymmetry to record.** The *primary* layer (A) did **not** fail: the interaction-level
association was preserved (+0.733 → +0.688, LOO-stable, p = 0.007), so the FAIL reason is *not*
"disappearance of the association". The FAIL is triggered by the pre-registered capacity-
contradiction condition, which lives in the *secondary*, derived-implication layer (B). That is
what the frozen rule mandates, and it is reported as such rather than softened: a preference that
flips with model width cannot support a structural operator-selection claim.

**Consequences (per the pre-registered rules):**

- `NS_BENCHMARK_ELIGIBLE` is **downgraded from CONDITIONAL**.
- ρ(\|ACF1\|) is **re-framed as sample-specific**, not a portable regularity.
- Ten of the fourteen assets are the discovery set (10/14 overlap), so this was never independent
  replication in any case.
- No NS / transport / diffusion / gating / router / MoE work was started, and none is authorised.

### 5.1 Required 12b annotations (applied to this report)

**(a) Label for the ≥3/4 rule.** `12a`'s ≥3/4 rule is a **pre-Layer-A adjudication rule, specified
after the secondary capacity-layer results were available** — *not* a fully pre-registered Layer-B
falsification threshold (`12a` was committed 17:32, after the capacity-layer results at 16:33).
What *was* genuinely pre-registered before any operator result: the asset selection rule, the
per-asset expected directions, and the PASS conditions (`11`, commit `669c80e`).

**(b) No upgrade, mandatory overlap statement.** This result does **not** move the relation from
**CANDIDATE** to **SUPPORTED**. Any quotation of ρ (here +0.688, or the discovery +0.733) must be
accompanied by the explicit statement that **10 of 14 assets overlap with the discovery set**.

**(c) Thresholds are operational criteria.** `ρ < 0.30`, `p > 0.10`, LOO-crossing-zero and ≥3/4 are
described as **operational gate criteria**, never as universal statistical significance standards.
(Known weakness retained, not acted on: at N ≈ 14, `p < 0.10` already implies a non-trivial ρ, so
the extra ρ > 0.30 condition carries limited independent information.)

**(d) Domain-shift stress-test cohort.** The four new assets are a **domain-shift stress-test
cohort**, *not* an independent validation cohort. Yahoo → Sohu and mixed-markets → A-shares changed
together, so provider, market domain and microstructure cannot be separated; and the four A-shares
share one market, calendar, macro environment and provider, so their cross-sectional independence
is weaker than "four independent assets".

**(e) Contemporaneous, not leakage.** The structural features and the interaction outcome are
measured **on the same held-out test segment** (verified in code, `12b` §2.5; e.g. BYD has 458 test
windows = `candidate_pool_features.n_samples`). The correct formulation — the reviewer's exact
sentences — is used instead of the word "leakage":

> **The current feature–outcome association is contemporaneous and descriptive, because both the
> structural features and the interaction outcome are measured on the same held-out test segment.
> It therefore cannot support an ex-ante operator-selection rule.**

and

$$
\text{descriptive association} \neq \text{predictive selection rule}
$$

**(f) Capacity wording.** The reviewer's sentences are adopted verbatim:

> **Operator preference is not invariant to nonlinear model capacity.**

> **The low-capacity conclusion does not survive increased nonlinear capacity on ETHUSD.**

The phrase **"representation bottleneck" is retired**; "capacity artefact" is used sparingly or not
at all, because width simultaneously changes representation power, optimization landscape,
regularization, variance and implicit bias. A mechanism claim would additionally require
training/validation error trajectories versus width, convergence evidence, multi-seed learning
curves, at least one other capacity axis, and a plateau.

**(g) Multiplicity.** The discovery stage **screened eight structural features** before \|ACF1\| and
realized volatility emerged, so the discovery p-values (0.016 / 0.038) are **not** multiplicity-clean
confirmatory p-values. No rescan of new features (Hurst, entropy, tail index, …) was performed. If
both features are carried forward as primary, a Holm correction is required; the cleaner design is a
**single primary predictor (\|ACF1\|)** with realized volatility as secondary, which is adopted for
any future cohort.

## 6. New-cohort-only descriptive check (N = 4)

Block emitted mechanically by `sv_gate_a.py`:

| rank | asset | \|ACF1\| | Δ_interaction |
|---|---|---|---|
| 1 | BYD | 0.0324 | −0.03221 |
| 2 | BOE | 0.0408 | −0.00038 |
| 3 | EASTMONEY | 0.1296 | −0.00300 |
| 4 | YANGHE | 0.1307 | +0.00002 |

Spearman ρ (new cohort only, N = 4) = **+0.800** (p = 0.200) — descriptive only, no confirmatory
claim at this sample size.

The held-out cohort does **not** reverse the direction (rank-increase in \|ACF1\| accompanies a
rank-increase in Δ_interaction), but its four points sit almost on zero except BYD, and the ±
differences are tiny: this view neither rescues nor condemns the pooled number — it shows the pooled
ρ is carried mostly by the discovery ten.

> **This descriptive N=4 check is not part of the frozen Gate-A decision rule.**

The verdict uses the pooled N = 14 analysis and the frozen rule, not this block. It is shown because
a clean pooled ρ must not be allowed to hide how the held-out cohort actually behaved (`12b` §6.2).

## 7. The two review rounds and our responses

| Round | Reviewer input (verbatim archive) | Our response |
|---|---|---|
| **Round 1** (received 2026-09-13 17:34) | `13_external_review.md` — accepted the Layer-A-primary / Layer-B-secondary split; held that even a favourable N=14 must not upgrade CANDIDATE→SUPPORTED; capped the best possible outcome at CONDITIONAL; demanded accurate dating of the ≥3/4 rule, operational-criteria framing, the 10/14 overlap statement, the domain-shift framing, the contemporaneous-descriptive limitation, capacity wording, and the multiplicity note | `12b` §1–§5: accepted in full. **No threshold changed**; labels/limitations corrected; all seven points carried into this report (§5.1 (a)–(g)) |
| **Round 2** (received 2026-09-13 17:36) | `13b_external_review_round2.md` — *"do not change any Gate threshold, FAIL condition or CONDITIONAL cap"*; two report-level annotations only: (1) call the same-segment relation *contemporaneous association*, not "leakage"; (2) add a **new-cohort-only (N=4)** descriptive block that does not enter the Gate | `12b` §6: both adopted as **report-level only**. Implemented here: §5.1(e) exact-sentence wording, and §6's mandatory N=4 block with the required disclaimer. The reviewer's own preferred formulation was also adopted for the gate script (`sv_gate_a.py` emits the N=4 block, and the disclaimer verbatim) |

**Enforcement check.** The thresholds in `sv_gate_a.py` were frozen in `12a` (17:32) and are
unchanged in this run: `rho_flip`, `flip`, `contradictions` and `match_dirn <= 1` → FAIL; the PASS
branch is unreachable per the pre-registered conditions. The script's output above is the verdict,
unedited.

## 8. Next step (single, non-expanding)

```
First obtain genuinely independent, prospective structural validation;
NS-inspired operators remain a downstream hypothesis, not the immediate next experiment.
```

No next-stage experiment is started by this report. If a future cohort is attempted, it must be
genuinely independent (same source, ≥15–20 assets spanning several \|ACF1\| values, ≥3 seeds per
asset, features computed strictly before the decision point, single primary predictor), because
N = 14 as constructed here cannot bear the weight of a portability claim — and, on the frozen rule,
it did not pass.
