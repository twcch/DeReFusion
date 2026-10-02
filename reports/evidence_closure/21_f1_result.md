# 21 · F1 result — capacity-controlled breadth panel (verdict: **SUCCESS**, with a large caveat)

**Run:** 2026-09-14 12:38–12:47 · **Script:** `reproduction/analysis/structure_routing_capacity_check.py`
`--assets USDJPY,EURUSD,SOX,DJI,BABA,NVO,TM --widths 23,64,128 --suffix _f1`
**Pre-registration:** `20_f1_preregistration.md` (locked at `bc6ae5c`, before execution)
**Judged strictly by the rules locked there. No rule, threshold, seed or definition was changed.**

## 1. Raw scale

`7 assets × 3 widths × 3 seeds = 63 paired settings` → **126 individual arm fits** (a linear and an
MLP fit in every paired setting) → **792 state-level rows**.

*Counting note (corrected 2026-09-14 after external review): the earlier wording called the 63
paired settings "operator fits", which collided with `16`'s count of **126** arm fits. Both numbers
describe the same run: 63 pairs, 126 individual arm fits, 792 state-level rows.*
Nothing was overwritten: outputs are `operator-regime-capacity_f1.csv` (the E4 file and the `_sv`
file are untouched).

## 2. Per-asset results (mean ΔMSE over states; negative = nonlinear-favoured)

| Asset | w23 | w64 | w128 | per-seed means (2021/2022/2023) @w64 | @w128 | seed-stable @64 / @128 |
|---|---|---|---|---|---|---|
| **TM** | +0.0812 | **−0.0156** | **−0.0466** | −0.0058 / −0.0242 / −0.0168 | −0.0352 / −0.0658 / −0.0389 | **yes / yes** |
| **DJI** | +0.0348 | **+0.0280** | **+0.0288** | +0.0192 / +0.0448 / +0.0200 | +0.0465 / +0.0056 / +0.0342 | **yes / yes** |
| **EURUSD** | +0.0785 | **+0.1175** | **+0.1347** | +0.1236 / +0.1029 / +0.1260 | +0.1521 / +0.1092 / +0.1429 | **yes / yes** |
| **USDJPY** | +0.3466 | **+0.1116** | **+0.1163** | +0.1213 / +0.1088 / +0.1046 | +0.1438 / +0.1022 / +0.1028 | **yes / yes** |
| NVO | +0.7330 | +0.6152 | +0.2244 | +1.0092 / +0.4272 / +0.4093 | +0.8153 / **−0.0360** / **−0.1062** | yes / **no** |
| SOX | +0.1085 | +0.0184 | **−0.0137** | +0.0320 / +0.0401 / **−0.0170** | +0.0049 / −0.0069 / −0.0389 | **no / no** |
| BABA | +0.0023 | +0.0005 | +0.0032 | −0.0067 / −0.0039 / +0.0119 | −0.0067 / +0.0030 / +0.0134 | **no / no** |

## 3. Application of the locked rules

- **Stable asset** = seed-stable at **both** width 64 and width 128 **with the same sign**.
  → **stable: TM (nonlinear), DJI (linear), EURUSD (linear), USDJPY (linear) = 4 of 7.**
  → **not stable: BABA** (never seed-stable), **NVO** (seed-unstable at 128), **SOX** (sign flips
  between widths and is seed-unstable at both).

- **Locked success criterion:** ≥ 2 additional stable assets **and** ≥ 1 **new** opposite-sign pair
  beyond ETH/BTC.
  → additional stable assets: **4** ✅ · new opposite-sign pairs: **TM×DJI, TM×EURUSD, TM×USDJPY = 3** ✅
  → **F1 = SUCCESS.**

- Reference (pre-existing three, unchanged): ETHUSD nonlinear and seed-stable at 64/128; BTCUSD
  linear and seed-stable at all widths; GSPC near parity (+0.0061 at 128).

## 4. What this licenses — and what it does not

**Licensed** (the only upgrade `16` §3 permits): *asset-level operator heterogeneity is replicated
on the existing source.* Preference is not a single-asset artefact: four further assets carry
seed- and width-stable preferences, in **both** directions (TM nonlinear; DJI/EURUSD/USDJPY linear).

**The caveat, stated as prominently as the success:** **3 of 7 additional assets fail the strict
criterion** (BABA, NVO, SOX). So preference is a **tendency with a large unstable minority**, not a
per-asset invariant. Two of the three failures are themselves capacity/seed phenomena (NVO moves
toward nonlinear as capacity grows; SOX flips sign between widths) — the same fragility that
produced Gate A's `FAIL`.

**A further honest observation** (from the run's own summary): per-seed direction consistency across
asset×state pairs degrades with capacity — **75/88 at w23 → 68/88 at w64 → 61/88 at w128**. Higher
capacity does not buy stability; it costs it.

**Not licensed:** no feature-based explanation (no `|ACF1|` claim — that remains sample-specific
after Gate A); no re-opening of Gate A; no restoration of `NS_BENCHMARK_ELIGIBLE`; no authorisation
for the dormant runner on `draft/ns-benchmark-runner`; no routing/gating work.

## 5. Consequence for the roadmap

- `16` §3 branch F is **completed**: the basic question ("is asset-level preference reproducible?")
  is answered **yes, for a majority of the panel, with a substantial unstable minority**.
- The remaining licensed step is unchanged: an **independent, prospective, same-source cohort**
  (`16` §4, C1) — currently **blocked on data acquisition** (`15_*`, `19`).
- Structural-candidate work stays closed (Gate A = FAIL); NS-inspired operator work stays a
  downstream hypothesis.
- Raw artifacts retained: `reproduction/results/operator-regime-capacity_f1.csv`, the summary and
  sign-reversal files with the `_f1` suffix, `f1_run_log.txt`, and the analysis script
  `reproduction/analysis/f1_analysis.py`.
