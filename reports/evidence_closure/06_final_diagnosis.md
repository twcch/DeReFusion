# 06 · Final Diagnosis (Q1–Q5 and the PATH gate)

Per spec §9, §16, §17. Every answer is given as
**Evidence → Interpretation → Confidence → Remaining uncertainty**, and direction is always kept
separate from statistical support: a negative estimate whose CI contains zero is described as
*"directionally negative but statistically inconclusive"*, never as a significant effect.

---

## Q1 · Does nonlinear computation have real value?

**Evidence.** (a) Framework level: the additive (nonlinear-residual) fusion beats the linear base
on GSPC at T=24 by 11.1% (seed 2021) and 15.4% (seed 2022), by 2.4% on BTCUSD and 1.8% on ETHUSD.
(b) Capacity-controlled proxy: at width ≥ 64, ETHUSD prefers the nonlinear operator in **36/36**
(state, seed) observations; BTCUSD prefers the linear operator in **36/36 at every width**; GSPC
collapses from +0.0598 (width 23) to +0.0061 (width 128), i.e. near parity. (c) In calm regimes the
two operators are statistically indistinguishable in every asset examined.

**Interpretation.** The value of nonlinear computation is **conditional**, not general:
`value = f(asset, capacity, condition)`. There is no asset in which the nonlinear branch is
uniformly better in the sense of "always and everywhere".

**Confidence.** Moderate-to-high for "conditional"; high for "not universally superior".

**Remaining uncertainty.** Operator *form* (no structured nonlinear operator has been tested);
capacity-controlled preference is only available for three assets.

## Q2 · Does volatility explain the nonlinear advantage?

**Evidence.** Interaction effect (Δ_high50 − Δ_low50, relative volatility): GSPC **−0.01205
[−0.01545, −0.00876]** (seed 2021) and **−0.02248 [−0.02719, −0.01792]** (seed 2022) — both
supported and sign-consistent; ETHUSD **−0.00943 [−0.01881, +0.00033]** — directionally negative
but statistically inconclusive; BTCUSD **+0.01502 [+0.00615, +0.02378]** — supported and
**opposite in sign**. Across the ten assets the sign spans zero (6 negative, 4 positive, 5 supported).

**Interpretation.** Volatility is **not a universal operator-selection criterion**. Within GSPC it
is a stable, seed-robust moderator; across assets it is inconsistent.

**Confidence.** High for "not universal"; moderate for "a real but asset-dependent relation".

**Remaining uncertainty.** Which asset property drives the sign difference: the two candidate
features are `|ACF1|` and realized volatility, both exploratory.

## Q3 · Is there a sample-level routing opportunity?

**Evidence.** 12 pre-registered structural states, capacity-matched operators, three seeds.
GSPC and BTCUSD: the linear operator is better in **every** one of the twelve states
(GSPC ΔMSE +0.047…+0.081, significant in all seeds; BTCUSD +0.129…+0.448). ETHUSD: between-state
variation of both signs at width 23, but at width ≥ 64 the preference reverses **for all twelve
states at once** — asset-wide, not state-specific. Across-state spread **shrinks** with capacity;
per-seed direction consistency **degrades** with capacity (31/36 → 29/36 → 28/36).

**Interpretation.** `NO_ROBUST_SAMPLE_LEVEL_ROUTING_EVIDENCE`. The apparent state-dependence seen at
small capacity is a capacity artefact, not a routing signal.

**Confidence.** High.

**Remaining uncertainty.** Whether a *different* structural representation (not volatility- or
autocorrelation-derived) could expose state-dependence. Untested and deliberately not pursued:
re-specifying the features now would be post-hoc.

**Consequence.** The adaptive-gating / router direction stays **paused**.

## Q4 · Is there asset-level operator heterogeneity?

**Evidence.** Under capacity control: ETHUSD → nonlinear (36/36 observations at widths 64 and
128), BTCUSD → linear (36/36 at every width), GSPC → near parity at width 128 (+0.0061). At the
interaction level the sign differs by asset and is supported in both directions (GSPC/EURUSD
negative; BTCUSD/BABA/NVO positive).

**Interpretation.** **Evidence supports asset-level operator heterogeneity**, i.e. the operator
choice is an asset-level configuration question rather than a per-sample decision.

**Confidence.** Moderate-to-high.

**Remaining uncertainty.** *Operator form* is unresolved; *sample-level routing* remains
unproven; the *structural explanation* is exploratory only; capacity-controlled evidence exists
for three assets only.

## Q5 · Does capacity affect the conclusions? (**mandatory**)

**Evidence.** 23 → 64 → 128: ETHUSD +0.0145 → −0.0281 → −0.0321 (12/36 → 36/36 → 36/36 preferring
nonlinear); GSPC +0.0598 → +0.0250 → +0.0061; BTCUSD +0.1783 → +0.1888 → +0.2998 (never nonlinear).
25 sign reversals, 23 of them ETHUSD; per-seed consistency falls with capacity.

**Interpretation.** Capacity **materially** affects the observed preference. The conservative and
permitted formulation is:

> *capacity sensitivity is a major uncertainty / confounder in the present evidence.*

**Explicitly not claimed:** that the earlier linear-favoured results are a *representation*
bottleneck. The evidence cannot distinguish "the nonlinear operator was under-capacity" from
"the nonlinear operator is intrinsically less suitable", and it does not do so for GSPC/BTCUSD.

**Confidence.** High that capacity matters; low for any specific mechanism.

**Remaining uncertainty.** Whether the capacity effect saturates, and whether it holds for
structured (rather than generic MLP) nonlinear operators.

---

## PATH gate (spec §16, §17)

**PATH 1 — candidate structural regularity.**

| # | PATH-1 condition | Verdict | Evidence |
|---|---|---|---|
| 1 | multi-asset consistent direction | ✅ | 6 negative / 4 positive; dominant side ≥ 3 assets |
| 2 | ≥1 structural feature with a stable exploratory association | ✅ | `|ACF1|` ρ=+0.733, p=0.016; realized vol ρ=+0.661, p=0.038 |
| 3 | not single-asset driven | ✅ | LOO ranges [+0.633,+0.817] / [+0.533,+0.900]; GSPC-dropped ρ=+0.600/+0.817 |
| 4 | direction survives LOO | ✅ | no sign flips for either candidate |
| 5 | not an artefact of GSPC seed merging | ✅ | s2021-only +0.685/+0.758; s2022-only +0.661/+0.733 → `GSPC_SEED_SENSITIVE = NO` |
| 6 | per-seed direction cross-check free of fatal contradiction | ✅ | GSPC sign-consistent across seeds; both seeds supported |

**All six conditions are satisfied → PATH 1 is upheld.** It remains a **candidate** regularity:
N = 10 assets, one seed per asset (GSPC two), sign spanning zero, and — per spec §6 — the
association is exploratory and must not be described as causal.

**PATH 2 and PATH 3 are not reached** (the structural association did not vanish, and the evidence
is not merely insufficient). They remain the fall-back verdicts if the candidate association fails
under additional seeds.
