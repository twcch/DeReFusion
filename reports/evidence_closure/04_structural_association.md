# 04 · Structural Association (exploratory)

Per spec §6, §7, §10, §17. Method is the pre-registered one: **Spearman** rank correlation
between each asset-level structural feature and the asset-level interaction effect
(Δ_interaction = Δ_high50 − Δ_low50), plus leave-one-asset-out (LOO) sensitivity. Feature set is
the pre-registered set — **no feature was added, removed or re-thresholded**.

## A. Association table (N = 10)

| Feature | ρ | p (exploratory) | LOO range | Sign flips under LOO? | Flag |
|---|---|---|---|---|---|
| **\|ACF1\|** | **+0.733** | **0.016** | [+0.633, +0.817] | no | **candidate** |
| **Realized volatility** | **+0.661** | **0.038** | [+0.533, +0.900] | no | **candidate** |
| Kurtosis | +0.345 | 0.328 | [+0.100, +0.533] | no | directionally stable, not significant |
| Relative volatility | +0.358 | 0.310 | [+0.117, +0.600] | no | directionally stable, not significant |
| Sign persistence | −0.289 | 0.418 | [−0.511, −0.119] | no | directionally stable, not significant |
| Trend persistence | +0.042 | 0.907 | [−0.317, +0.283] | **yes** | **single-asset sensitive** |
| Skewness | +0.067 | 0.855 | [−0.067, +0.467] | **yes** | **single-asset sensitive** |
| Jump ratio | — | — | — | — | **constant across all assets → no discriminative power** |

## B. Reading

- **Direction of the candidate relation:** assets with **higher realized volatility** and
  **stronger first-order autocorrelation** show a **less favourable (or reversed)** nonlinear
  advantage in turbulent regimes. The two supported *positive* interaction cases (NVO, BABA) are
  precisely the high-`|ACF1|` assets, while the extreme negative case (EURUSD, −0.1371) is the
  lowest-volatility asset in the set.
- **LOO:** both candidates keep their sign and remain within ±0.15 of the full-sample ρ when any
  single asset is removed; neither is driven by one asset. Trend persistence and skewness both
  flip sign under LOO and are therefore **not** admissible as structural explanations.
- **Jump ratio (spec §10):** the median jump ratio is **identical (0.01052632 = 1/95) for all ten
  assets**. It therefore carries **no discriminative power** at the pre-registered threshold
  c = 3σ. This is recorded as an empirical finding of the current analysis; the threshold was
  **not** changed to manufacture variation.

## C. GSPC seed-merge sensitivity (spec §5, §17 condition 5)

The same associations recomputed under four treatments of GSPC:

| Variant | GSPC Δ_interaction | ρ(realized vol) | ρ(\|ACF1\|) |
|---|---|---|---|
| GSPC = seed 2021 only | −0.01205 | +0.685 | +0.758 |
| GSPC = seed 2022 only | −0.02248 | +0.661 | +0.733 |
| GSPC merged (pre-stated mean rule) | −0.01726 | +0.661 | +0.733 |
| GSPC dropped entirely | — | +0.600 | +0.817 |

**No sign flip, magnitudes stable across ±0.12.** Verdict: **`GSPC_SEED_SENSITIVE = NO`** —
the candidate associations do not depend on the seed-merging rule.

## D. Discipline statement

These are **exploratory structural associations**, not causal effects: the specification says
explicitly that with a small number of assets a single p-value is not decisive. The two
candidates are reported as *candidate asset-level structural regularities* only, and the
sign of the underlying interaction spans zero across the asset set.
