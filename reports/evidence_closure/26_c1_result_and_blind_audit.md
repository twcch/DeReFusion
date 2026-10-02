# 26 · C1 result and independent blind audit

**Date:** 2026-09-18  
**Execution:** complete, 120/120 runs (20 assets × 2 arms × 3 seeds).  
**Scientific decision:** **FAIL** — the prospective `|ACF1|` association did not replicate.

## Result

The pre-outcome analyst program (`c1_analysis.py`) uses the mean across three
seeds of the frozen high-50 minus low-50 interaction. On the new cohort:

- Spearman rho = **-0.1519**, exploratory p = **0.5227**;
- leave-one-asset-out range = **[-0.3439, -0.0702]**;
- largest influence = **META**, max absolute change in rho = **0.1920**;
- secondary realised-volatility rho = **-0.3263**, p = **0.1603**.

The expected sign reversed, `|rho| < 0.30`, `p > 0.10`, and the frozen
single-asset influence rule triggered. Per reports 23 and 24, `|ACF1|` is
retired permanently as this project's structural bridge. Gate A is not
overwritten, F1 remains a narrow heterogeneity result, and no replacement
feature may be searched to rescue C1.

## Independent check

An independent Codex task, isolated from analyst-only outputs, verified all
267 allowed files byte-for-byte, all 20 CSV lock hashes, 240 raw arrays and the
predictor leakage boundary. CSV-to-`true.npy` alignment error was at most
`4.76105e-07`; all 20 predictor windows ended before the cutoff.

The handoff did not whitelist the already pre-outcome `c1_analysis.py`, so the
reviewer correctly declined to choose the interaction variant and cross-seed
aggregation. Its sensitivity analysis nevertheless found `|rho| < 0.30` and
`p > 0.10` for the three-seed mean, median and every individual seed. Thus the
**failure decision is robust**, but exact analyst-versus-independent numerical
equality is not claimed.

## Integrity findings retained

The blind handoff exposed: an initially wrong full commit SHA; a manifest hash
of pre-commit rather than archived bytes; newline conversion of locked CSVs;
an omitted predictor dependency; and a final omission of the pre-outcome
aggregation program. Each stop remains in history. None changes the arrays,
cohort, predictor, thresholds or robust failure conclusion, but the handoff
must not be described as flawless or exactly reproduced.

## Consequence

- do not revive `|ACF1|`, feature routing, a gate or MoE;
- do not interpret C1 as evidence for or against Navier–Stokes dynamics;
- continue, if desired, with the modern-baseline and rolling-origin programme
  in `docs/LITERATURE_AND_NEXT_PLAN.md`.
