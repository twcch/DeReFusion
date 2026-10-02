# 29 · C1 independent blind replay — final

**Date:** 2026-09-28

**Status:** complete; independent reviewer found no material numerical discrepancy.

**Decision:** **FAIL / did not replicate**. This confirms, but does not broaden, the frozen C1 decision.

## Integrity and blindness

The independent reviewer worked from the existing Stage-1 package and did not
open analyst outcome files until its raw replay and verdict were saved and
hashed. All **267/267** files in the original manifest passed byte-length and
SHA-256 verification; the manifest SHA-256 is
`112dfe0c75bc7215266c0ffc3ea75c47d2bbd079bad72c95a9a2e6cfce85276d`.
Earlier partial-stop records were preserved.

The frozen analysis program is identified by its Git path at commit
`d14cebaeed69779e94a2d1521acdb73df0d6e13e`. Its LF Git-blob SHA-256 is
`612f1ee21f90bc08bb8ac65156290874423370d365c04f540144a28ff742e830` (5,780
bytes). The Windows CRLF worktree representation is SHA-256
`527d063afff6c02a0784672d0b68a3764a1e8724093fd39876f37c9418fd9b5d` (5,917
bytes); LF normalization is byte-identical to the Git blob. The initial hash
stop was therefore a line-ending representation difference, not changed code
or rules.

## Independent result

The reviewer recomputed the predictor and interaction from the 20 locked CSVs
and 120 pairs of raw prediction/truth arrays, using the pre-cutoff predictor
and frozen relative-volatility stratification definitions.

- Predictor leakage: **20/20 pass**; every final source index is before the
  forecast cutoff.
- Maximum CSV-to-`true.npy` alignment error: `4.761048888468622e-07`.
- Paired `true.npy` arrays between model arms: exact equality.
- Primary interaction: relative-RV `high50 − low50`, averaged across seeds
  2021, 2022, and 2023.
- Spearman `ρ = −0.1519`; LOO range `[-0.3439, -0.0702]`; most influential
  asset META, maximum `|Δρ| = 0.1920`. The LOO range does not cross zero.
- Frozen failure conditions triggered: sign reversal, `|ρ| < 0.30`,
  exploratory `p > 0.10`, and the single-asset influence criterion.

The blind replay agrees with the analyst's **FAIL / did not replicate** verdict.
The result does not overwrite Gate A, change F1's narrow breadth conclusion,
or establish anything about Navier–Stokes dynamics.

## Comparison with analyst outputs

- `c1_predictor.csv`: **20/20 rows**, all published columns match.
- `c1_interactions.csv`: **60/60 rows**, all 13 published fields match within
  `1e-12`; the largest floating-point residual is below `1e-16`.
- No analyst-versus-reviewer numerical discrepancies were found in the
  predictor or interaction tables.
- `low_mse_delta` and `high_mse_delta` are the Q1+Q2 and Q4+Q5 arm deltas;
  their difference equals `inter_q45_q12`. They are not the primary 50/50
  stratum deltas.

## p-value implementation difference

The frozen program uses `scipy.stats.t.sf` when SciPy is available, producing
`p = 0.5226819111403139` (published as `0.5227`). The independent runtime lacked
SciPy and used the intended normal-approximation fallback, producing
`p = 0.5144436936916705` (published as `0.5144`). That fallback references
`np.math`, which NumPy 2 no longer provides; the reviewer used a runtime-only
compatibility injection without editing the frozen source. This environment
path difference does not change rho or the verdict; both p-values exceed the
frozen `0.10` threshold.

## Preserved audit artifact

The independent reviewer retained its full read-only report and frozen outputs
in its separate audit workspace. Report SHA-256:
`0a0c669ae55257b24d711a0469384908f7fa730ebf2231ea9ff937081f367b58`.
No source repository files were modified or pushed by that reviewer. This
repository report records its final findings while retaining the older stop
history in the independent workspace.
