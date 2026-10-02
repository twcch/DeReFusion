# 27 · C1 rule provenance and blind-audit scope correction

**Date:** 2026-09-28
**Status:** audit-scope clarification; no outcome, cohort, metric, threshold, seed, or verdict rule is changed.
**Authority:** this note records the provenance of the already frozen C1 analysis program. It is not a retroactive pre-registration or an amendment to `23`.

## 1. Finding

The C1 decision rules were not absent from the repository. The frozen analyst program
`reproduction/analysis/c1_analysis.py` supplies the operational choices omitted from the
independent reviewer's 267-file Stage-1 manifest. The reviewer's stop was therefore correct for the
manifest it received, but the handoff was incomplete.

The exact program was committed as `d14cebaeed69779e94a2d1521acdb73df0d6e13e` at
2026-09-14 13:37:16 +08:00 and is byte-identical in the audit source commit
`ff03eafca5a5600648389df9f29ec85ccc4f5276`. Its SHA-256 is
`527d063afff6c02a0784672d0b68a3764a1e8724093fd39876f37c9418fd9b5d` (5,917 bytes).
The T005 panel start recorded in `25` is 13:36 on the same date, with roughly 70 minutes per run;
the repository's execution record identifies this as the pre-outcome analysis program. No C1
outcome is recorded at the time this script was committed.

## 2. Frozen primary analysis, verbatim in substance

The program fixes the following choices:

- Primary interaction column: `inter_high50_low50` (high-50 minus low-50 relative-volatility
  strata). `inter_q45_q12` is reported as a secondary/alternative column and is not substituted for
  the primary statistic.
- Per-asset interaction: arithmetic mean across seeds 2021, 2022, and 2023.
- Expected association direction: positive Spearman rho between pre-cutoff `acf1_abs_pre` and the
  per-asset primary interaction.
- Failure conditions, any one sufficient: sign differs from positive; `abs(rho) < 0.30`; two-sided
  exploratory `p > 0.10`; the leave-one-asset-out rho range includes zero; or dropping the most
  influential single asset changes rho by more than `0.5 * abs(full_sample_rho)`.
- The primary cohort is the 20-asset C1 cohort only. The discovery cohort and pooled N=14 are
  context only.

These rules are implemented in the hash-identified script above and were present in the pre-outcome
analysis program. This note makes that source explicit for auditability; it does not add a new
criterion after observing the result.

## 3. Independent-audit scope repair

The earlier blind handoff verified the 267 listed files and the cohort/predictor integrity, then
correctly stopped because the manifest did not include `c1_analysis.py`. It consequently did not
know which of the two interaction columns was primary or how to aggregate seeds. Its reported
sensitivity calculations are useful diagnostics, but they do not establish exact reproduction of
the frozen primary analysis.

For a scope-complete independent replay, add the exact frozen `c1_analysis.py` identified above to
the audit input set, while retaining the original 267-file manifest and all prior stop records. The
reviewer must recompute the primary interaction from raw arrays using `inter_high50_low50`, average
the three seeds per asset, apply the frozen rules above, and only then compare against the analyst's
published outputs. Record both manifests/hashes and preserve every discrepancy. Do not replace the
original Stage-1 manifest or describe the earlier partial audit as a complete exact replication.

The repository's report `26` records a FAIL on the frozen analyst analysis and notes that the
independent sensitivity results were directionally consistent, while exact analyst-versus-auditor
numerical equality was not established. This scope correction does not upgrade that limitation;
the complete independent replay remains outstanding.

## 4. Compute availability and authorization boundary

P4 remains available for this project; no expiry or loss of access is assumed. Completing this
audit requires no model training and therefore no P4 run. P4 may be scheduled for a separately
authorized experiment under its frozen protocol. Its availability alone does not clear any
experiment gate or resolve the separate Stage-A execution deviation.

## 5. Provenance checks

- Frozen analysis source commit: `d14cebaeed69779e94a2d1521acdb73df0d6e13e`.
- Blind-audit source commit: `ff03eafca5a5600648389df9f29ec85ccc4f5276`.
- `c1_analysis.py` content comparison between those commits: identical.
- Independent reviewer's latest recorded Stage-1 checks: 267/267 file hashes passed; 20/20 cohort
  CSV hashes matched the lock table; leakage checks passed for 20/20 assets; raw truth alignment
  maximum absolute difference was `4.76105e-07`.
- Outstanding: a scope-complete independent replay of the frozen primary statistic and a difference
  list against the analyst output.
