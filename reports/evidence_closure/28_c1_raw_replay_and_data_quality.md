# 28 · C1 raw-artifact replay and data-quality note

**Date:** 2026-09-28  
**Status:** raw replay complete; separate blind review pending.  
**Scope:** read-only recomputation; no model training, protocol change, or verdict change.

## Result

The 120 C1 runs (20 assets × 2 arms × 3 seeds) were replayed from the Stage-1
raw artifacts using the frozen relative-volatility stratification procedure.
The 20-asset predictor was recomputed with the frozen pre-cutoff rule, then the
frozen `c1_analysis.py` analysis was applied to the recomputed interaction table.

- Verified all **267/267** files listed in `BLIND_STAGE1_MANIFEST.csv`; no hash
  mismatches or missing files.
- Recomputed predictor: **20/20** rows match the published predictor table,
  including the pre-cutoff values and all leakage checks.
- Recomputed interactions: **60/60** asset-seed rows match the published
  interaction table across its numeric and Boolean fields.
- Recomputed predictor and interaction CSVs are byte-identical to the published
  files: SHA-256 `80e158d40ee3fe85ddf9111514a4f6036c395ca5bfe84ea3e2a1c49632613737`
  and `ea36d39df14ace62253baeed0090314a8589922d5ef2f305b24af1dffb582d57`,
  respectively.
- The frozen analysis reproduces **ρ = −0.1519, p = 0.5227**, LOO range
  **[−0.3439, −0.0702]**, and META as the most influential asset
  (**max |Δρ| = 0.1920**). It triggers sign reversal, `|ρ| < 0.30`, `p > 0.10`,
  and the single-asset influence criterion. The C1 verdict remains **FAIL**;
  no Gate A or F1 conclusion is changed.

The replay used the Stage-1 source cohort at audit commit
`ff03eafca5a5600648389df9f29ec85ccc4f5276`, the stratification source recorded
with T005 execution commit `d17822f6a5040d0271e13ad3a192792d4085a635`, the
pre-cutoff predictor source from `d14cebaeed69779e94a2d1521acdb73df0d6e13e`,
and the frozen analysis program at the path in that Git commit. The Windows
working-tree copy has SHA-256
`527d063afff6c02a0784672d0b68a3764a1e8724093fd39876f37c9418fd9b5d` (5,917
bytes, CRLF); the raw Git blob has SHA-256
`612f1ee21f90bc08bb8ac65156290874423370d365c04f540144a28ff742e830` (5,780
bytes, LF). With `core.autocrlf=true`, normalizing the worktree line endings
produces a byte-for-byte match to the committed blob. The earlier review stop
on this hash mismatch was a line-ending representation mismatch, not a source
code or decision-rule discrepancy. The two source commits contain the same Git
blob. Recomputed tables and per-run stratification JSONs are retained in the
isolated audit workspace, not substituted for the original handoff or published
files.

## Blindness boundary

This replay was performed in the project-owner Codex task after the prior C1
result report had already been read. It is an exact raw-artifact re-execution,
but **not a blind third-party audit** and is not represented as one. The earlier
reviewer's partial-stop record remains valid for its original scope. A separate
review task has now been asked to perform the scope-complete replay without
seeing outcome files; its independent result is still pending.

## Data-quality finding: WTI logarithmic returns

The locked WTI series contains one non-positive Close observation (index 1078,
minimum −37.63). Since the frozen predictor uses `diff(log(Close))`, 96 of its
1,821 pre-cutoff windows contain an undefined logarithm. The implementation
silently excludes these windows through `nanmedian`; 1,725 windows remain finite
for both primary predictor components. The predictor values and frozen verdict
reproduce exactly, and excluding WTI as an exploratory sensitivity still gives
ρ = −0.1158, p = 0.6369 (N = 19), so this caveat does not rescue C1. Nevertheless,
log returns are not defined for negative futures prices. Future protocols must
pre-specify a price-change transform or exclude such instruments before seeing
outcomes; do not retroactively alter C1 or its frozen verdict.

## Audit interpretation

No numerical error or false result was found in the published C1 predictor,
interaction table, or headline analysis. The main correction is one of scope:
the exact numerical match is now established by this owner-side replay, while
the stronger independent-blind verification remains open until the separate
reviewer reports. The WTI transform issue is a methodological limitation, not
a reason to suppress or rewrite the historical result.
