# 02 · Experiment Integrity

Per spec §13. "Same protocol?" = were the compared arms run under an identical protocol;
"Raw output?" = does a raw artifact exist (and is it versioned); "Seed reproducible?" = is the
seed specified in the committed command; "Leakage checked?" = were future-information paths
audited; "Post-hoc selection?" = were assets/seeds/regimes dropped after seeing results.

| Experiment | Same protocol? | Raw output? | Seed reproducible? | Leakage checked? | Post-hoc selection? | Current verdict |
|---|---|---|---|---|---|---|
| E1 baseline reproduction | ✅ identical flags across arms | ⚠️ local only (git-ignored `results/`) | ✅ `--rand_seed 2021` | n/a (supervised, chronological split) | ✅ none | **OK — `RAW_NOT_VERSIONED` gap only** |
| E2 gate ablations | ✅ identical to E1 except `--model` | ⚠️ local only | ✅ | n/a | ✅ none | **OK — same gap** |
| E3 volatility stratification | ✅ same RV definition across assets, same bootstrap | ✅ JSON+TXT versioned | ✅ per-run tag | ✅ causal RV; **thresholds from train split only**; absolute mode flagged as time-confounded | ✅ all 10 assets reported, incl. inconclusive ones | **OK** |
| E4 capacity sensitivity | ✅ **only** hidden width varies (data/features/splits/optimizer/lr/batch/epochs/early-stop/seeds/eval identical) | ✅ CSV versioned | ✅ three seeds | ✅ states reuse the frozen definitions | ✅ none | **OK** |
| E5 structural-state proxy | ✅ both operators share optimizer, steps, batch, lr, early stop, preprocessing, seed | ✅ CSV + raw JSON versioned | ✅ three seeds | ✅ train-only medians for thresholds; k-means fitted on train only | ✅ no state dropped; all 12 reported | **OK** |
| E6 asset-level heterogeneity | ✅ frozen protocol across assets | ✅ summary CSV versioned | ⚠️ 1 seed/asset (GSPC 2 of 3) | ✅ causal RV; GSPC merged by a pre-stated rule | ✅ all assets reported incl. inconclusive | **OK — `SEED_COVERAGE_PARTIAL`** |
| E7 PATH gate | ✅ criteria implemented mechanically before the N=10 run | ✅ verdict in report | n/a | n/a | ✅ none — verdict left unchanged when SOX was added | **OK** |
| E8 seed-merge sensitivity | ✅ same association recomputed under four GSPC variants | ✅ this report | ✅ | n/a | ✅ none | **OK** |

## Explicit findings

1. **No leakage found.** All structural features and regimes are computed from the input window
   only; all thresholds/medians/standardisation parameters come from the training split; the
   stratification's volatility measure is strictly causal and the de-trended variant uses a
   trailing median `RV_{t-120:t-1}`.
2. **No post-hoc selection found.** No asset, seed or regime was removed after inspection.
   The one asset that is *small and inconclusive* (SOX, +0.0044, CI contains 0) was added and
   reported like the others; it did **not** change the verdict.
3. **Two documented incident repairs (not selection):** the `SOX/revin-DLinear` run stalled twice
   and was re-launched **with identical configuration** (silent deadlock caused by killing
   DataLoader worker processes / memory pressure). The reproducibility-relevant lesson is recorded
   in `docs/ROADMAP.md` and in the stage report appendix.
4. **Gaps declared, not hidden:** `RAW_NOT_VERSIONED` (training tensors) and
   `SEED_COVERAGE_PARTIAL` (1 seed/asset at the asset level). Neither affects the direction of any
   conclusion; both bound the confidence we attach to it.
