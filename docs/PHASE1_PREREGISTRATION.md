# Phase 1 pre-registration: when does complexity pay?

**Status:** frozen before any Phase 1 outcome is generated; implementation
amendment v1.1 recorded below.
**Date:** 2026-09-19.  
**Scope:** modern-baseline refresh and temporal robustness; this document does
not reopen Gate A, F1 or C1.

## 1. Question and claims

The primary question is whether DeReFusion improves the RevIN-wrapped DLinear
baseline sufficiently and consistently to justify its residual branch. The
secondary question is where DeReFusion sits relative to representative modern
controls under the same financial protocol and compute reporting.

The study may support only one of the following claims:

1. **consistent benefit:** DeReFusion has a negative paired loss difference
   relative to DLinear in both frozen asset strata, with a cluster-bootstrap
   95% interval excluding zero and without a cost increase that dominates the
   accuracy gain;
2. **conditional benefit:** the pooled interval or one stratum crosses zero,
   but pre-declared asset/horizon summaries show repeatable heterogeneity;
3. **no demonstrated benefit:** neither condition is met.

No outcome licenses an asset-level router, an `|ACF1|` rescue, or a physical/NS
mechanism claim.

## 2. Frozen data

The only confirmatory strata are those recorded in
`reproduction/results/dataset_registry.csv`:

- `original-10` (10 assets);
- `c1-20` (20 assets).

They are always reported separately. Data files and their SHA-256 values must
match the registry before a manifest is accepted. The following supporting
assets are calibration-only and can never enter a Phase 1 result:
`BOND10Y`, `BONDETF`, and `CITICSEC`.

Known data defects remain visible rather than silently repaired: five FX feeds
have OHLC-envelope violations, and WTI contains non-positive April 2020 prices.
Models receive the frozen OHLC rows; metrics requiring logs must mark undefined
windows instead of altering prices.

## 3. Models and stages

### Stage A — execution calibration (15 fits; no scientific results)

Run `revin-DLinear`, `DeReFusion`, `revin-PatchTST`, `revin-iTransformer`, and
`revin-TimesNet` on the three calibration assets, horizon 24, seed 2021. This
stage checks imports, tensor shapes, output retention, timing, and manifest
generation. Its losses are not compared or published as evidence.

### Stage B — modern-baseline screen (300 fits)

Use the 30 frozen assets, the five available models above, horizons 1 and 24,
and seed 2021 under the legacy chronological split. This screen establishes
runtime and gross failures. It does not select assets, change the primary model
pair, or determine which failures to hide.

TimeMixer and a state-space model are declared **implementation gaps**, not
silent omissions. They may enter a separately versioned extension only if
their code, environment and hyperparameters are frozen before any extension
outcome is opened. Chronos, TimesFM and Moirai remain a separate zero-shot
track because training regime and contamination risk are not comparable.

### Stage C — primary confirmation (360 fits)

Compare only `revin-DLinear` and `DeReFusion` on all 30 assets, horizons 1 and
24, and seeds 2022, 2023 and 2024. The primary estimand is the paired difference

`delta = MSE_DeReFusion - MSE_DLinear`.

Negative values favour DeReFusion. Stage B seed 2021 is excluded from the
primary estimate.

### Stage D — temporal robustness (144 fits)

Use the pre-declared 12-asset sentinel panel:

- equity/index: AAPL, BABA, GSPC, N225;
- crypto/commodity: BTCUSD, ETHUSD, GOLD, WTI;
- FX/rates: AUDUSD, EURUSD, USDJPY, TLT.

Compare the primary pair at horizons 1 and 24, seed 2024, and test years 2023,
2024 and 2025. For test year `Y`, training ends on 31 December `Y-2`, validation
uses calendar year `Y-1`, and test uses calendar year `Y`. Each split may borrow
only the preceding `seq_len` rows as input context. Scaling is fit on training
rows only.

Stage D uses explicit, end-exclusive date boundaries. A fixed 70/10/20 split
must not be relabelled as rolling-origin evaluation.

## 4. Fixed training protocol

- task: long-term forecast;
- input/target: OHLC multivariate input, `Close` target;
- `seq_len=96`, `label_len=48`, `enc_in=4`, `dec_in=4`, `c_out=1`;
- `d_model=32`, moving-average kernel 25 where applicable;
- `n_heads=8`, two encoder layers, one decoder layer, `d_ff=2048`, dropout
  `0.1`, factor `1`, and `timeF` embeddings where applicable;
- Adam, learning rate `1e-4`, MSE loss, cosine schedule;
- maximum 30 epochs, patience 5, validation-selected checkpoint;
- test split is evaluated once after training;
- deterministic CUDA settings are enabled where supported and any unsupported
  operation is recorded;
- no model-specific hyperparameter search in Phase 1.

Stage A may begin only after the manifest checker passes; Stage B begins only
after all Stage A jobs produce the required artifact set. Stage D remains
downstream of Stages A--C even though its loader prerequisite is now met.

## 5. Outcomes and statistics

Primary:

- normalized test MSE paired by asset, horizon and seed;
- the Stage C DeReFusion-minus-DLinear difference, reported separately for
  `original-10` and `c1-20`;
- asset-cluster bootstrap confidence intervals with an immutable bootstrap
  seed and equal asset weighting.

Secondary:

- MAE and RMSE;
- directional accuracy, explicitly labelled secondary;
- per-horizon and per-asset win proportions;
- parameter count, training wall time, inference milliseconds per sample and
  peak memory;
- accuracy-cost Pareto membership.

Holm correction is applied within the declared modern-model family for Stage B
pairwise comparisons. No single pooled mean may hide the two strata. No asset
may be removed because its sign is inconvenient.

## 6. Artifact contract

Every fit must retain:

- exact command and Git commit;
- data SHA-256 and environment summary;
- `pred.npy`, `true.npy`, metrics and checkpoint-selection log;
- parameter count, wall time, inference time and memory;
- a per-file SHA-256 inventory.

Calibration commands must use a Phase-1-specific `--result_log`; they may not
append to the legacy global result ledger.

A batch is incomplete if any headline row lacks its raw arrays. Reruns use a
new attempt identifier; they never overwrite the failed attempt.

## 7. Stop and failure rules

- Import or shape failure pauses the affected model family; it does not permit
  replacement after results are inspected.
- Data hash mismatch blocks the asset.
- More than 5% unexplained run failures blocks the stage-level headline.
- Stage C is reported even when it fails.
- Stage D is descriptive robustness, not a route for reversing Stage C.
- No feature search, router, gate or MoE work begins from Phase 1 outcomes.

## 8. Authorized next action

The immediate authorized action is manifest generation and Stage A calibration.
No Stage B–D training is authorized until Stage A artifacts and the rolling
split implementation gap are reviewed in the repository.

## 9. Implementation amendment v1.1 (2026-09-19; before outcomes)

The loader now accepts `--split_mode dates` with end-exclusive `--train_end`,
`--val_end` and `--test_end`. Validation and test partitions borrow exactly the
preceding `seq_len` rows as input context, while every forecast label stays
inside its declared year. Scaling remains fitted on training rows only, and
the three boundaries are embedded in the experiment identifier to prevent
cross-origin artifact overwrite.

This amendment resolves an engineering prerequisite only. It changes no
asset, model, horizon, seed, estimand, success rule or stage ordering. Unit
tests cover leakage boundaries, training-only scaling, invalid inputs and
legacy-ratio compatibility; `audit_phase1_date_splits.py` checks all 72 frozen
asset/year/horizon combinations before Stage D may be considered runnable.
The machine-readable configuration also makes the unchanged harness defaults
for heads, layers, feed-forward width, dropout, attention factor and time
embedding explicit, and Phase 1 commands request deterministic PyTorch
algorithms.
