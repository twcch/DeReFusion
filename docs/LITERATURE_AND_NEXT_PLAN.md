# Literature map and next research plan

**Cut-off:** 2026-09-18. Only primary papers, venue pages and first-party
research pages are used for technical claims below.

## 1. What the current evidence actually asks next

The repository does not need another unbounded architecture search. Gate A
closed feature-based routing, F1 showed real but imperfect asset-level operator
heterogeneity, and C1 is the final prospective test of the one remaining
structural candidate. The next stage should therefore separate three questions:

1. **forecasting quality:** which modern baseline is strongest under the exact
   financial protocol?
2. **robustness and cost:** are gains stable across assets, seeds, horizons and
   compute budgets?
3. **operator inductive bias:** only after (1) and (2), can an operator-style
   architecture beat an ordinary matched nonlinear model?

No result in this repository supports skipping directly to a physical PDE or
Navier–Stokes claim.

## 2. Relevant frontier

### 2.1 Multi-scale supervised forecasters

- [TimeMixer (ICLR 2024)](https://openreview.net/pdf?id=7oLshfEIC2) uses
  decomposable multiscale mixing with MLP blocks. It is a high-priority control
  because it tests multi-scale structure without attention or a physical story.
- [Pathformer (ICLR 2024)](https://openreview.net/forum?id=lJkOCMP2aW) uses
  adaptive multi-scale pathways. It is relevant as a strong routing-style
  baseline, but the project's negative routing evidence means it must be a
  benchmark, not a presumed mechanism.
- PatchTST, iTransformer, TimesNet, DLinear and RevIN remain essential reference
  families because they span patching, variate attention, 2-D temporal modeling,
  linear decomposition and distribution-shift normalization.

### 2.2 Time-series foundation models

- [TimesFM](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
  introduced a 200M-parameter decoder-only patched model pretrained on 100B
  time points; by 2026,
  [TimesFM-3](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
  reports native multivariate/covariate support, 330M parameters and more than
  one trillion pretraining points.
- [Moirai](https://openreview.net/pdf?id=Yd8eHMY1wz) trains universal forecasting
  transformers on the 27B-observation LOTSA archive.
- [Chronos](https://www.amazon.science/blog/adapting-language-model-architectures-for-time-series-forecasting/)
  tokenizes scaled values for language-model-style forecasting;
  [Chronos-2](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)
  adds multivariate, covariate-informed and in-context zero-shot forecasting.
- [Mamba4Cast](https://openreview.net/pdf?id=YBOQ5HnzI6) is a useful state-space
  counterpoint to Transformer TSFMs and produces the horizon in one pass.
- [GIFT-Eval](https://arxiv.org/abs/2410.10393) supplies a diverse, non-leaking
  benchmark design: 23 datasets, more than 144,000 series and 17 baseline
  families. Its central lesson for this project is evaluation breadth and
  contamination control, not copying its scale.
- A 2026 [accuracy–energy benchmark](https://proceedings.mlr.press/v309/guibert26a.html)
  reports strong dataset dependence in accuracy and architecture dependence in
  energy. Runtime, memory and energy should therefore be first-class metrics,
  not appendices to MSE.

### 2.3 Neural operators and the NS boundary

- [Fourier Neural Operator](https://openreview.net/pdf?id=c8P9NQVtmnO) learns
  mappings between function spaces for families of parametric PDEs.
- [DeepONet](https://www.nature.com/articles/s42256-021-00302-5) separates input
  function encoding from output-domain coordinates to approximate nonlinear
  operators.
- [PINO](https://arxiv.org/abs/2111.03794) combines data with residual constraints
  from a **known** PDE family.
- [Laplace Neural Operator](https://www.nature.com/articles/s42256-024-00844-4)
  targets ODE/PDE solution operators and transient dynamics through a
  pole–residue representation.

These papers do not justify saying that price formation is a fluid. They assume
or learn mappings over function spaces, often with spatial fields, initial or
boundary conditions, and known differential equations. The present OHLC panel
has none of those ingredients. A Fourier or Laplace layer may still be tested as
an **architectural inductive bias**, but "NS-inspired" must not be upgraded to
"physics-informed" unless a measurable state, invariant and falsifiable
differential constraint are supplied independently of the forecast outcome.

## 3. Recommended program

### Phase 0 — C1 closed and record frozen

Deliverables:

- independent input verification and sensitivity recomputation;
- a qualified comparison recording the omitted aggregation rule;
- immutable result report 26 applying report 24's wording;
- no feature replacement, rescue cohort or threshold change.

Stop condition: any hash, grid or outcome discrepancy remains an integrity
finding until resolved; it is never averaged away.

### Phase 1 — modern baseline refresh

Create a new pre-registration before running outcomes.

**Cohort:** use the fixed original 10 and C1 20 as separate reporting strata;
never pool them without displaying both.  
**Horizons:** 1, 7, 24 and 36 trading/calendar steps, declared per market.  
**Seeds:** at least 3 for trained models.  
**Models:** seasonal-naive and drift; DLinear+RevIN; DeReFusion; PatchTST;
iTransformer; TimesNet; TimeMixer; one state-space model; zero-shot Chronos-2,
TimesFM-3 and Moirai where licences and hardware permit.  
**Metrics:** MAE, MSE/RMSE, MASE or scaled MAE, directional accuracy as secondary,
coverage/calibration for probabilistic models, parameter count, wall time,
peak memory and energy proxy.  
**Statistics:** paired per-window loss differences, asset-cluster bootstrap,
Holm correction within each declared model family, and effect sizes with
confidence intervals. No ranking from a single pooled mean.

Primary question: does DeReFusion improve a simple linear baseline sufficiently
and consistently to justify its extra residual branch after modern controls are
included?

### Phase 2 — data and protocol hardening

Before architecture novelty:

1. produce source/provider/licence metadata for every active CSV;
2. define exchange calendars explicitly instead of applying business-day
   frequency to crypto;
3. decide and pre-register treatment of corporate actions, adjusted prices,
   negative prices and independently rounded FX OHLC fields;
4. add rolling-origin evaluation with multiple decision dates;
5. separate hyperparameter selection assets from final assets;
6. add deterministic CUDA settings where supported and record exact hardware;
7. use isolated environments/model cards for each TSFM and screen pretraining
   overlap or contamination.

### Phase 3 — optional operator-bias benchmark

This phase is optional and is **not** licensed by an NS mechanism claim.

Compare, under matched parameter count, training budget and input information:

- linear temporal map;
- MLP residual;
- ordinary dilated temporal convolution;
- compact state-space residual;
- FNO-style spectral residual;
- Laplace-style residual only if the implementation is stable.

Use the same RevIN wrapper and direct additive fusion. Do not add a learned
router. The operator candidate succeeds only if its advantage over both the
generic MLP and ordinary convolution is seed-consistent, survives compute
matching and rolling-origin evaluation, and is not carried by one asset.

Required failure wording: "the tested operator inductive bias did not improve
the matched forecasting benchmark." It must never be generalized to "NS is
false in finance."

### Phase 4 — paper route

Choose the paper only after Phase 1:

- **Route A, likely strongest:** a prospective, negative-results-aware study of
  decomposition/residual complexity, capacity and cross-asset heterogeneity.
- **Route B:** a contamination-aware financial TSFM benchmark with accuracy,
  calibration and energy reporting.
- **Route C, highest risk:** a matched operator-inductive-bias paper, but only
  if Phase 3 passes its pre-registered controls.

## 4. Explicit non-goals

- no post-hoc feature search after C1;
- no claim that higher volatility selects a nonlinear operator;
- no sample-level gate/router/MoE revival;
- no use of test loss during training or manual model selection;
- no pooled headline that hides asset, seed or horizon failures;
- no use of "physics-informed", "conservation" or "Navier–Stokes" without a
  specified, measured and independently defensible physical constraint.
