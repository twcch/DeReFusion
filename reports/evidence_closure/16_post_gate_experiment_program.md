# 16 · Post-Gate Experiment Program and NS-Inspired Feasibility Boundary

**Date:** 2026-09-13  
**Status:** planning document only — no experiment or model implementation is authorised here.  
**Authority:** this document governs work proposed *after* the frozen Gate A result in `14_*`.
It does not alter any result, threshold, definition, seed, metric, split, horizon, window, capacity,
or decision rule in `01`–`15`.

## 1. Current checkpoint

Gate A remains pending until the already-specified framework outputs are analysed and the `14_*`
report is written. `PASS` is unreachable and the live outcomes are `FAIL` or `CONDITIONAL`.
No new experiment should start before that report exists. Finishing Gate A is completion of an
existing frozen protocol, not a new model-development stage.

The evidence determines the order of work:

1. sample-level routing is not supported and stays closed;
2. asset-level operator heterogeneity is supported but capacity-controlled coverage is only three
   assets;
3. `|ACF1|` is only a contemporaneous, exploratory candidate explanation;
4. a temporal NS-inspired nonlinear operator has not been tested and is not yet eligible for
   implementation.

## 2. Rules shared by both verdict branches

- Preserve all existing definitions and the frozen Gate A rule. A future study gets a separate
  pre-registration; it may not rewrite the old one.
- Keep the 70/10/20 chronological split, `T=24`, `L=96`, seeds 2021/2022/2023, existing metrics,
  widths 23/64/128, twelve structural states, and all current feature and regime definitions.
- Fix the asset universe and exclusion rules before inspecting new operator outcomes. Report every
  eligible asset and seed, including failures and ties.
- Keep raw per-run predictions, targets, configurations, logs and summary tables. A summary without
  its raw inputs is incomplete.
- Treat `|ACF1|` as the sole primary structural predictor. Realized volatility is secondary and
  descriptive; do not rescan the other six rejected features.
- Do not implement adaptive gating, routing, MoE, attention, a pressure term, or an NS module. Do
  not place any temporal NS-inspired nonlinear operator inside DeReFusion during these branches.

## 3. Branch F — Gate A = FAIL

### Question

Is asset-level operator preference itself reproducible when capacity and seeds are controlled, or
was the three-asset result too narrow to be a stable property?

### Single experiment F1 — complete the existing-asset capacity panel

Run the already-defined linear-map versus MLP proxy on **all seven original Yahoo assets that lack
capacity-controlled results**. Do not select only `|ACF1|` extremes. Use all twelve states, all three
frozen widths and all three frozen seeds. This is an ordinary-operator stability experiment; it
does not test a structural explanation and does not test NS.

**Prerequisites**

1. `14_*` records `FAIL` under the unchanged rule.
2. The seven-asset list and file hashes are locked before execution.
3. A dry inspection confirms identical preprocessing, optimizer and stopping settings to E4/E5.
4. Storage naming is idempotent and cannot overwrite existing raw results.

**Resource envelope**

- 7 assets × 3 widths × 3 seeds × 2 operators = **126 operator fits**.
  *Correction (2026-09-13, see `17a_agent_check_notes.md` §2.1): the twelve structural states are
  an **evaluation stratification over a single fit per (asset, width, seed)**, not a training
  multiplier. The earlier figure of 1,512 over-counted by that factor; measured cost ≈2–3 s per fit
  makes F1 a **minutes-scale** job, not an hours-scale one.*
- Analysis cost is negligible relative to fitting; CPU-hours must be estimated from existing E4/E5
  logs before scheduling rather than guessed here.
- Parallelism changes wall time only; it may not change seeds, batch size or early stopping.

**Falsification / stop criteria**

- The claim that asset-level preference is reproducible fails if the completed panel contains no
  pair of assets with opposite mean ΔMSE signs that are each consistent across all three seeds and
  both widths 64 and 128.
- An individual asset is not considered stable if its mean ΔMSE sign differs between widths 64 and
  128 or across seeds.
- If the claim fails, stop structural-candidate and temporal NS-inspired nonlinear operator work.

**Note (2026-09-14, operator-approved — keeps the audit trail coherent):** the C1 cohort study
(`23`) was launched **after** this FAIL branch and **after** F1 completed. It is therefore an
**operator-approved post-F1 prospective closure test**, not a contradiction of this section: its
result **never overrides Gate A**. See `24_c1_precommitted_interpretation.md` §4 for the authorisation
record and §6 for the boundaries that hold in both outcomes.
  Record the result as evidence that operator preference is not yet a stable asset property.
- If opposite stable preferences exist, the only supported upgrade is “asset-level heterogeneity
  replicated on the existing source”; no feature-based explanation follows automatically.

## 4. Branch C — Gate A = CONDITIONAL

### Question

Does the interaction-level `|ACF1|` association replicate prospectively in a genuinely independent,
same-source asset cohort?

### Single experiment C1 — prospective same-source cohort

Acquire one cohort of **15–20 assets from one provider and one declared market-data policy**. Lock
the complete eligible universe and exclusions before operator outcomes exist; include all eligible
assets rather than choosing only predictor extremes. Compute `|ACF1|` with the unchanged formula
using data available no later than the validation cutoff, then measure the frozen interaction
outcome on the later test split. The primary Spearman/LOO analysis uses the new cohort only; the old
ten or pooled N=14 may be shown only as labelled context.

**Prerequisites**

1. `14_*` records `CONDITIONAL` under the unchanged rule.
2. One source can supply complete, consistently adjusted 2016–2025 OHLC data for the entire new
   cohort; no composite-provider cohort is allowed.
3. Asset universe, history/quality exclusions, provider fields, duplicate handling and hashes are
   pre-registered before feature inspection and operator outcomes.
4. The validation cutoff, test period and rule preventing any test-period information from entering
   the predictor are mechanically checked.
5. Every framework arm uses the frozen protocol and at least the three frozen seeds.

**Resource envelope**

- 15–20 assets × 2 framework arms × 3 seeds = **90–120 framework training runs**.
- Each run additionally retains per-sample predictions needed for the unchanged 4,000-resample
  paired bootstrap and interaction calculation.
- Independent assets have priority over adding further seeds to the discovery assets.

**Falsification / stop criteria**

Apply the existing Gate A operational criteria without adjustment to the new-cohort-only primary
analysis. The candidate fails if the Spearman sign reverses, ρ drops below +0.30, exploratory
`p` rises above 0.10, the LOO range crosses zero, one asset drives the sign, or an integrity problem
is found. Failure closes `|ACF1|` as the project's structural bridge and blocks a temporal
NS-inspired nonlinear operator benchmark in this research line under the current evidence. A
favourable result upgrades only portability of the interaction-level candidate; it still does not
show that NS works or justify asset/sample routing.

## 5. Highest-value next experiment

Conditional on Gate A being `CONDITIONAL`, **C1 is the single highest-value new experiment**.
The evidence matrix already answers the routing question negatively, while the structural bridge
is the weakest live link: R6 is directional only, contemporaneous and multiplicity-contaminated.
C1 directly tests that link with prospective timing and independent assets. It has higher decision
value than another seed on an old asset, another gate, or any new model family.

If Gate A is `FAIL`, C1 is not run; F1 becomes the only next experiment because a failed structural
bridge makes it necessary to establish the more basic asset-heterogeneity claim first.

## 6. Is the NS-inspired idea feasible?

### Short verdict

**Possible in principle, scientifically unqualified at present.** The useful insight is not “the
market obeys Navier–Stokes”. It is the narrower possibility that a small, parameter-tied temporal
operator could impose a productive bias for local nonlinear propagation and smoothing. Current
evidence neither supports nor refutes that proposition because no temporal NS-inspired nonlinear
operator has been tested.

### The narrow form that could eventually be defensible

After C1 succeeds, a later charter may specify a **temporal NS-inspired nonlinear operator** as a
local, parameter-tied residual update combining:

- a nonlinear flux-difference term (advection-like propagation across neighbouring time steps),
- a fixed-form second-difference term (diffusion-like smoothing), and
- an ordinary residual forcing map.

This is a hypothesis about inductive bias on a one-dimensional temporal grid, not a physical claim
about markets. It must have no learned router, no attention, no MoE, no pressure analogue and no
sample-level switching. The exact equations are deliberately not fixed here because implementation
is not authorised and the structural bridge has not passed prospective validation.

### Why it might work

- Parameter tying could express local propagation/smoothing with fewer degrees of freedom than a
  dense MLP, which is potentially useful in finite-data forecasting.
- Operator form remains genuinely unresolved after the generic linear/MLP comparisons.
- Capacity sensitivity shows that a fair test must separate form from parameter count; it does not
  eliminate the possibility that a better bias wins at equal capacity.

### Why it may fail

- Time is not a spatial coordinate, and no conserved quantity or governing PDE has been identified.
- State-dependent smoothing is close to existing convolution/SSM ideas; novelty may disappear once
  matched against ordinary temporal convolutions.
- Jump ratio has no discriminative power under the frozen definition, and sample-level routing is
  already unsupported.
- Any gain may be explained by capacity, optimization or regularization rather than NS-inspired
  structure.

### Eligibility for a later benchmark

A model benchmark may be proposed only if C1 survives. Before any implementation, its separate
pre-registration must name a prospective structural signature, an ordinary temporal-convolution
baseline, the matched parameter and compute budgets, and a failure rule. The idea is rejected if
its advantage over the generic nonlinear and ordinary convolution baselines is not seed-consistent
under the existing metrics, or disappears under parameter/compute matching. Negative results must
receive the same prominence as positive results.

## 7. Explicitly retired directions

- sample-level router, adaptive gate, MoE and attention;
- post-hoc HMM/regime replacement or feature/threshold search;
- high-volatility-only success as the primary NS benchmark, because volatility is not a universal
  selector;
- conservation and pressure analogies without a measured financial invariant or constraint;
- claims that eligibility, analogy or architectural novelty constitute evidence of effectiveness.
