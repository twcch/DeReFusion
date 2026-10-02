# Experimental doctrine: gates for computational research

**Status:** proposed; not yet adopted.
**Date:** 2026-09-28 (rev. 2, after external review).
**Scope:** how a protocol in this repository is built, gated and reported.
**Relationship to frozen documents:** this document governs the *construction*
of future protocols. It does not retroactively bind `phase1-v1.1-2026-09-19`,
and it does not convert any past deviation into a compliant one. Where it would
have caught something Phase 1 did, the remedy is a v1.2 amendment signed before
the affected stage runs — not an appeal to this text.

Every clause below is anchored to an external document or a measured incident.
Clauses written only from intuition are marked as such. §6 lists the anchors and
states, per anchor, how it was verified.

## 0. Why this exists

The repository already has strong *epistemic* hygiene: immutable receipts,
content hashes, attempt lineage, preregistered estimands. What it lacks is a
*process* layer that would notice when the thing being measured is not the thing
that was named. The incidents below all passed every existing check.

| Incident | Passed every existing check because | Caught by |
|---|---|---|
| I1 `revin-TimesNet` is **15.96×** its reference configuration (75,010,401 params vs 4,700,225; 99.95% of them in the Inception kernels) | nothing compared the configuration to its reference implementation | L1.2–L1.5 |
| I2 A05 recorded `status: resource_blocked` while its own `termination` field read `manual controlled interrupt during backward pass` | no check compared a status field against a termination field | L2.2–L2.3 |
| I3 `PACKAGE_ROOT` was a single path; parking CPU packages outside it made the runner see an **empty** root and select all 15 fits as fresh attempt-1 | it failed silently — no error, no conflict, just a different selection | L2.5 |
| I4 The C1 implementation silently dropped 96 windows via `nanmedian` where `diff(log(Close))` was undefined | the dropped windows were never counted, so the N looked correct | L0.3, L0.5, L0.7 |
| I5 Five FX feeds carry OHLC-envelope violations at 1.1–6.4% of rows, uniformly across train/val/test | declared qualitatively in prereg §2 with no count, magnitude or per-split incidence | L0.4, L0.6 |
| I6 Cross-platform newline conversion broke every frozen hash at once | no clause declared a line-ending policy for hash-compared artifacts | L1.7 |
| **I7** `tests/test_phase1_model_shapes.py` asserts shapes under a hand-written `d_ff=128` while the frozen configuration sets `d_ff=2048`; the gate was green and blind on **3 of 5** models | the fixture is a hand-copy of the configuration, so the two drift independently — and did. Measured against the frozen config: TimesNet 4,699,969 vs 75,010,369 (15.96×), iTransformer 29,376 vs 278,976 (9.5×), PatchTST 35,232 vs 284,832 (8.1×). The two it covered correctly — DLinear 4,664 and DeReFusion 28,436 — are exactly the two that never read `d_ff` | L1.8 |

The common shape: **a number was correct in itself and wrong in what it was
attached to.** The gates below exist to make that specific failure loud.

I7 is the sharpest of the seven, because it is the only one where **the gate
exists, passes, and is a statement about a model that is never run.** I1 is a
missing check; I7 is a check that manufactures false assurance. A missing check
leaves a hole a reader can see. A green check that covers nothing is worse than
no check, because it is evidence of a property nobody has.

It is worse than that, and the measurement is worth stating plainly. The
fixture's `d_ff=128` did not merely mis-state TimesNet: it made iTransformer
9.5× and PatchTST 8.1× too small as well. The gate was correct for DLinear and
DeReFusion — the only two models in the cohort that never read `d_ff`.
**The check was right precisely for the models that could not be wrong, and
wrong for every model that could.** This is structural rather than unlucky: a
drifted key is invisible to every architecture that does not read it, so a
fixture-level drift necessarily produces a gate that is green exactly where it
is not needed. It follows that the reach of such a gate is not merely unknown —
it is *inversely* correlated with where the risk is.

## 1. Four principles

- **P1 — One number, one provenance.** Every number that appears in a report
  resolves to an artifact hash and the configuration that produced it. A number
  whose provenance is a later reconstruction is labelled as such. The vocabulary
  is W3C PROV-O's: the number is an entity, produced by an activity, attributed
  to an agent. A record missing the agent is incomplete, not merely terse.
- **P2 — Assert before execute, and score the assertions honestly.** Every gate
  is an assertion evaluated *before* the expensive step it guards. But not every
  assertion can be automated, and a doctrine that demands automation where it is
  impossible will be quietly ignored. Each gate therefore scores **1** where a
  system runs and verifies it repeatedly, **0.5** where a human executes it and
  **documents the result**, and **0** where it is absent or fails. A stage's
  score is the **minimum over its gate groups**, not the mean. A check that can
  only be run by reading output is not disqualified by that fact; it is scored
  0.5, and it caps the stage.
- **P3 — Every deviation produces an assertion.** A deviation record that ends
  in a disposition and no new check has not finished. The disposition fixes this
  instance; the assertion is what stops the next one. The pattern is the
  constraint-suggestion cycle: observe empirically → form a candidate assertion
  → review it as a human → adopt it into the standing check set. The middle step
  is not optional; a suggestion adopted unreviewed is how a gate encodes a
  transient artifact of one dataset.
- **P4 — The count of trials is part of the result.** Selection over N
  configurations inflates the best observed performance whether or not the
  selection was deliberate, and the inflation does not care about motive. Every
  protocol declares N before outcomes exist, reports it, and reports the winner
  deflated. A result presented without its N is not interpretable — not
  "weaker", not "less rigorous": uninterpretable.

## 2. Layers

Each layer has an entry condition, assertions, the statuses it may assign, and
an exit condition. A layer may not be entered until the previous layer's exit
condition holds. Nothing in L2–L5 may be used to revise an L0 or L1 verdict.

### L0 — Input eligibility

Establishes that each asset is a well-formed instance of the declared task.
Runs before any model, and its output is a per-asset status table.

- **L0.1** Every asset is identified by a content hash, verified at consumption
  time and not only at registration.
- **L0.2** Every asset declares its **time grid** (trading calendar vs calendar
  day) and the wall-clock span that `seq_len` and each horizon denote on that
  grid. Assets on different grids are made commensurable by **exactly one
  declared mechanism**, named before outcomes:
  - **(a) common index** — all series aggregated onto a shared synchronous time
    index before any comparison;
  - **(b) scale-free normalisation** — each series scored against a
    same-frequency, same-grid reference, so pooling happens on dimensionless
    ratios;
  - **(c) rank estimand** — the target is redefined as a cross-sectional rank
    within the cohort, which removes the comparability question at the estimand
    level rather than correcting for it.

  These three are not interchangeable and none is a default. A comparison whose
  commensuration mechanism is unstated is not a comparison.
- **L0.2a** *Comparability of grids is decided by an artifact, not by
  inspection.* A reference distribution is computed once from the declared grid
  and stored; a candidate grid is compared to it by a metric whose units can be
  stated in one sentence; a failure names the specific value and timestamp that
  caused it. **A test with no effect-size floor is inadmissible**: the
  chi-square test fires on contamination of roughly 0.01% in 10⁸ points, which
  is why the standard practice reports a maximum per-value probability change
  rather than a p-value.
- **L0.3** The declared target is well-defined across the asset's observed range
  **in every split**. A target undefined on any row of a split is a finding, not
  a filter.
- **L0.4** Input-channel defects and target-channel defects are separated and
  reported separately. A defect confined to input channels is input noise; a
  defect touching the target is a validity question. These do not carry the same
  consequence and must not share a status.
- **L0.5** No observation is removed, offset, clamped, filled, or dropped for
  being non-finite without a **pre-declared rule** naming the transform, the
  affected row set, and the sensitivity analysis that will accompany it. Silent
  removal through a NaN-skipping reducer is removal.
- **L0.6** Every defect is **quantified**: affected row count, magnitude
  distribution, and per-split incidence. A qualitative declaration is not
  sufficient for a confirmatory cohort.
- **L0.7** Defect handling is declared **before outcomes are visible**. A defect
  first noticed after outcomes exist may be reported and bounded by sensitivity
  analysis; it may not be handled by a newly written rule.
- **L0.8** An artifact larger than its host's limit is stored by a declared
  mechanism — a content-addressed external store with a committed pointer, or an
  immutable release — never by committing the blob. Version identity uses
  immutable numbered versions plus an alias; a mutable stage pointer is not
  admissible as provenance. A retention policy is declared per artifact class,
  because no host prescribes one.

**Statuses:** `eligible` · `inapplicable` · `requires-resolution` ·
`excluded-by-protocol` · `protocol-deviation`.
**Exit:** every asset in the declared cohort carries one status and its evidence.

### L1 — Implementation fidelity

Establishes that each named model **is** the thing its name refers to. Runs
before any number from that model is used for planning or comparison.

- **L1.1** Each model names its reference implementation as repository, file and
  commit, plus the exact reference configuration.
- **L1.2** A line-by-line hyperparameter comparison against the reference is
  recorded, with every departure and its stated reason.
- **L1.3** A departure is admissible only with a **scale check** on the values
  that set capacity: `d_ff/d_model`, `d_model/n_heads`, and parameter count
  against the reference configuration.
- **L1.4** A model whose capacity differs from its reference by more than the
  declared factor is **not that baseline**, whatever its filename says. It may
  run as a declared variant, reported under a different name.
  *Threshold: **TBD**. `4×` is used as a placeholder because it admits the
  conventional FFN expansion while rejecting I1 (15.96×) and the
  head-dimension departure (64 → 4). **No external source publishes a
  capacity-equivalence threshold** — this was searched for and not found. The
  number is therefore a project choice, and it must be replaced by a stated
  basis before any protocol cites it as a rule.*
- **L1.5** A hyperparameter is **not shared across architectures when it does not
  denote the same quantity in them.** Sharing a name is not sharing a meaning.
  An equal-capacity-budget grid is admissible for models where the parameters
  play the same role, and inadmissible where they do not.
- **L1.6** Determinism settings and any unsupported operations they disable are
  recorded.
- **L1.7** Any artifact whose hash is compared across platforms has a declared
  line-ending policy.
- **L1.8** **A check on a frozen configuration reads the frozen configuration.**
  A fixture that restates a configuration by hand tests the restatement. Where
  the two can drift they will, and the check stays green throughout. The shape
  gate was the worked example: under a hand-written `d_ff=128` it made TimesNet
  4,699,969 parameters, iTransformer 29,376 and PatchTST 35,232, against the
  frozen configuration's 75,010,369, 278,976 and 284,832 — and it passed. It was
  green because the two architectures it happened to get right are the two that
  never read `d_ff`. **A gate whose reach is unknown must be assumed wrong
  wherever the drifted key is read, and a drifted key is read only by the models
  that can be wrong.**
  Two channels must both be closed, because a check that reads the frozen file
  but restates anything else has moved the drift rather than removed it:
  1. every value the executing harness **passes** is read from the frozen
     configuration, never transcribed;
  2. every value the harness does **not** pass is read from the harness's own
     defaults, for the same reason — production silently takes those defaults,
     so a default that moves alters the frozen protocol with no recorded event.
  The gate also asserts that its model list still equals the frozen model list,
  so a stale mapping fails the check instead of quietly narrowing it.
  *Status: both channels are closed in `tests/test_phase1_model_shapes.py` as of
  the commit that added this clause. The gate now builds its configuration from
  the frozen file and the manifest, reads `run.py`'s defaults for `top_k`,
  `num_kernels` and `activation`, and asserts its module map covers
  `available_models`. Verified by measurement, not by the colour of the test:
  under the repaired gate TimesNet reports 75,010,369 parameters, matching the
  Stage A receipt digit for digit.*

**Statuses:** `reference-matched` · `declared-variant` · `unverified`.
**Exit:** no model enters L2 or above at `unverified`.

### L2 — Calibration

Establishes feasibility and cost. **Produces no scientific claim.**

- **L2.1** Calibration measures cost and feasibility only. Its losses are not
  compared, ranked or published as evidence, and every artifact it emits says so
  in its own metadata.
- **L2.2** A resource verdict (`resource_blocked`, `timed_out`) must derive from
  a threshold **validated on the same device class**. An unvalidated threshold
  produces a finding about the threshold, not about the model.
- **L2.3** A `status` field and a `termination` field must not contradict each
  other. Where they do, the contradiction invalidates the *status*, and the
  termination record stands as the account of what happened. A later record may
  not overwrite the earlier one; it may only be added.
- **L2.4** Attempt lineage: each logical run has exactly one validated successful
  attempt. Zero is incomplete; more than one is a conflict and fails stop.
- **L2.5** **The attempt visibility domain is closed.** Every reader of attempt
  state resolves the same set of roots. A reader that globs a subset is a silent
  failure generator and is itself a defect.
- **L2.6** No resource plan may rest on a configuration that has not passed L1.
- **L2.7** A device change inside a stage is a **protocol event**, not an
  operational choice, and requires a prior amendment however small its scope.

**Statuses:** `calibration_pass` · `resource_blocked` · `failed` · `timed_out` ·
`artifact_invalid` · `infrastructure_interruption`.
**Exit:** every fit carries a status whose supporting fields do not contradict it.

### L3 — Screen

Establishes runtime and gross failure across the declared cohort.

- **L3.1** Evaluation keys are generated **before** models run; every compared
  model covers the same keys, including incomplete test batches.
- **L3.2** Screen does not select assets, change the primary model pair, or
  determine which failures to hide.
- **L3.3** A model family that fails or is blocked at screen is recorded with its
  reason and is not silently dropped from later stages.
- **L3.4** **Where the protocol admits any feedback loop, the final evaluation is
  sealed.** A loop that returns scores during development lets the configuration
  be tuned against the evaluation set, and the loss is measurable: in the largest
  recorded instance, 63.7% of entrants beat the benchmark on their best
  submission but only 48.4% on the submission they finally chose — a loss of
  15.3 percentage points attributable to acting on validation feedback. Either
  the loop is absent, or the final block is evaluated without it.

**Exit:** a coverage table showing every declared key covered by every compared
model, or an explicit statement of which keys are not.

### L4 — Confirmation

- **L4.1** The estimand, its strata and its pooling rule are declared before
  outcomes.
- **L4.2** Only pre-declared pairs are compared.
- **L4.3** Every dynamic transform — scaling, imputation, feature computation,
  selection, tuning — is fit on the applicable training prefix or inner
  validation loop only, and the fit domain is recorded per artifact.
- **L4.4** An active confirmation protocol is not modified. Approved changes take
  a **new protocol ID**.
  *Basis: the operative part of this clause is the new protocol ID, not the word
  "independent". No general regulatory rule requires independent reproduction of
  a final primary analysis. In clinical research, separation is mandated only
  for unblinded **interim** analyses; the general rule is that a specification
  must be **replicable by a third party**, which is a property of the
  specification, not a role. Presenting the stronger reading as a ported
  requirement would be inventing a rule rather than porting one.*
- **L4.5 — The trial count is declared, not discovered.** N is fixed before the
  first run and each seed is a trial, not a technical detail. Configurations are
  justified by the research question, not by available compute. Where trials are
  correlated — as repeated seeds are — the effective count is derived by a
  **declared dimension-reduction step** (e.g. principal components over the
  trial-outcome matrix), not taken as the raw count.
- **L4.6 — Minimum sample length is checked, not assumed.** Before running,
  compute `MinBTL = 2·ln(N)/E[max]²` for the declared N against the available
  sample, and state the outcome in the design section. MinBTL is a **necessary,
  non-sufficient** condition. When it fails, the protocol's honest framing is
  *comparison under a multiple-testing correction*, not an edge claim, and the
  text must say so in those words. Worked reference: with 5 years of data, no
  more than ~45 independent configurations may be tried before an in-sample
  Sharpe of 1 corresponds to an expected out-of-sample Sharpe of zero; with 2
  years, the figure is 7.
- **L4.7 — The winner is reported with a deflation table.** For the selected
  configuration, and for each declared stratum separately: N, the variance of
  the trial statistics, sample length, skewness, kurtosis, and the resulting
  deflated statistic. Reporting only the selected configuration's raw statistic
  is incomplete. Reporting only the pooled winner, without per-stratum rows, is
  incomplete.
- **L4.8 — Selection overfitting is estimated and gated.** Over the declared
  configuration set, compute the probability of backtest overfitting by
  combinatorially symmetric cross-validation — an even number of partitions (16
  is conventional), N ≫ 10 so the rank statistic has sufficient granularity, all
  configurations on the declared common index — and report the estimate.
  **A value above 0.05 is a rejection, not a caveat.** State the method's own
  two limitations alongside the number: the symmetric split is unsuitable under
  strong autocorrelation, and the estimator weights all sample statistics
  equally.
- **Scope note for L4.6–L4.8.** The minimum-length bound and the deflation
  machinery are derived for a selection criterion that is a **Sharpe-like
  ratio**. L4.8's estimator is the general one: its inputs are a matrix of
  per-period outcomes and a metric estimable on subsamples, and it is
  explicitly model-free and non-parametric, so **it transfers to an error
  metric unchanged**. L4.6's constant and L4.7's deflation **do not**. A
  protocol that selects on forecasting error rather than on a Sharpe-like ratio
  must mark those two as having **no derivation in the source literature**, and
  disclose the trial count as an *uncorrected* caveat rather than present a
  corrected result. A protocol states which of the two cases it is in **before
  outcomes exist**, because the distinction determines what the confirmatory
  claim is allowed to say.
- **L4.9 — Labels are purged and embargoed where the cohort is pooled.** Where
  labels span an interval (an h-step-ahead target covers `[t, t+h]`), training
  rows whose label interval overlaps any evaluation row are removed, and an
  embargo of at least the label span is applied forward from each evaluation
  block. Where the cohort is pooled into one model, the purge is applied **across
  assets as well as across time**, since label intervals of different assets
  overlap.
  *Verified status in this repository: the condition is satisfied by
  construction, and this is a strength worth recording rather than a defect.
  `Dataset_Custom` sizes each split's usable index range as
  `len(slice) − seq_len − pred_len + 1`, so the maximum target end index is
  exactly the split's `border2`, and the target never crosses the boundary.
  Boundary leakage of up to `h−1` steps, which a naive chronological split
  admits, cannot occur. The cross-asset clause is additionally inapplicable here
  because each run reads exactly one asset file, so no pooled model exists.
  Both facts must be re-verified for any protocol that changes either the loader
  or the pooling.*
- **L4.10 — Every gate names who can satisfy it.** A gate whose evidence is an
  *object* — a deposit, a hash, a published artifact — may be satisfied by the
  author. A gate whose verdict *asserts validity* cannot be: it requires a named
  party other than the author, or it is recorded as **self-assessed** and scored
  0.5 under P2. This is the artifact-badging distinction applied as a rule: only
  object-availability is author-serviceable, and every validity-asserting badge
  requires an independent evaluator. *Synthesis — mine, not a ported standard.
  The honest consequence for a solo project is that its validity-asserting gates
  are self-assessed by construction, and the doctrine must say so rather than
  implying otherwise.*

### L5 — Temporal robustness

- **L5.1** Rolling splits use explicit, end-exclusive date boundaries.
- **L5.2** A fixed-ratio split is never relabelled as rolling-origin evaluation.
- **L5.3** Each split may borrow only the preceding `seq_len` rows as input
  context; scaling is fit on training rows only.
- **L5.4** Sensitivity analyses declared before outcomes are reported alongside
  the primary result; those declared after are labelled exploratory and carry no
  confirmatory weight.

## 3. Deviation, amendment and reporting

- **X1** Every deviation produces three things: a record, a root cause, and **an
  assertion added to this doctrine or to the affected protocol's gates**. A
  deviation that produces no assertion has not been closed.
- **X2** A violation is recorded as a violation. It is not relabelled as a
  technicality, and its motive does not make it compliant. Splitting a rule into
  "letter" and "spirit" and claiming only the letter was touched is relabelling.
- **X3** A post-hoc record does not acquire amendment authority by being written
  down. Authority comes from being written before the outcome, not from being
  written carefully. A recorded change that cannot obscure the previous record
  is the standard to meet; a change that replaces it is a different act
  entirely.
- **X4** Motive, consequence and compliance are three separate questions. A
  favourable answer to one does not answer the others.
- **X5** Unsupported numeric thresholds are marked `TBD` and named as such in any
  gate table, together with whether a research-question, power, precision or
  resource analysis is needed to set them.
- **X6** N is disclosed in the abstract, not buried in a methods subsection. A
  reader who cannot see the trial count cannot weigh the result.
- **X7** **Do not inherit a terminology convention silently.** The terms
  "reproducibility" and "replication" are used with **opposite meanings** in two
  live conventions: in the National Academies / NISO sense, reproduction reuses
  the original data and code while replication collects new data; the ACM
  badging scheme deliberately swapped the two. Every document in this repository
  states which convention it uses and what it means by each term. A reader who
  assumes the other convention must be visibly wrong, not quietly misled.
- **X8** A partial outcome is reported as a partial outcome. Where a check
  succeeds for some strata and not others, both are published with equal
  prominence, and the partial result is not dressed with a badge or a label that
  implies the whole. A partially satisfied claim that is visible in the record
  is a contribution; the same claim silently promoted is a defect.

## 4. Gate table format

Every gate evaluation returns a table of:

| check | input | status | credit | evidence | next permitted action |
|---|---|---|---|---|---|

`credit` ∈ {`0`, `0.5`, `1`} per P2. A row whose evidence is a narrative rather
than an artifact hash **or a named human sign-off** is not a pass. The stage
score is the minimum credit within each gate group, and the stage is reported by
its vector of group minima — a single averaged figure hides the weak gate, which
is the only one that matters.

## 5. Incident index

Clauses exist because of specific events, not in the abstract. The mapping is
maintained so that a clause whose incident has been structurally eliminated can
be retired, and so that a clause cannot be weakened without confronting what it
was written for.

| Clause | Incident |
|---|---|
| L0.2, L0.2a | BTCUSD 3,653 calendar-day rows vs equities ~2,514 trading-day rows |
| L0.3, L0.5, L0.7 | I4 — silent `nanmedian` removal of 96 undefined-log windows in C1 |
| L0.4, L0.6 | I5 — five FX feeds, violations unquantified in prereg §2 |
| L0.8 | 4 untracked checkpoints at 286.8 MB each, against a 100 MiB host limit |
| L1.2–L1.5 | I1 — `revin-TimesNet` at 15.96× its reference |
| L1.7 | I6 — cross-platform newline conversion |
| **L1.8** | **I7 — the shape test's hand-copied `d_ff=128`** |
| L2.2–L2.3 | I2 — A05 status/termination contradiction |
| L2.5 | I3 — single-rooted `PACKAGE_ROOT` |
| L3.4 | no incident in this repository; written from the measured M5 leaderboard loss |
| L4.4 | the deviation record's letter-vs-spirit framing |
| L4.5–L4.8 | no incident in this repository; written from the multiplicity literature, and from Phase 1's N |
| L4.9 | no incident; written from the leakage literature, and records a **verified strength** of the existing loader |
| L4.10, L3.4 | no incident; written from the badging-role and feedback-loop evidence |

Note on rows with no incident: they are the only clauses not earned by a
failure here. They are included because the failure they prevent is documented
elsewhere and expensive, and they are marked so that a future reader can tell
which parts of this doctrine are paid for and which are borrowed.

## 6. External anchors

Sources are tagged by how they were verified. `verified` = read in full at the
primary document. `partial` = the document was reachable but a specific number
or quotation was not. `secondary` = concordant summaries only; **do not quote
these verbatim.**

| Clause | Anchor | Tag |
|---|---|---|
| P1 | W3C PROV-O (entity / activity / agent; 12-term starting point) | verified |
| P2 | Breck et al., "The ML Test Score," SysML 2019 — 28 checks in 4 areas, scored 0 / 0.5 / 1, aggregated by minimum over areas, not mean | verified |
| P3 | Deequ constraint suggestion — suggestions are generated from data and must be human-reviewed before adoption | verified |
| P4, L4.5, L4.7, X6 | Bailey, Borwein, López de Prado & Zhu, *Notices of the AMS* 61(5), 458–471 (2014) — "a backtest which does not report the number of trials N … makes it impossible to assess the risk of overfitting" | verified |
| L4.6 | Same source, Theorem 3.1: `MinBTL < 2·ln(N)/E[max_N]²`; and the authors' own caveat that it is necessary and not sufficient | verified |
| L0.2(a), L4.8 | Bailey et al., "The Probability of Backtest Overfitting" — CSCV requires a true matrix with synchronous rows, and prescribes aggregating configurations to a common index; PBO > 0.05 as the conventional rejection, S = 16, N ≫ 10; authors' own stated limitations | verified (journal volume/pages unverified) |
| L4.7 | Bailey & López de Prado, *J. Portfolio Management* 40(5), 94–107 (2014) — the four-item disclosure list (N, variance of trial statistics, sample length, skewness/kurtosis) | verified |
| L0.2(b) | M4 evaluation code — MASE scaled by each series' own in-sample seasonal-naive error at its own frequency; OWA normalised against a same-frequency benchmark; results reported per frequency, never pooled across raw series | verified (the cross-frequency aggregation weight is **unknown**) |
| L0.2(c) | M6 — the target redefined as a cross-sectional rank probability over the cohort, the only one of the three mechanisms that does not assume commensurability | verified |
| L3.4 | M5 accuracy competition — 63.7% beat the benchmark on their best submission vs 48.4% on their selected submission; the sealed test phase with no leaderboard | verified |
| L4.9 | López de Prado, *Advances in Financial Machine Learning* Ch. 7 and Ch. 12 — purging and embargo; **chapter and section titles are safe to cite, verbatim quotations are not** | secondary |
| X3 | 21 CFR 11.10(e) — "Record changes shall not obscure previously recorded information" | verified (quoted in the regulatory review) |
| X7 | NASEM (2019) *Reproducibility and Replicability in Science*; NISO RP-31-2021 (ORO / ORO-A / ROR / ROR-R / RER badges); ACM's documented swap of the two terms | verified (NASEM, NISO); ACM page unreachable, definitions from index |
| X8 | NISO RP-31-2021 — partially satisfied findings "should be made visible in some way in the scholarly record", and should not receive a badge | verified |
| L0.8 | GitHub file-size limits (100 MiB hard, 50 MiB warning); DVC pointer-plus-remote including plain SSH/SFTP; MLflow model stages deprecated as of 2.9.0 in favour of immutable versions plus aliases | verified |
| L4.4 | ICH E9(R1) §A.2 (the specification must be replicable by a third party); FDA DMC guidance §6.4 (separation mandated for unblinded interim analyses, not final ones) | verified |
| L4.10 | ACM artifact badging v1.1 — the scheme deliberately prescribes no reviewer roles; only object-availability is author-serviceable, every validity-asserting badge requires a party other than the author | verified (roles); index (badge definitions) |
| L0.2a | TensorFlow Data Validation — maximum per-value probability change with a prior-derived bound, adopted because the chi-square test fired on 7 of 10 trials at 0.01% contamination in 10⁸ points; the schema is a version-controlled production asset | verified |

### Known gaps in this doctrine

Stated rather than hidden:

1. **L0.2a's metric and bound are named but not yet selected.** The pattern is
   adopted; the specific distance and its threshold are `TBD` and require the
   reference distribution to be computed first.
2. **L1.4's factor has no external precedent** — searched for, not found. It is
   a project choice presented as one.
3. **L4.10 makes the solo-project constraint explicit but does not solve it.**
   Under this clause, every validity-asserting gate in this repository is
   self-assessed and scores 0.5. That is the honest score, not a limitation to
   be worked around by redefining the gate.
4. **No clause covers who adjudicates a disputed gate.** The badging evidence
   says the venue decides, post-publication. Until there is a venue, disputes
   are recorded open, not resolved by fiat.
5. **The anchors marked `secondary`** — AFML's chapter content, the "10 Reasons"
   list, and the ACM badge definitions — must be verified against the primary
   documents before any of them is quoted in a manuscript.
6. **L4.6 and L4.7 do not transfer to an error-metric protocol.** The
   minimum-length constant and the deflation statistic are derived for a
   Sharpe-ratio selection criterion, and this repository's comparison is on
   forecasting error. The scope note appended to L4.8 records the consequence:
   the multiplicity correction can be *estimated* (L4.8 transfers) but not
   *computed* (L4.6, L4.7 do not), so the trial count is disclosed as an
   uncorrected caveat. Closing this is an open methodological question, not a
   missing citation, and it must not be papered over by quoting a constant that
   was derived for a different metric.
