# DeReFusion repository and evidence audit

**Audit date:** 2026-09-18  
**Scope:** local workspace plus `origin/main`; code, data, reports, raw evidence,
claims, reproducibility and repository hygiene.  
**Rule:** a negative result is evidence and is retained. A redundant copy,
broken path or unsupported claim is not evidence and may be corrected or
removed.

## 1. Executive finding

This is a research fork of the public DeReFusion/TSLib codebase, not a clean
release of the paper's official experiment package. It has nevertheless built a
substantial, unusually explicit audit trail around reproduction, gating,
capacity controls and prospective testing.

The defensible result stack is:

1. the DeReFusion pipeline and its baselines can be run on the supplied OHLC
   data under a shared harness;
2. the simple additive model should not be described as universally superior;
3. a volatility-aware learned gate was worse than direct addition in the tested
   setting, and sample-level routing received no stable support;
4. Gate A failed under its frozen rule because operator preference changed sign
   with capacity on BOE and EASTMONEY;
5. F1 succeeded narrowly: four of seven additional assets were seed- and
   width-stable, but three of seven were not;
6. C1 completed all 120 locked runs and failed prospectively (`rho=-0.1519`,
   `p=0.5227`). An independent result-blind audit verified the sealed inputs
   and failure robustness, while exposing an omitted aggregation-rule dependency.

The project is therefore best described as an **evidence-closure study of when
simple and nonlinear residual operators differ**, not as evidence for a
Navier–Stokes mechanism, an asset router or a universal forecasting winner.

## 2. Verification boundary

### Independently recomputed in this audit

- data registry: 61 CSV files, 149,474 rows;
- F1 grid: 792 state rows = 63 paired settings = 126 arm fits;
- F1 verdict: stable `TM` (nonlinear) and `DJI`, `EURUSD`, `USDJPY` (linear);
- C1 delivery completeness: 20 assets × 2 arms × 3 seeds = 120 runs,
  240 prediction/target arrays, with the result-free Stage-1 manifest;
- C1 analyst pipeline executed from the locked arrays; the independent task
  verified inputs and robust failure but could not certify exact numerical
  equality because the whitelist omitted the aggregation program.

### Verified as internally consistent, not independently re-trained here

- original reproduction runs and gate-variant results;
- the N=10 structural exploration;
- Gate A's N=14 table and capacity outputs;
- all historical training logs and checkpoints.

This distinction matters: checking a report against its own retained raw files
is stronger than trusting prose, but weaker than rerunning every neural network
fit on a second machine.

## 3. Claims audit

| Claim | Audit status | Correction / boundary |
|---|---|---|
| This fork is the paper's "official code" | **incorrect** | It is a research fork based on the public implementation and TSLib. |
| All models are directly comparable | **partly supported** | The trained models share the main data/split/training harness. Zero-shot foundation models have a different training provenance and must be reported separately. |
| Test data did not influence fitting | **algorithmically supported, procedurally weak** | Checkpoint selection used validation loss, but the training loop printed test loss every epoch. The code now leaves test data untouched until final evaluation. Historical numerical results were not altered. |
| `|ACF1|` explains operator preference | **candidate only** | Discovery involved eight screened features, N is small, and Gate A mixed discovery with four domain-shift assets. C1 is the prospective closure test. |
| Gate A externally validated the bridge | **false** | Gate A = FAIL under the frozen capacity-contradiction rule; 10/14 assets were from discovery. |
| Asset-level operator heterogeneity exists | **narrowly supported** | F1 added four stable assets in both directions, but 3/7 failed the strict rule. It is a tendency, not an invariant. |
| A structural feature can route models | **not supported** | Sample-level routing stayed closed; contemporaneous association is not an ex-ante selector. |
| Financial prices obey Navier–Stokes dynamics | **unsupported and scientifically overclaimed** | No state variables, conservation law, boundary conditions or PDE residual have been identified. NS remains an untested architectural analogy only. |
| NS-inspired models were falsified | **false** | No such model has been run. Correct wording: "not tested and not refuted; current evidence does not authorise the benchmark." |

## 4. Data audit

The generated registry is
[`reproduction/results/dataset_registry.csv`](../reproduction/results/dataset_registry.csv).
All files have the expected schema, sorted unique dates, and finite complete
numeric cells.

Findings requiring disclosure:

- FX OHLC envelope inconsistencies occur in `AUDUSD`, `EURUSD`, `GBPUSD`,
  `USDCAD` and `USDJPY`. They are small and likely reflect field-level rounding;
  no source values were silently replaced.
- WTI has two rows with non-positive prices around the April 2020 futures
  dislocation. Log-return features are undefined around them. The current C1
  feature aggregation ignores resulting `NaN` values via `nanmedian`; this is a
  methodological limitation, not data corruption.
- the previous README overstated six original equity/index row counts and the
  end date. The registry is now the source of truth.
- 27 files are labelled `supporting-unregistered`: they are retained data, not
  authorised additions to a confirmatory cohort.

## 5. Code and reproducibility audit

Corrections made in this pass:

- removed per-epoch test evaluation from long-term training;
- replaced a Unicode error marker that could crash GBK terminals;
- made C1 interaction analysis use the current repository and interpreter;
- moved research-analysis outputs from an external personal directory into
  `reproduction/results/`;
- made the structural-validation launchers repository-relative;
- made Yahoo data download paths repository-relative;
- split minimal and optional dependencies, and repaired the Dockerfile's
  reference to a nonexistent root `requirements.txt`;
- added a deterministic dataset audit and registry.

Remaining limits:

- CUDA determinism is not forced; multi-seed reporting is still necessary;
- optional foundation-model packages have heavy, platform-sensitive dependency
  trees and should use isolated environments;
- several historical reports record now-superseded operational states. They are
  retained only inside the numbered evidence chain and explicitly marked as
  historical rather than rewritten after the fact;
- C1 was executed on CPU with Python 3.11.9 / torch 2.5.1, while some receiving
  analyses were run in an existing Python 3.13 environment. The raw array hashes
  and frozen scripts, rather than environment identity alone, anchor comparison.

## 6. Evidence chronology and present status

| Stage | Status | What it means |
|---|---|---|
| Paper/code reproduction | complete enough for audit | Core runs and outputs retained; not every paper claim was retrained in this pass. |
| Gate/fusion probes | complete | Learned gating did not justify replacing direct addition in the tested setting. |
| Structural discovery | exploratory | `|ACF1|` emerged after screening; cannot be presented as confirmatory. |
| Gate A | **FAIL** | Frozen capacity contradiction triggered; bridge not validated. |
| F1 breadth panel | **SUCCESS with major caveat** | 4/7 new stable assets and both preference signs; 3/7 unstable. |
| C1 execution | complete | 120/120 runs, raw arrays and provenance present. |
| C1 independent recomputation | qualified completion | Inputs and failure robustness verified; exact equality is not claimed because the handoff omitted the aggregation rule. |
| NS/operator benchmark | not authorised | No implementation result exists. |

## 7. Repository cleanup policy

Removed or consolidated:

- inherited TSLib shell scripts for unrelated ETT/ECL/weather/anomaly tasks;
- obsolete top-level roadmaps and external-review briefs superseded by this
  audit and the numbered evidence chain;
- empty log/error files and `.DS_Store`;
- superseded rendered report variants, keeping the latest bilingual IEEE stage
  report and its source.

Deliberately retained:

- negative results and failed-gate reports;
- raw predictions, targets, manifests, hashes and checkpoints supporting active
  claims;
- frozen pre-registrations and historical numbered reports;
- execution anomalies, including the C1 watchdog false-kill history and the
  Stage-1 manifest handoff hash correction.

Deleting inconvenient evidence would make the repository smaller but the
research weaker. Cleanup is therefore based on evidentiary value, not whether a
result was positive.

## 8. Completion assessment

| Dimension | Rating | Rationale |
|---|---:|---|
| Code/package hygiene | 75% | Core paths and install story repaired; optional backends remain platform-sensitive. |
| Data traceability | 85% | Hash registry and cohort labels exist; provider metadata for some supporting files remains incomplete. |
| Reproduction evidence | 80% | Strong retained artifacts; not every original paper table was independently rerun. |
| Hypothesis discipline | 90% | Frozen gates, negative-result retention and explicit claim boundaries are unusually strong. |
| Independent confirmation | 80% | F1 recomputed locally; C1 inputs and failure robustness independently checked, with a disclosed handoff ambiguity. |
| Publication readiness | 65% | Evidence is organised, but a paper-quality statistical model comparison and contamination-aware TSFM benchmark remain future work. |

**Overall:** the project is beyond exploratory prototyping and has a credible
audit trail, but it is not yet evidence for a new NS-based method or a finished
state-of-the-art forecasting paper. The strongest near-term publication is a
carefully framed negative/heterogeneity study with prospective closure, not an
architecture novelty claim.
