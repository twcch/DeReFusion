# 18 · NS-inspired operator: experiment design DRAFT (design only — **not authorised, not executed**)

**Status:** design document. **Nothing here is scheduled, and no experiment may start from this
file.** It changes no definition, threshold, seed, metric, split, horizon, window or capacity
setting, and it does not touch `11` / `12a` / `12b`.

**Prerequisites before this design may even be promoted to a real pre-registration:**

1. `14_*` records Gate A (and if it records `FAIL`, this design is closed — see `16` §3);
2. if Gate A records `CONDITIONAL`, the independent **prospective same-source cohort**
   (`16` §4, branch C1) must first replicate the structural candidate;
3. the promotion happens as a **separate file with its own commit**, and its own frozen criteria
   (Gate-A correlation thresholds are **not** transplantable — see `17` §"没有指标遗产").

**Source:** the direction, the four bridge elements and the three requirements below follow the
external assessment archived in `17_ns_directions_astra.md` (29 KB, verbatim), with the corrections
recorded in `17a_agent_check_notes.md` (§2.1 fit counts; §2.2 parameter matching).

---

## 1. Requirements this design must satisfy (from `17`, `ns_hypothesis_charter.md`)

| Requirement | How this design satisfies it |
|---|---|
| **Prediction effectiveness ≠ mechanism explanation** | two separate hypotheses, H-pred and H-mech (§3), each with its own falsification |
| **Pre-fixed structural marker** | the marker is fixed here, before any run (§5), and must be causal |
| **Removal test** | arm A4 deletes the claimed structure from arm A3 (§4) |
| **Ordinary-architecture reproduction test** | arm A5 hands the same quantity to an ordinary model as an explicit feature; plus A1 (temporal convolution) and A2 (SSM) as matched baselines |
| **Capacity matching** | all arms equal total parameters within ±2 %, identical optimiser/steps/batch/lr/early-stopping/seeds; FLOPs reported (§6) |
| **No router / gate / MoE / attention / pressure / learned switching** | the candidate is a single fixed-form operator; no input-dependent branch selection |
| **Terminology** | *temporal NS-inspired nonlinear operator*; the physical reading is explicitly disclaimed |

## 2. The one retained direction, stated so it can be wrong

A **temporal NS-inspired nonlinear operator** as the nonlinear branch of a decomposition–residual
forecaster: a local, parameter-tied residual update on a one-dimensional temporal grid, composed of

1. an **advection-like term** — a state-dependent local shift/transport of the hidden sequence;
2. a **second-order scale term** — a local variance-like quantity
   `q_t = C(x²)_t − (C(x)_t)²` over a small temporal window `C`, used to modulate the update;
3. an **ordinary residual forcing map** — a plain learned map for what the first two do not explain.

**Honest caveat carried in the design (from `17`):** `q_t` is computable by ordinary convolution
plus squaring, and a fixed smoothing is realisable by ordinary convolution. So the *only* thing this
design can establish is whether **using** that structure as an inductive bias adds anything beyond
receiving the same information as a feature. That is exactly what A5 tests.

## 3. Hypotheses and their falsification

- **H-pred (effectiveness):** the candidate (A3) predicts better than the best of the capacity-matched
  ordinary arms {A0 MLP, A1 temporal convolution, A2 SSM}.
  *Falsified if:* A3 does not beat the best ordinary arm with the **same sign in all three seeds**,
  or the margin falls below the pre-specified minimum (see §7), or the advantage disappears once
  parameters **and** compute are matched.
- **H-mech (mechanism):** the gain is attributable to the second-order scale term.
  *Falsified if:* **A4 ≈ A3** (removing the term changes nothing, within the pre-specified tolerance),
  **or A5 ≈ A3** (an ordinary model given `q_t` as a feature reproduces the gain), **or** the marker in
  §5 fails to explain **when** the gain appears.

Rejecting H-pred does **not** reject the NS frame as a physical analogy — it only shows this
implementation is not useful here. Rejecting H-mech while H-pred survives means *"some structure
helped, but not the one claimed"*.

## 4. Arms (each trained under the frozen protocol)

| Arm | What it is | Purpose |
|---|---|---|
| **A0** | MLP nonlinear branch (parameter budget P) | generic baseline |
| **A1** | dilated temporal convolution, same P | ordinary local structure baseline |
| **A2** | simple diagonal linear state-space (SSM), same P | long-range ordinary baseline |
| **A3** | candidate: advection-like term ⊕ second-order scale term ⊕ residual forcing, same P | the candidate |
| **A4** | A3 with the second-order scale term **removed** (parameters redistributed to the residual map) | **removal test** |
| **A5** | ordinary model (A0/A1 hybrid of equal P) that **receives `q_t` as an explicit input feature** | **bias-vs-feature test** |

Every arm is trained on the same windows with the same seeds; the frozen framework (linear base +
additive fusion + RevIN) is unchanged around the branch.

## 5. Pre-fixed structural marker (must be causal)

- **Marker:** for each forecast window, compute `q` over the **input window only**
  (`seq_len = 96`), i.e. from data at or before the forecast origin — never from the test window.
- **Prediction:** the candidate's advantage concentrates in windows with **high `q`** (locally
  volatile, second-order-active segments).
- **Test:** stratify the paired per-window difference (A3 − best ordinary arm) by pre-fixed `q`
  quantiles computed on **training data only**, and require the ordering to be monotone with a
  seed-consistent sign.
- **Leakage check (must print a line in the run log):** the marker is computed from indices
  `< border1_test`; any overlap invalidates the run. This is the trap that made the N=10 features
  descriptive-only (`12b` §2.5) and it is not repeated here.

## 6. Protocol and matching rules (frozen values)

`T = 24`, `seq_len = 96`, `label_len = 48`, 70/10/20 chronological split, seeds 2021/2022/2023,
Adam, lr 1e-4, batch 32, ≤30 epochs with patience 5, cosine schedule, MSE loss, paired bootstrap
4000 resamples for every reported difference, plus paired win rate. Metrics: MSE, MSPE, 95 % error
quantile, win rate; the volatility stratification (High = Q4+Q5, Low = Q1+Q2) is reported as the
frozen strata, and the twelve structural states are used as **evaluation strata only** (they are not
training multipliers — corrected in `17a` §2.1).

**Matching:** total parameters equal within ±2 % across A0–A5; identical step counts and batch size;
FLOPs per forward pass reported; **state explicitly which width the matching holds at** (the linear
map is parameter-matched only at width 23 — `17a` §2.2).

## 7. Pre-specified decision rules (written before any run; not transplantable from Gate A)

1. **Margin:** A3 must beat the best ordinary arm by ≥ the median of the bootstrap 95 % CI half-width
   observed for the same metric on the same assets in E4/E5 — i.e. the margin must exceed the
   measurement noise actually observed in this project, not a convenient constant.
2. **Seed consistency:** the sign must hold in **all three** seeds; a single-seed win is recorded as
   *unstable*.
3. **No single-asset wins:** the effect must appear on **≥ 2 of the pre-registered assets**.
4. **Tolerance for the removal test:** A4 is "no different" if its difference from A3 lies inside the
   same CI-based band; in that case H-mech is falsified.
5. **Feature test:** if A5 is within that band of A3, the NS-inspired bias is recorded as *not adding
   anything beyond a supplied feature*, and the direction is retired.
6. **Compute control:** if the advantage vanishes when steps and FLOPs are matched, it is attributed
   to compute, not to structure.
7. **Negative results are reported with the same prominence** as positive ones.

## 8. Resource envelope

| Step | Count | Measured cost | Estimate |
|---|---|---|---|
| **Step 0 — diagnostic (no training):** does a pre-fixed causal `q`-stratification align with the nonlinear advantage on the existing assets? | 0 trainings | — | minutes (post-processing of stored predictions) |
| **Step 1 — arms A0–A5 × 3 seeds × the pre-registered cohort** | 6 arms × 3 seeds × N assets | framework-scale run ≈ 15–17 min effective at 2-way parallelism (`17a` §2.1 context) | for N = 15–20: **6×3×15–20 = 270–360 runs ≈ 4–6 days wall-clock** on this host |

**Step 0 first.** It is free relative to Step 1 and can kill the direction before any training.
**Operational note:** long campaigns on this host must run under the idempotent self-healing launcher
and be verified by process CPU time (detached batches died silently three times on 2026-09-13).

## 9. What this design does **not** claim

- It does not claim the NS frame is correct, useful, or physically apt for markets.
- It does not claim `q_t` is novel (it is a local variance; ordinary convolution computes it).
- It does not authorise implementation, and it is not an eligibility upgrade: Gate A's
  `CONDITIONAL` cap still applies, and eligibility remains **not** evidence.
- It does not port any Gate-A statistical criterion into the operator test.
