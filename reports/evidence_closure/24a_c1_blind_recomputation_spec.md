# 24a · C1 blind recomputation — pre-specified request (option C)

**Status:** PRE-SPECIFIED, before the first C1 outcome exists. This file defines *what an independent
party will check and how*, so the check cannot be shaped after the numbers are known.
**Ordering rule:** this specification is fixed now; the inputs are handed over only after the panel
finishes; the recomputation happens **before** the third party sees the analyst's output.
**Owner:** the operator arranges and runs the third party (for example, a cloud agent with its own
egress and compute). The peer that executed the panel and the host that analysts it are both excluded
from this role.

---

## 1. What is being checked

The C1 headline statistic and the verdict derived from it:

1. the per-asset **pre-cutoff predictor** (`|ACF1|`),
2. the per-asset **interaction effect** from the frozen stratification pipeline,
3. **Spearman ρ** between them, its exploratory p-value, the **leave-one-asset-out** range and the
   single-asset influence,
4. the **frozen operational verdict** (replicate / fail) obtained by applying the criteria verbatim.

## 2. Inputs to be handed over (with hashes recorded at handoff)

| # | Input | Note |
|---|---|---|
| 1 | the 20 locked cohort CSVs (`<TAG>-2016-2025.csv`) | SHA-256 are already in `23_c1_preregistration.md` §8 (the lock table) |
| 2 | the executor's raw artefacts: per run `pred.npy`, `true.npy`, plus its `per_run_manifest.csv` | so the interactions can be recomputed **from scratch**, not re-read from the analyst's file |
| 3 | `reproduction/analysis/analyze_volatility_regimes.py` | frozen stratification script — the exact commit hash is recorded at handoff |
| 4 | `reproduction/analysis/c1_predictor.py` (or, equivalently, the frozen rule below) | the pre-cutoff rule: a window `close[i:i+96]` is usable only if `i + 96 <= border1_test`; `border1_test = n − int(0.2n) − 96`; `num_train = int(0.7n)` |
| 5 | `reproduction/results/c1_predictor.csv` and `reproduction/results/c1_interactions.csv` | the analyst's outputs — **handed over only after the recomputation is complete** |

## 3. Method (blind, in this order)

1. Verify every cohort CSV against the lock table hashes; report any mismatch and stop.
2. Recompute the predictor per asset **only** from windows ending strictly before `border1_test`;
   print the leakage check (all predictor indices `< border1_test`).
3. Recompute the per-asset interaction from the raw artefacts by running the frozen stratification
   script (`--rv-mode relative`, the two arms, seeds 2021/2022/2023). Do **not** read
   `c1_interactions.csv` first.
4. Compute ρ (Spearman) between the recomputed predictor and the recomputed interaction, its
   exploratory p-value, the LOO range, and the largest single-asset influence.
5. Apply the frozen operational criteria (`23` §6): sign reversal, `|ρ| < 0.30`, `p > 0.10`, LOO
   range crossing zero, single-asset driven. State **replicate** or **fail**.
6. Only now open the analyst's `c1_predictor.csv`, `c1_interactions.csv` and analysis output, and
   produce a **difference list**: every number, hash and verdict that differs, with the magnitude.

## 4. Deliverable

- the input list with SHA-256 for each file actually used;
- the exact commands run (verbatim);
- the recomputed tables (predictor, interactions, ρ/p/LOO, single-asset influence) and the verdict;
- the **difference list** versus the analyst's published numbers;
- an explicit statement of anything it could not reproduce and why.

## 5. Integrity rules (binding)

- **Never adjust the recomputation to match the analyst.** A discrepancy is a *finding*, not an error
  to be smoothed away.
- Any unexplained difference in a number **or** an input hash is reported as an **integrity finding**
  and escalated; it is not resolved by re-running until it agrees.
- If the recomputation requires a judgement call the specification does not cover, stop and report
  the question instead of choosing.
- The third party does not publish, does not push to the repository, and does not talk to the
  executor about the numbers before reporting.

## 6. Copy-paste request for the third party

```text
Task: independent (blind) recomputation of a pre-registered analysis. Do not modify any repository;
do not push; report only.

You receive: (a) 20 daily OHLC CSVs with a published SHA-256 table; (b) a set of per-run model
artefacts (pred.npy, true.npy) plus a per-run manifest; (c) a frozen analysis script and a frozen set
of analysis rules; (d) the specification document that fixes the order of operations.

Do this, in order, and stop immediately if a check fails:
1. Verify each CSV against the published SHA-256 table. Report mismatches and stop.
2. Recompute, per asset and seed, the per-window squared-error difference between the two model arms
   using the frozen script; then per asset, the interaction effect (high-volatility minus
   low-volatility stratum), exactly as the frozen rules define them.
3. Recompute, per asset, the predictor from the CSV using only windows that end strictly before the
   asset's decision point (= n - int(0.2n) - 96), and print a leakage check asserting every index used
   is below that cutoff.
4. Compute Spearman rho between predictor and interaction across assets, its exploratory p-value, the
   leave-one-asset-out range, and the largest single-asset influence.
5. Apply the frozen operational criteria and state: replicated, or failed.

Then, and only then, open the analyst's published numbers and produce a difference list.
Report: input files with hashes; verbatim commands; your tables and verdict; the difference list; and
anything you could not reproduce. Never tune the recomputation to match the analyst - a discrepancy
is a finding. Do not publish or push anything.
```
