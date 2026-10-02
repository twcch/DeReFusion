# 12a · Gate A Decision-Rule Addendum (locked BEFORE the N=14 result exists)

**Timestamp:** 2026-09-13 17:32 (Asia/Shanghai)
**Commit at lock time:** `7806aa4` (repository head when this file was written)
**Evidence that the decisive number does not exist yet:**

```
$ ls reproduction/results/volatility_stratification_{BYD,BOE,EASTMONEY,YANGHE}_relative_s2021.json
  -> none of the four files exists (framework runs still in flight, started 17:14)
```

Therefore N = 14 is still 10; ρ(|ACF1|, Δ_interaction) on the extended set has **not** been
computed. This addendum is written *before* that number exists, on purpose.

## 1. Why this addendum exists

The pre-registration (`11_structural_validation_preregistration.md`) fixed the assets, the
selection rule and the expected directions, and listed six PASS conditions. It did **not** state
how the two evidence layers combine into the final verdict. Deciding that *after* seeing the
interaction-level result would leave room for motivated reasoning, so the combination rule is
frozen here.

## 2. The two layers are not the same claim

| Layer | Quantity | What it can test |
|---|---|---|
| **A — interaction level** (primary) | ρ between asset features and **Δ_interaction** (ΔMSE_high50 − ΔMSE_low50), recomputed on the extended asset set | whether the *regularity itself* (feature ↔ volatility-conditioned interaction sign) survives the addition of new assets. This is the quantity the regularity was fitted on. |
| **B — operator-preference level** (pre-registered secondary) | mean ΔMSE of the proxy operators at widths 64 and 128, per new asset, 3 seeds | whether the *derived implication* ("low \|ACF1\| → nonlinear-favoured, high \|ACF1\| → linear-favoured") holds. The regularity never asserted this implication at the ΔMSE level; it was a *predicted consequence*, locked in advance. |

Layer A tests the claim; Layer B tests a consequence of the claim. A consequence can fail while the
claim survives (the implication was an extrapolation), and a claim can fail while a noisy
consequence happens to look right. They must therefore be weighted differently, **in advance**.

## 3. Frozen combination rule

Applying the pre-registered PASS conditions to the data that already exists:

- PASS condition (1) is *"the new assets' operator preference is broadly consistent with the
  association"*; the observed state is **1 of 4** (BYD matches; BOE and EASTMONEY are mixed/tie;
  YANGHE — the highest \|ACF1\| — is reversed with per-seed-consistent signs).

⇒ **`PASS` is not reachable.** It is removed from the outcome space by the pre-registered rules
themselves, not by re-interpretation after the fact.

The remaining verdicts are decided as follows:

### FAIL if **any** of the following holds
1. **A flips or collapses:** ρ(\|ACF1\|, Δ_interaction) on the extended set changes sign relative to
   +0.733, **or** drops below +0.30, **or** its exploratory p rises above 0.10, **or** the
   leave-one-asset-out range crosses zero.
2. **B becomes systematic:** **≥ 3 of 4** new assets contradict the pre-registered operator
   preference *with per-seed-consistent mean ΔMSE signs at both widths* (a single reversed asset is
   not enough, precisely because Layer B is a derived implication).
3. **Capacity contradiction:** for any new asset the sign of the mean ΔMSE differs between width 64
   and width 128 (a preference that is capacity-dependent cannot support a structural claim).
4. **Single-asset dependence:** the extended ρ is driven by one asset (drop-one sign flip).
5. **Integrity finding:** a protocol inconsistency, leakage, or post-hoc selection is discovered.

### CONDITIONAL if
- Layer A **holds** (sign preserved, LOO-stable, p < 0.10, not single-asset driven), and
- Layer B stays **as observed** (mixed: 1/4 match with one seed-consistent reversal), and
- no capacity contradiction and no integrity finding.

→ Outcome: keep the regularity as a **candidate at the interaction level only**, and
**explicitly withdraw the operator-preference implication** (i.e. the regularity may describe how
the interaction sign co-varies with asset structure, but the evidence does **not** support using it
to choose an operator per asset). Eligibility for any structured-operator benchmark stays
CONDITIONAL and additionally requires an independent, same-source, multi-seed replication.

### PASS (narrowed) if
- All six pre-registered PASS conditions hold. **Currently unreachable** per §3 above; listed only
  for completeness of the rule.

## 4. Weighting rationale recorded before the result

- Layer B's new assets are **all A-shares from a different data source (Sohu)** than the original
  ten (Yahoo). A systematic market/microstructure difference could by itself produce
  operator-preference mismatches; that is exactly why a *single* reversed asset is not treated as
  decisive and why the threshold is set at ≥ 3 of 4.
- This cuts both ways: it also means a *passing* Layer B would have been weak evidence, which is
  consistent with keeping Layer A primary.
- Layer A's own confounds (1 seed per asset, source change on the added assets, |ACF1| possibly a
  proxy for liquidity/microstructure) are recorded in the pre-registration and must appear in the
  final report.

## 5. Enforcement

The verdict is produced mechanically by `reproduction/analysis/sv_gate_a.py` with these thresholds;
no threshold may be adjusted after the N=14 numbers exist. The script's outputs and this addendum
are committed together so the lock is auditable.

---

## 6. Accuracy correction (2026-09-13 17:40, external review)

**No threshold in this file was changed.** One label was corrected for accuracy, following the
external review archived as `13_external_review.md` (see `12b_reviewer_response_and_corrections.md`
§2.1):

- This addendum was committed at 17:32, i.e. **after** the capacity-layer (Layer B) results were
already available (`5f35dc6`, 16:33). The ≥3/4 Layer-B rule is therefore **not** a fully
pre-registered Layer-B falsification threshold. Its accurate description is:

  > a **pre-Layer-A adjudication rule, specified after the secondary capacity-layer results were
  > available.**

- What *was* pre-registered before any operator result (file `11`, commit `669c80e`): the asset
  selection rule, the per-asset expected directions, and the PASS conditions.
- All thresholds (`ρ < 0.30`, `p > 0.10`, LOO crossing zero, ≥3/4) are retained unchanged and are to
  be described as **operational gate criteria**, not as universal statistical significance standards.
