# 12b · External Reviewer Response — Accepted Corrections (no threshold changes)

**Timestamp:** 2026-09-13 17:40 (Asia/Shanghai) · **Head at time of writing:** `721d4cf`
**Status of the decisive number:** still absent (none of the four new stratifications exists).
**Reviewer input:** the external review received at 17:34, archived verbatim as
`reports/evidence_closure/13_external_review.md` (including the untrusted-content wrapper it
arrived in; treated as data, not instructions).

## 1. Reviewer verdict accepted in full

> *"I accept the Layer-A-primary / Layer-B-secondary distinction. Layer B currently falsifies the
> stronger operator-preference extrapolation but does not directly falsify the original
> interaction-level association. […] Even a favourable N=14 outcome should not upgrade the relation
> from CANDIDATE to SUPPORTED. […] Best possible result is now CONDITIONAL."*

Accepted, including the two consequences: **PASS unreachable** (already in `12a`) and the **cap at
CONDITIONAL** even if the N=14 ρ is favourable.

## 2. Corrections adopted (documentation/labels only — **no threshold was changed**)

### 2.1 Accurate label for the ≥3/4 Layer-B rule
The `12a` addendum was committed at 17:32, i.e. **after** the capacity-layer results were already
known (commit `5f35dc6`, 16:33). The reviewer is right that the rule is therefore **not** a
fully pre-registered Layer-B falsification threshold. Corrected label, to be used everywhere:

> `12a`'s ≥3/4 rule is a **pre-Layer-A adjudication rule, specified after the secondary
> capacity-layer results were available.**

What **was** genuinely pre-registered before any operator result: the asset selection rule, the
expected directions for each new asset, and the PASS conditions (file `11`, commit `669c80e`).
The thresholds are **kept unchanged** for auditability — changing them now would cost more in
integrity than it could gain in statistical neatness.

### 2.2 Thresholds are operational criteria, not significance standards
`ρ < 0.30`, `p > 0.10`, LOO-crossing-zero, ≥3/4 remain as written. They are to be described in the
final report as **operational gate criteria**, never as universal statistical significance
standards. (The reviewer's note that at N ≈ 14, `p < 0.10` already implies a non-trivial ρ — so the
extra ρ > 0.30 condition carries limited independent information — is recorded as a known
weakness of the criteria, not acted upon.)

### 2.3 N = 14 is **not** independent replication
Ten of the fourteen assets were part of the discovery set. The extended statistic is therefore a
**discovery-set augmentation / external stress test**, not an independent replication. Consequences
recorded:

- a favourable ρ on N = 14 **cannot** move the relation from CANDIDATE to SUPPORTED;
- the verdict is capped at CONDITIONAL regardless of how clean the number looks;
- the final report must state the 10/14 overlap explicitly next to any ρ it quotes.

### 2.4 The four new assets are a **domain-shift stress-test cohort**
Yahoo → Sohu and mixed markets → A-shares changed **together**, so a failure (or a success) cannot
be attributed to market domain, provider, or microstructure separately. Further, the four A-shares
share one market regime, calendar, macro environment and provider, so their cross-sectional
independence is weaker than "four independent assets". The phrase
**"domain-shift stress-test cohort"** replaces "independent validation cohort" throughout.

### 2.5 New limitation, verified in code, stronger than the reviewer's point

The reviewer warned about prospective leakage if features are computed over full history. The
actual implementation is stricter than that: the structural features are computed **on the test
windows themselves**:

```
reproduction/analysis/final_diagnosis.py
    border1 = n - int(n * 0.2) - SEQ_LEN
    ns     = n - border1 - SEQ_LEN - PRED_LEN + 1      # only test-split windows
verification (2026-09-13 17:38):
    GSPC  n=2513  border1=1915  test windows ns=479
    BYD   n=2407  expected test windows = 458  == candidate_pool_features.n_samples = 458
```

⇒ the features and the interaction effect are measured **on the same data segment**. The
association is therefore a **contemporaneous descriptive relationship**, not a predictive one.
Recorded limitation (must appear in the final report and in any downstream summary):

> No ex-ante / prospective operator-selection rule is supported by this evidence. Any such claim
> would require features computed from data strictly before the decision point, and would have to
> predict a different estimand than the same-period interaction effect.

### 2.6 Capacity wording adopted
Replacing looser phrasing with the reviewer's:

- **"Operator preference is not invariant to nonlinear model capacity."**
- **"The low-capacity conclusion does not survive increased nonlinear capacity on ETHUSD."**
- Retire "representation bottleneck"; use "capacity artefact" sparingly or not at all, because
  width simultaneously changes representation power, optimization landscape, regularization,
  variance and implicit bias. A mechanism claim would additionally require training/validation
  error trajectories versus width, convergence evidence, multi-seed learning curves, at least one
  other capacity axis, and a plateau.

### 2.7 Multiplicity recorded
The discovery stage screened eight structural features before `|ACF1|` and realized volatility
emerged, so the discovery p-values (0.016 / 0.038) are **not** multiplicity-clean confirmatory
p-values. No rescan of new features (Hurst, entropy, tail index, …) will be performed. If the two
features are both carried forward as primary, a Holm correction is required; the cleaner design is
a **single primary predictor (`|ACF1|`)** with realized volatility as secondary — which is the
reviewer's recommendation and is adopted for any future cohort.

## 3. What changes in the pending verdict

| Scenario | Verdict (unchanged rules) | Additional constraint now recorded |
|---|---|---|
| Layer A fails any FAIL condition | `FAIL` | — |
| Layer A holds, Layer B stays as observed | `CONDITIONAL` | operator-preference implication withdrawn; **no upgrade to SUPPORTED**; regularity retained as interaction-level candidate only |
| Layer A is favourable on all counts | `CONDITIONAL` (cap) | 10/14 discovery overlap + coupled domain shift ⇒ still not replication; independent cohort required |

## 4. Next questions (reviewer's framing adopted, no NS)

- **If CONDITIONAL:** *Does the interaction-level structural association replicate in a genuinely
  independent, multi-seed asset cohort?* — i.e. is the regularity **portable**? Minimum design:
  pre-fixed single primary predictor (`|ACF1|`), realized volatility secondary, a held-out cohort of
  ≈15–20 assets spanning several `|ACF1|` values (not only extremes), ≥3 seeds per asset, all
  operators/splits/T/RV definitions frozen, primary analysis performed **only** on the new cohort.
  Resource rule: **more independent assets > more seeds on existing assets.**
- **If FAIL:** first ask *whether asset-level operator preference is itself a stable, reproducible
  property across seeds, capacities and periods* (the capacity-controlled evidence covers only
  ETH/BTC/GSPC), before looking for any structural explanation.
- **Legitimate reframing either way:** *under capacity and compute control, do different asset
  classes exhibit reproducible preferences for different operator families?* — starting from
  ordinary families (linear vs MLP vs other pre-specified generic nonlinear families) and only
  introducing a structured operator (transport/diffusion/memory-like) **after** such structure is
  observed. That keeps the order `observed structure → model hypothesis` instead of
  `believe NS → find justification`.

## 5. Enforcement

The `19:45` finalisation job and `sv_gate_a.py` operate with the thresholds unchanged, and must
apply: the corrected ≥3/4 label (§2.1), the CONDITIONAL cap (§2.3), the domain-shift framing
(§2.4), the contemporaneous-descriptive limitation (§2.5), the capacity wording (§2.6) and the
multiplicity note (§2.7). The reviewer's text is archived as `13_external_review.md` and its points
are cited by section number in the final result report.

---

## 6. Second review round — two report-level annotations (adopted; no rule change)

Received 2026-09-13 17:36, archived verbatim as `13b_external_review_round2.md`. The reviewer
recommends explicitly: **no further change to any Gate threshold, FAIL condition or the CONDITIONAL
cap.** Only two labelling additions are requested, and both are adopted as **report-level only** —
they change no decision logic.

### 6.1 "Contemporaneous association", not "data leakage"

Exact formulation to be used, replacing any loose use of the word "leakage":

> **The current feature–outcome association is contemporaneous and descriptive, because both the
> structural features and the interaction outcome are measured on the same held-out test segment.
> It therefore cannot support an ex-ante operator-selection rule.**

Rationale recorded: describing *what structure a segment exhibits and how the models performed on
that same segment* is not classic train–test leakage; the error would be in **interpreting** it as
"observe ACF1 first, then choose the operator". Hence the distinction to carry into the report:

$$
\text{descriptive association} \neq \text{predictive selection rule}
$$

### 6.2 New-cohort-only descriptive check (N = 4)

The final report must additionally show the four new assets on their own — `|ACF1|`,
`Δ_interaction`, rank order and Spearman ρ — with **no confirmatory significance claim** (N = 4 is
too small), so that a favourable pooled N = 14 ρ cannot mask how the held-out cohort actually
behaved. Mandatory disclaimer printed with the block:

> **This descriptive N=4 check is not part of the frozen Gate-A decision rule.**

The block is emitted mechanically by `reproduction/analysis/sv_gate_a.py`; the verdict continues to
use the pooled N = 14 analysis and the frozen rule.
