# NS-Frame Hypothesis Charter (thinking-only, not evidence)

**Status:** preparatory conceptual work. **Not** an experiment authorisation. **Not** evidence.
Nothing here changes the frozen Gate rules, the ongoing structural validation, or the
`CONDITIONAL` cap on `NS_BENCHMARK_ELIGIBLE`.
**Purpose:** give an outside thinker (e.g. ChatGPT) a disciplined frame in which to develop the
NS *idea*, so that whatever comes out is eventually **testable and falsifiable**, instead of
drifting into complexity-first modelling.

---

## 1. What "NS" means here (agreed interpretation)

"NS" is a **broad conceptual frame**: a family of dynamical-systems intuitions about **transport,
mixing, forcing, invariance, and multi-scale interaction** over time. It is explicitly **not** a
commitment to the Navier–Stokes equations, not a claim that markets are fluids, and not a
requirement to copy any formula.

Two layers, deliberately separated:

| Layer | Allowed to be | Must be |
|---|---|---|
| **Inspiration layer** | loose, metaphorical, exploratory | clearly marked as analogy |
| **Claim layer** | — | falsifiable, capacity-matched, pre-specified |

> **The frame may be loose. Any resulting claim may not be.**
> Analogy is a generator of hypotheses; it is never evidence.

## 2. Inspiration layer — candidate frames (analogy only)

Each row is a *way of thinking*, with the kind of behaviour it would predict if the analogy were
useful. None of it is asserted to be true.

| Frame | What it suggests an operator should do | Naive failure mode |
|---|---|---|
| **Advection / transport** | learn a *state-dependent shift or warp* of information along time, not just a fixed convolution | shifts in price series may be artefacts of index/level, not dynamics |
| **Diffusion / mixing** | learn a *state-dependent smoothing* (volatility-linked diffusivity) instead of a fixed moving average | over-smoothing destroys jump information |
| **Operator splitting / composition** | compose several *cheap* operators (advection ⊕ diffusion ⊕ forcing) rather than one big block | composition can be an expensive re-parameterisation of an MLP |
| **Conservation / invariants** | preserve some quantity across the forecast window (scale, sum, rank structure) | markets have no conserved quantity; a fake invariant hurts |
| **Forcing / sources** | separate *externally driven* movement (news, regimes) from *self-propagating* movement | external drivers are unobserved here |
| **Multi-scale cascade** | couple fast and slow components rather than one flat receptive field | multi-scale = a known trick, not an NS-specific claim |
| **Viscosity ↔ regime** | a smoothness parameter that *changes* with market regime (calm ↔ turbulent) | our own evidence says the linear/nonlinear advantage is capacity-sensitive, so a regime knob may just re-encode capacity |
| **Shock / discontinuity** | handle jumps explicitly rather than smoothing across them | jump ratio is *constant* in our data at the pre-registered threshold — no variation to exploit |
| **Particle / Lagrangian view** | treat tokens/samples as transported particles (attention-like) | already exists as attention; must show what NS adds |
| **Optimal transport of distributions** | adapt to *distribution shift* of returns rather than to point forecasts | a different problem formulation, not a drop-in operator |

## 3. Bridge requirement (the part that makes it science)

Every frame from §2 must be paired with **all four** of the following before it can be considered
for any experiment:

1. **Predicted structural signature** — a concrete property of the data that this operator family
   should exploit (e.g. "high state-dependent shift asymmetry", "volatility-linked local smoothness").
2. **A prospective detector** — how that signature is measured using **only data strictly before
   the decision point**. (Our current features are contemporaneous with the outcome, so they cannot
   support an ex-ante rule — that limitation must not be repeated.)
3. **A capacity-matched baseline** — the operator compared against a generic alternative with the
   **same parameter budget**, same protocol, same seeds (linear map and MLP at matched width, as
   already done here).
4. **A pre-registered falsification criterion** — what result would count as *no advantage*,
   decided before running.

## 4. Consistency with the negative evidence already obtained

Any proposal must be able to account for these results, not wish them away:

- **Capacity sensitivity:** the nonlinear branch's apparent disadvantage at width 23 does **not**
  survive increased capacity on ETHUSD (36/36 reversal). Any "structured operator helps" claim
  must therefore be tested **capacity-matched**, and its gain must exceed a generic-capacity
  explanation.
- **Asset dependence:** operator preference differs by asset (ETH nonlinear / BTC linear / GSPC
  parity). A universal operator claim is already excluded.
- **No sample-level routing evidence:** the state → operator mapping is unsupported; proposals that
  require per-sample switching inherit a burden of proof.
- **Contemporaneous, not prospective:** past features explain the present outcome only
  descriptively. Prospective claims require pre-decision features.
- **Jump ratio carries no information** in this dataset at the pre-registered threshold, so
  discontinuity-based frames have nothing to exploit here (without changing thresholds, which is
  forbidden).

## 5. Deliverables of the preparatory phase (paper only)

1. **Frame map** (extend §2): operator families, each with its predicted signature.
2. **Signature → detector table**: for each family, a *prospective* measurable detector, plus the
   cheapest way to compute it.
3. **Minimal falsification design per family**: smallest capacity-matched experiment that could
   kill it, with pre-registered verdict criteria.
4. **Negative-evidence compatibility table**: how each family explains §4's results.
5. **Failure-mode register**: where the metaphor actively misleads (no compressibility analogue,
   discrete trades ≠ continuum, non-stationarity breaks conservation, level/scale artefacts
   masquerading as transport).
6. **Priority ordering**: which family is the *cheapest to falsify* rather than the most
   attractive — that is the one to test first.

## 6. Prohibitions for this phase

- ❌ No experiment is authorised by this charter; the gate rule and the CONDITIONAL cap stay frozen.
- ❌ No code changes in the project repository, no NS module, no transport/diffusion layer.
- ❌ No claim of the form "markets are fluids" / "Navier–Stokes governs financial dynamics" except
  as an explicitly labelled analogy.
- ❌ No use of this document to justify skipping independent prospective structural validation.
- ❌ No metric-shopping, no threshold changes, no post-hoc selection — the existing discipline
  applies to any later test.
- ✅ Everything produced here is labelled **"hypothesis design, not evidence"**.

## 7. Open questions handed to the thinking phase

1. Which NS-flavoured property is **cheapest to falsify** on financial data, and what is its
   prospective detector?
2. Is there an NS-flavoured operator whose *inductive bias* cannot be reproduced by a
   capacity-matched MLP or attention block (i.e. what is genuinely new)?
3. Which frames should be **abandoned up front** as unfalsifiable or already covered by existing
   architectures?
4. Given the negatives in §4, what is the *narrowest* claim worth testing — i.e. an asset- or
   regime-specific statement rather than a universal one?
5. If no frame survives questions 1–3, what is the honest conclusion about the NS frame itself?
