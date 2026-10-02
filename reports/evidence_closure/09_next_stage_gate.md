# 09 · Next-stage Gate (single decision)

Per spec §18 and §19. This is the one and only gate: it decides **eligibility to enter an
NS-inspired nonlinear operator benchmark**. It does **not** implement anything, and this stage
stops here.

## Gate inputs

| Condition | Status |
|---|---|
| Path B still holds (operator form is the open question, not routing) | ✅ yes |
| Reproducible evidence of conditional value of the nonlinear branch | ✅ yes (ETH 36/36 at two widths × three seeds; GSPC framework-level across two seeds) |
| Asset-level heterogeneity reasonably clear | ✅ yes (ETH nonlinear / BTC linear / GSPC parity, capacity-controlled) |
| Capacity confound known and controllable | ✅ known; **controllable only by protocol** (capacity-matched arms) — not yet controlled in the asset-level analysis |
| Structural candidate at least partially supported | ⚠️ partial (two LOO-stable features at p<0.05, N=10, one seed per asset) |

## Decision

```
NS_BENCHMARK_ELIGIBLE = CONDITIONAL
```

**Rationale.** The *direction* of every step is consistent and the negative result (no robust
sample-level routing) is strong; but the *statistical* basis of the structural part is not yet
sufficient to commit compute to a benchmark:

- the asset-level analysis rests on **one seed per asset** (GSPC two of three planned);
- the candidate associations have **N = 10** and the underlying interaction sign **spans zero**;
- the capacity-controlled operator preference exists for **three assets only**;
- the "conditional value" evidence comes from a **generic MLP proxy**, not from a structured
  nonlinear operator — so nothing yet shows that a structured operator would exploit the
  heterogeneity.

Per spec §18-C this is exactly the "direction consistent, statistical support not yet sufficient"
case, so the gate reads **CONDITIONAL**, and the next stage may only be proposed with the
additional validation below.

## Conditions that would upgrade the gate to YES

1. **Seed coverage:** ≥ 2 seeds per asset in the asset-level analysis (or ≥ 6 assets with 2 seeds),
   reported as seed-level robustness, not just merged values.
2. **Capacity control beyond the proxy:** extend the width sweep (23/64/128) to the remaining
   assets — at minimum to SOX, BABA, NVO (the extremes in `|ACF1|`), so that "ETH nonlinear /
   BTC linear" is not a two-asset pattern.
3. **Pre-registered confirmatory test of the structural candidate:** a held-out set of assets or a
   multiplicity-controlled test of `|ACF1|` and realized volatility against the interaction sign.
4. **Framework-level capacity-matched comparison** of a structured nonlinear operator versus the
   generic one, so the benchmark tests *operator form* rather than *operator presence*.

## Mandatory framing for any downstream document

> **Do not interpret eligibility as evidence that NS works.**

The gate asserts only that the *question* — whether a structured (temporal NS-inspired) nonlinear
operator is worth testing — is now well-posed and supported by a reproducible evidence chain. The
answer to that question is unknown; the protocol also requires that negative evidence be reported
with equal prominence.

**Stop condition honoured:** this stage ends here. No router, gate, MoE, attention, pressure term
or NS module was implemented, and no next-stage experiment was started.
