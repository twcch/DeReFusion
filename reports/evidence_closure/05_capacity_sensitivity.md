# 05 · Capacity Sensitivity

Per spec §8 and §9-Q5. The capacity experiment varies **one** quantity — the nonlinear
operator's hidden width — and holds everything else identical (data, features, splits, optimizer,
learning rate, batch size, epochs, early stopping, seeds, evaluation protocol, and the
pre-registered structural states).

## A. Mean ΔMSE by capacity (negative = nonlinear better)

| Hidden width | MLP params | GSPC | BTCUSD | ETHUSD |
|---|---|---|---|---|
| 23 | 9,431 | +0.05982 (0/36 prefer nonlinear) | +0.17832 (0/36) | +0.01454 (12/36) |
| 64 | 26,200 | +0.02503 (7/36) | +0.18884 (0/36) | **−0.02807 (36/36)** |
| 128 | 52,376 | +0.00606 (8/36) | +0.29983 (0/36) | **−0.03205 (36/36)** |

(36 = 12 structural states × 3 seeds.)

## B. The three pre-registered checks

1. **Sign reversals across widths:** 25 state-level reversals in total — **ETHUSD 23**, **GSPC 8
   (all of them at seed 2023 only → seed-unstable)**, **BTCUSD 0**.
2. **Spread across states:** the range of ΔMSE across states **shrinks** with capacity for GSPC
   (0.106 → 0.098 → 0.060) and ETHUSD (0.089 → 0.048 → 0.040); the BTCUSD increase
   (0.540 → 0.655 → 1.120) is driven by the magnitude of a single high-error state, not by a
   preference change.
3. **Per-seed direction consistency:** degrades with capacity — 31/36 → 29/36 → 28/36.

## C. Verdict and mandatory caveat

- **`CAPACITY_SENSITIVE = YES`** (magnitude: **HIGH** for ETHUSD, MODERATE for GSPC, NONE for
  BTCUSD).
- The nonlinear preference of ETHUSD is **asset-wide** at sufficient capacity: **all twelve
  states** reverse, not a subset. That is asset-level heterogeneity, not state-level specialization.
- **Mandatory conservative wording (spec §9-Q5):** the earlier uniform "linear is better" readings
  must **not** be attributed wholesale to a *representation* bottleneck. What the evidence
  supports is the weaker statement:

  > *capacity sensitivity is a major uncertainty / confounder in the present evidence.*

  In particular, the width-23 result for ETHUSD is **not** evidence that "the nonlinear operator
  is useless"; it is evidence that this particular capacity was insufficient for this asset.
  Conversely, the fact that GSPC's mean ΔMSE collapses to ≈ +0.006 at width 128 while BTCUSD's
  grows to +0.300 shows that capacity does not simply "fix" a universal deficit.

## D. Consequence for the routing question

Capacity changes the *level* of the preference far more than its *state-dependence*: no state
gains a stable, seed-consistent advantage over the linear operator at any width. This is the
quantitative basis for the "no robust sample-level routing evidence" finding in `06`.
