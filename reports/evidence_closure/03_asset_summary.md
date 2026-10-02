# 03 · Asset-level Summary

Per spec §14. **Two layers must not be conflated:** *asset-level* operator preference
(measured under capacity control in the proxy experiment, E4/E5) and the *interaction sign*
(measured on the framework's two branches, E3/E6). They are different quantities.

## A. Asset table (N = 10)

| Asset | Nonlinear preference (capacity-controlled, width 128) | Interaction sign | Interaction 95% CI | Direction vs significance | Capacity sensitivity | Structural association (RV / \|ACF1\|) | LOO status |
|---|---|---|---|---|---|---|---|
| GSPC | near parity (+0.0061) | negative | [−0.02719, −0.00876] | directionally negative **and statistically supported** | high (0.060→0.025→0.006 mean ΔMSE) | 0.0080 / 0.076 | stable |
| BTCUSD | linear (+0.2998, all capacities) | positive | [+0.00615, +0.02378] | directionally positive **and statistically supported** | none (0.178→0.189→0.300, 0/36) | 0.0254 / 0.084 | stable |
| ETHUSD | **nonlinear** (−0.0321) | negative | [−0.01881, +0.00033] | directionally negative **but statistically inconclusive** | **high — flips at width 64** | 0.0358 / 0.053 | stable |
| EURUSD | n/a (not run) | negative | [−0.16421, −0.11061] | directionally negative **and statistically supported** | n/a | 0.0042 / 0.066 | stable |
| USDJPY | n/a (not run) | negative | [−0.02606, +0.00226] | directionally negative **but statistically inconclusive** | n/a | 0.0062 / 0.072 | stable |
| SOX | n/a (not run) | positive | [−0.00527, +0.01435] | directionally positive **but statistically inconclusive** | n/a | 0.0211 / 0.115 | stable |
| DJI | n/a (not run) | negative | [−0.01939, +0.00266] | directionally negative **but statistically inconclusive** | n/a | 0.0075 / 0.074 | stable |
| BABA | n/a (not run) | positive | [+0.02155, +0.05986] | directionally positive **and statistically supported** | n/a | 0.0255 / 0.082 | stable |
| NVO | n/a (not run) | positive | [+0.04636, +0.14937] | directionally positive **and statistically supported** | n/a | 0.0220 / 0.156 | stable |
| TM | n/a (not run) | negative | [−0.02759, +0.01129] | directionally negative **but statistically inconclusive** | n/a | 0.0174 / 0.069 | stable |

**Counts:** negative 6 · positive 4 · statistically supported 5 (GSPC, EURUSD negative;
BTCUSD, BABA, NVO positive) · inconclusive 5.

**GSPC** aggregates seeds 2021 and 2022 into one asset (pre-stated rule); both seeds are
individually supported and sign-consistent, and seed-level detail is reported separately in
`04_structural_association.md`.

## B. What the table does and does not support

- **Does support:** genuine *asset-level* operator heterogeneity — under capacity control ETHUSD
  prefers the nonlinear operator in 36/36 observations while BTCUSD prefers the linear one in
  36/36 at every width, with GSPC near parity.
- **Does not support:** a *sample-level* (within-asset state → operator) mapping — see `06`.
- **Coverage caveat:** the capacity-controlled preference exists only for three assets (GSPC,
  BTCUSD, ETHUSD). For the other seven only the interaction sign is available. Declared, not hidden.
- **Do not** read a "best model on average" ranking out of this table: the quantity is the
  *conditional* value of the nonlinear branch, and its sign is what matters.
