# 07 · Evidence Matrix

Per spec §12. Status vocabulary: **Supported** · **Partially supported** · **Directional only** ·
**Inconclusive** · **Not supported**.

| Research Question | Existing Evidence | Direction | Statistical Support | Cross-Asset Stability | LOO Stability | Confidence | Status |
|---|---|---|---|---|---|---|---|
| R1 Nonlinear computation has real value | additive fusion > linear base (GSPC +11.1% / +15.4% over two seeds; BTC +2.4%; ETH +1.8%); capacity-controlled preference (ETH nonlinear 36/36; BTC linear 36/36); calm-regime ties everywhere | conditional, asset- and capacity-dependent | supported on GSPC (both seeds), marginal on BTC/ETH | **inconsistent across assets** | n/a | moderate-high | **Partially supported** (conditional value) |
| R2 Volatility explains the nonlinear advantage | interaction Δ_high50−Δ_low50 per asset | GSPC negative, EURUSD negative, BTC/BABA/NVO positive | 5 of 10 supported | **sign spans zero (6 neg / 4 pos)** | stable for the candidate features, not a volatility test | high for "not universal" | **Directional only** (not a universal law) |
| R3 Sample-level routing opportunity | 12 states × 3 assets × 3 seeds, capacity-matched | no state preference survives capacity | GSPC/BTC linear in 12/12 states; ETH flip is asset-wide | consistent across assets in the negative sense | n/a | high | **Not supported** (`NO_ROBUST_SAMPLE_LEVEL_ROUTING_EVIDENCE`) |
| R4 Asset-level operator heterogeneity | capacity-controlled preference + interaction signs | ETH → nonlinear, BTC → linear, GSPC → parity | supported at widths 64/128 (36/36 each) | consistent for the three measured assets | n/a | moderate-high | **Supported** |
| R5 Capacity affects conclusions | width 23/64/128 sweep | ETH flips, GSPC collapses, BTC unaffected | 25 sign reversals; consistency 31→29→28/36 | measured on 3 assets only | n/a | high that it matters | **Supported** (magnitude HIGH) |
| R6 Structural explanation of the heterogeneity | Spearman + LOO on 8 pre-registered features | `|ACF1|` and realized volatility positive | p = 0.016 and 0.038 at N=10 | sign spans zero across assets | stable (no flips) | moderate | **Directional only** (candidate regularity) |
| R7 Jump ratio as a structural feature | median jump ratio identical (0.01052632) for all 10 assets | — | — | — | — | high | **Not supported** (no discriminative power at c = 3σ) |
| R8 GSPC conclusion depends on seed merging | recomputed under 4 GSPC treatments | stable | ρ within ±0.12, no sign flip | n/a | n/a | high | **Not supported** (`GSPC_SEED_SENSITIVE = NO`) |
| R9 Adaptive gating / α(X) routing pays | gatev1 0.06767 vs additive 0.06230 on GSPC | worse than parameter-free addition | single protocol, one seed | n/a | n/a | moderate-high | **Not supported** |

| R10 Component ablation (woLSTM / woTransformer / woDy) | 6 runs on a second machine, single seed 2021, identical protocol; env calibration reproduced (0.070065 vs our 0.07007) | woTransformer 10.71% worse than the full model (clean); woLSTM 1.29% **better** than full on GSPC; BTCUSD all three arms worse than full | calibration inside the pre-registered band; ablation deltas rest on **one seed** | 2 assets only | n/a | low-moderate | **Partially supported** (component necessity not established) |
| R11 Validity of the woDy ablation | `woDy` is architecturally identical to `revin-DLinear` (4,664 params) yet differs by **8.28%**; traced to the weight-init convention (`1/L·1` vs PyTorch default) | not interpretable as a component-removal effect | n/a | n/a | n/a | high | **Not supported** (`CONFOUNDED_BY_INIT`) |
| R12 Cross-machine reproducibility of the baseline | a second, independently configured machine reproduced the GSPC T=24 `revin-DLinear` calibration (0.070065 vs 0.07007) | agrees | inside the pre-registered band 0.066–0.074 | 1 asset, 1 model | n/a | moderate-high | **Supported** |

## Notes on reading the matrix

- "Directional only" is the correct status whenever the estimate has a sign but the interval is
  not separated from zero, or when the sign is not stable across assets. It must not be upgraded
  to "supported" in any downstream summary.
- R2 and R6 are deliberately kept at "directional only": both have *some* statistical support
  (5/10 interactions supported; two features p<0.05) but the sign instability across assets is the
  dominant fact.
- R3 is a **negative** result and is the strongest single finding in the matrix (highest
  consistency, multiple seeds, capacity-controlled).
