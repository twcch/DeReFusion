# Phase 1 completed-artifact audit and P4–RTX 5060 Ti bridge

**Audit date:** 2026-09-29

**Protocol:** `phase1-v1.1-2026-09-19`
**Scope:** read-only validation of archived P4 Stage B outputs and the authorized, nonconfirmatory RTX 5060 Ti bridge. This report does not interpret the screening results or make an equivalence claim.

## Executive result

- **P4 archive:** 245 of 300 planned Stage B fits are present; all 245 passed receipt/hash checks, frozen split/target checks, finite-array checks, and metric recomputation. The 55 planned `revin-TimesNet` fits are absent from the P4 archive.
- **RTX 5060 Ti bridge:** all 34 authorized fits passed the same per-artifact checks. All 34 bridge outputs have a P4 counterpart; their `true.npy` arrays are bitwise identical.
- **No integrity/provenance-index or pair-index errors were found.** The P4 environment record does not include a driver-version field, so that portion of its runtime provenance is incomplete.
- **Cross-hardware numerical equivalence is not established.** Median metric differences are small, but some paired differences are material; there was no precommitted equivalence margin, and the bridge authorization explicitly prohibits pooling. Keep the P4 and RTX 5060 results as separate execution tracks.

Machine-readable per-fit evidence is in [P4-RTX5060-bridge-audit-20260929.json](P4-RTX5060-bridge-audit-20260929.json). The audit implementation is [audit_phase1_execution_stacks.py](../../reproduction/analysis/audit_phase1_execution_stacks.py).

## What was checked

For each present attempt, the audit verified the receipt identity/status and protocol/stage, all five receipt-bound artifact hashes (`pred.npy`, `true.npy`, `metrics.npy`, `checkpoint.pth`, `run.log`), dataset and frozen split/prediction-key identities, expected test-target reconstruction, array shapes and finite values, and recomputed six metrics from the saved predictions and targets. It then paired the 34 bridge fits with their indexed P4 attempts and compared exact targets, predictions, and metrics.

Across 279 validated fits, this covered **1,395 receipt-bound files** and **279 metric recomputations**. This establishes package integrity and target alignment, not correctness of every model implementation or cross-device equivalence.

## Execution-stack differences

| Track | Code commit | GPU | Python | PyTorch / CUDA runtime | Driver recorded |
|---|---|---|---|---|---|
| P4 archive | `bf738b9c42104197ee6204f9bcbd223fe0b3c3e2` | Tesla P4 | 3.11.15 | 2.5.1+cu121 / 12.1 | Not present in the archived fingerprint |
| RTX 5060 Ti bridge | `c218ccc2cd98d3532648b7401fd2b6443cdcdf1d` | NVIDIA GeForce RTX 5060 Ti | 3.11.15 | 2.7.1+cu128 / 12.8 | 610.62 |

The commits differ. A commit diff shows no changes under `models/`, `exp/`, `data_provider/`, `utils/`, or `layers/`; the bridge commit does change the experiment runner to support isolated supplemental runs and distinct run labels. Thus the bridge is useful evidence about numerical sensitivity under the two recorded stacks, but it is not a controlled software-identical GPU-only experiment.

The currently running RTX 8000 TimesNet batch is a third execution environment and must likewise remain separately identified; this bridge does not certify it as numerically equivalent to either P4 or RTX 5060 Ti.

## Paired numerical differences

Relative differences below are computed against the P4 metric for each paired fit. They are descriptive summaries over the 34 selected pairs, not confidence intervals or acceptance bounds.

| Metric | Median relative difference | Maximum relative difference |
|---|---:|---:|
| MAE | 0.105% | 10.747% |
| MSE | 0.193% | 16.691% |
| RMSE | 0.096% | 8.726% |
| MAPE | 0.111% | 10.838% |
| MSPE | 0.275% | 20.081% |
| R² | 0.019% | 2.028% |

Prediction relative-L2 difference had a 0.114% median and 7.115% maximum. The largest relative MSE differences were 16.69% for BTCUSD / iTransformer / horizon 1, 12.22% for EURUSD / PatchTST / horizon 1, and 8.06% for EURUSD / iTransformer / horizon 1. These tail cases mean “the medians are small” cannot be substituted for a validated equivalence margin.

The paired bridge is small and selected for coverage; test windows overlap and are dependent. No hypothesis test, post-hoc correction, rescaling, ranking claim, or pooling is warranted from it.

## Stage B and compute implications

The P4 Stage B authorization states that a screening decision must not be taken while planned fits remain. The archived P4 track is therefore partial (245/300). The separately authorized RTX 8000 queue covers those 55 missing TimesNet fits; it is a nonconfirmatory supplemental track, not a retroactive extension of the P4 hardware run. Preserve the per-track provenance and do not pool metrics across tracks absent a prospective amendment with an explicit equivalence margin and decision rule.

At the **2026-09-29 07:50 UTC** snapshot, the RTX 8000 first TimesNet fit was still in progress at epoch 8 (no completed fit yet); the epoch log showed approximately 82–86 seconds per epoch, validation loss improved at epoch 7, and the run had already consumed about 12 minutes. Live telemetry showed 99–100% GPU utilization and about 15.7/46.1 GB VRAM in use. This is compute-bound, not VRAM-capacity-bound. The 55-fit queue needs an average of at most 8.73 minutes per fit to finish within 8 hours; the first fit had already exceeded that average while still training. A rough extrapolation of just 8 epochs at the observed rate is about 11 minutes per fit, or 10+ hours for 55, before allowing for fits that run longer. Thus the current 8-hour cap is unlikely to complete all 55, though the first receipt is needed for a better estimate. Running more fits concurrently on the same GPU is unlikely to shorten the queue materially and adds contention. If the 8-hour wall-clock cap is strict, additional compute is likely needed via a **separately authorized, same-stack** worker; a second 5060 Ti is not a substitute for RTX 8000 numerical equivalence, and its current local-supplement authorization forbids pooling.

## Recommended disposition

1. Keep the current RTX 8000 run on its authorized 55-row manifest, without changing its configuration or extending its 8-hour authorization in place.
2. Do not start a duplicate 5060 Ti queue or add same-GPU concurrency merely to consume spare VRAM.
3. If the 8-hour cap proves insufficient after the first fit completes, make a prospective operational decision about a second same-stack worker and issue a distinct, immutable authorization/output root before launching it. Do not split the queue across software stacks and then treat the combined result as one homogeneous screen.
4. Keep the 5060 Ti bridge as an audit/calibration artifact. Any future equivalence claim requires a predeclared tolerance and independent review; until then the cross-device consistency question remains unresolved.
