# Fixed-checkpoint replay audit: existing computation preserved

Date: 2026-09-29. Status: completed diagnostic inference; no training.

All 34 frozen bridge settings were replayed on the RTX 5060 Ti using both
their archived P4 checkpoint and their original local checkpoint: 68 full
test-set inference passes in 43.45 seconds. No optimizer, backward pass,
new fit, production evaluator, or production artifact write was used.

The audit verified the frozen data and prediction-key identities, unchanged
model/loader/evaluator sources, strict checkpoint loading, array shapes,
finite outputs, and exact reconstructed test labels. All 68 label arrays
were bitwise identical to their references; all bound original input files
passed the final immutability check. The 34 pairs also share identical
input-batch and scaler hashes.

| Reference checkpoint / predictions | Replays | Exact prediction arrays | Maximum absolute prediction difference | Maximum relative prediction L2 difference | Maximum relative MSE difference |
|---|---:|---:|---:|---:|---:|
| P4 | 34 | 0 | 0.000559330 | 0.00814349% | 0.00418402% |
| Local RTX 5060 Ti control | 34 | 16 | 0.0000462532 | 0.000215571% | 0.000281410% |

These are descriptive maxima, not acceptance margins. CUDA library defaults,
including TF32 flags, were preserved and recorded, rather than silently
changing numerical precision. Small nonzero differences also exist in the
same-stack control; inference replay is not a bit-exactness guarantee.

The fixed-checkpoint replay differences are much smaller than the differences
between independently trained P4 and RTX 5060 models in the earlier bridge
audit. This helps distinguish inference reproduction from changes in learned
weights. It does not identify a hardware-only causal effect, establish
equivalence, or authorize pooled model rankings. Tolerances remain TBD.

## Artifact retention and next processing step

All original P4, local bridge, and RTX 8000 training artifacts remain intact.
The new small replay arrays, per-setting audit records, bound input inventory,
and complete report are retained separately under
[`checkpoint-forward-5060-v1`](../../reproduction/results/phase1/checkpoint-forward-5060-v1/report.json).
The implementation is
[`audit_phase1_checkpoint_forward.py`](../../reproduction/analysis/audit_phase1_checkpoint_forward.py).
Its seven CPU safety tests passed; the CUDA run also completed successfully.

Completed RTX 8000 packages should be copied into an isolated intake root,
verified against their own manifest/receipts, and then replayed diagnostically
on the local GPU. Running attempts must not be copied as completed packages.
CPU hash/target/metric checks and local GPU inference can overlap remote
training; these activities do not require another training fit.

Stage C remains unstarted. Its 360-row data/key/manifest dry-run passed, but
parallel execution and the local CUDA 12.8 stack still require a prospective
Stage C execution decision. This audit grants no Stage C/D authority.
