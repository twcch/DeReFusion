# Protocol freeze

- Protocol version: `phase1-v1.1-2026-09-19`
- Normative protocol: `docs/PHASE1_PREREGISTRATION.md`
- Execution controls: `docs/PHASE1_OUTSOURCING_EXECUTION_PLAN.md`
- Frozen manifests:
  - `reproduction/results/phase1/B_screen.manifest.csv`
  - `reproduction/results/phase1/C_confirmation.manifest.csv`
  - `reproduction/results/phase1/D_temporal_robustness.manifest.csv`

The worker may not change the model grid, datasets, splits, seeds, horizons,
metrics, success criteria or analysis. Infrastructure problems create a failed
attempt and a stop; they do not authorize repair of scientific inputs.

Stage B--D execution remains prohibited until a repository-recorded §8 gate
review explicitly authorizes the relevant stage.
