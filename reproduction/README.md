# Reproduction guide / 复现指南

Run every command from the repository root. Research-specific code and retained
outputs live here; the core training harness remains in `run.py`, `exp/`,
`models/`, `layers/` and `data_provider/`.

## Layout

```text
reproduction/
  analysis/       deterministic audits and result recomputation
  batches/        bounded experiment launchers
  data/           acquisition and conversion helpers
  results/        tracked small tables, JSON/TXT summaries and registries
  c1_handoff/     C1 predictions/targets, commands, manifests and hash inventory
```

## Environment

- Recommended: Python 3.11 and torch 2.5.1.
- Install `requirements/core.txt` after a machine-appropriate PyTorch build.
- CPU runs require `--no_use_gpu`.
- On Windows, set `PYTHONIOENCODING=utf-8` when replaying scripts that print
  Chinese or mathematical symbols.
- C1 execution provenance records Python 3.11.9 / torch 2.5.1+cpu. Some
  receiving-side tabular analyses used an existing Python 3.13 environment;
  the frozen scripts and input hashes anchor the comparison.

## Core checks

```bash
# 61-file schema, chronology, hash and OHLC audit
python reproduction/analysis/dataset_audit.py

# Recompute F1 summaries from the retained 792-row table
python reproduction/analysis/f1_analysis.py

# Verify and rebuild the result-free C1 Stage-1 handoff manifest
python reproduction/analysis/build_c1_blind_manifest.py

# Validate the frozen Phase 1 grid without starting training
python reproduction/analysis/build_phase1_manifest.py

# Audit all 72 real rolling split combinations without training
python reproduction/analysis/audit_phase1_date_splits.py

# Preview the result-free Phase 1 data contract and common prediction keys
python reproduction/analysis/build_phase1_preflight_artifacts.py

# Write the deterministic contract, split manifest, and compressed key table
python reproduction/analysis/build_phase1_preflight_artifacts.py --write

# Preview the 15 calibration fits (training requires explicit --execute)
python reproduction/batches/run_phase1_calibration.py
```

The canonical single-model training command is in the root README. Analysis
scripts no longer write to the former sibling `05_research_intelligence`
directory; all active outputs stay under `reproduction/results/`.

## C1 evidence package

`c1_handoff/` contains the complete 20-asset × 2-arm × 3-seed panel:

- 120 run directories;
- 120 `pred.npy` and 120 `true.npy` files;
- commands and log tails;
- executor and receiver manifests;
- frozen analysis and predictor code;
- per-file SHA-256 inventories;
- `BLIND_STAGE1_MANIFEST.csv`, which intentionally excludes analyst results.

The first handoff message mistakenly supplied the SHA-256 of the pre-commit
Windows working-tree manifest. Git line-ending normalization changed the stored
blob. The correct Git-object hash was recomputed and the discrepancy was logged
before the independent reviewer opened the result inputs. This operational
mistake is part of the audit trail, not hidden metadata.

## Data caveats

See [`../dataset/README.md`](../dataset/README.md). In particular, WTI's
non-positive April 2020 prices make log-return features undefined locally, and
five FX feeds contain small OHLC envelope inconsistencies. Scripts must disclose
their handling; they must not mutate source rows in place.

## Output discipline

- Never overwrite an earlier experiment family; use explicit suffixes.
- Keep prediction/target arrays, configuration, logs and hashes for every
  headline result.
- A summary table without its raw inputs is incomplete.
- Test data is evaluated only after validation-selected training is complete.
- Do not add an asset, seed, feature or threshold to rescue a failed frozen
  result.

## Phase 1 modern-baseline programme

The frozen protocol is [`../docs/PHASE1_PREREGISTRATION.md`](../docs/PHASE1_PREREGISTRATION.md),
with machine-readable settings in `configs/phase1_modern_baselines.json`.
Generated manifests live in `results/phase1/`. The manifest builder performs
data-hash and model-availability checks and never launches training. Explicit
end-exclusive date boundaries are implemented and audited across all frozen
Stage D asset/year/horizon combinations; Stage D is now engineering-ready but
remains downstream of Stages A--C under the frozen stage order.

Before any Stage B--D model run, `phase1_data_contract.csv`,
`phase1_split_manifest.csv`, and `phase1_prediction_keys.csv.gz` freeze the
confirmatory data identity and test prediction keys without loading model
outputs. B/C share identical fixed-split keys; D uses its explicit date origins.
This preflight does not resolve the separately recorded Stage A deviation,
or authorize training.

The shared B/C/D launcher is `batches/run_phase1_stages.py`. It defaults to
read-only dry-run, verifies dataset identity and precomputed key hashes, and
continues past prior attempts only when their full packaged artifacts pass
SHA-256 checks. A partial, failed, or ambiguous attempt stops the batch. Fits
are deliberately sequential (one GPU job at a time), and `--limit` allows a
bounded batch; rerunning the same stage skips only verified completed packages.
No automatic retry is implemented. A retry must receive its own reviewed
authorization and attempt identifier.

The P4 path is explicitly selected with `--device cuda`; do not rely on device
auto-detection. Current Stage A P4 receipts show a TimesNet peak of 4,674.9 MiB
and 2,519 seconds training time for one h=24 fit. This supports single-fit
feasibility on the 8 GB card, not a claim that every future fit has the same
resource use. Stage B contains 60 TimesNet fits; a naive linear extrapolation
from that one calibration is about 42 GPU-hours for those fits alone, so plan
for multiple lease windows and resume only from hash-verified completed
packages. This is a rough capacity estimate, not a deadline guarantee. Run
batches sequentially inside a persistent remote session
(such as tmux), stop the queue before the lease expires, and leave enough time
for the longest observed fit plus packaging. A lease cut during an individual
fit leaves an immutable interrupted attempt that cannot be auto-retried; it
needs explicit failure adjudication and a separately approved retry. Between
completed batches, restart with the same frozen commit, environment
fingerprint, and GPU model. Stage B
must remain on one GPU model and one locked software environment for the full
panel. P4 availability or this engineering readiness does not itself pass the
Stage A/scientific gates.

Example preflight commands from the repository root:

```bash
python reproduction/batches/run_phase1_stages.py --stage B_screen --limit 40 --device cuda
python reproduction/batches/run_phase1_stages.py --stage C_confirmation --device cuda
python reproduction/batches/run_phase1_stages.py --stage D_temporal_robustness --device cuda
```

Execution is intentionally fail-closed. `--execute` requires a separate
versioned authorization JSON and a passing worker environment-fingerprint
JSON, bound to the exact source commit, requested stages, and device. The
current repository state has no such Stage B/C/D authorization, so these
commands only preview work; do not add `--execute` until the required gates and
decision record are reviewed. Intake analysis remains a separate review step;
packaged runs are labeled `completed_unreviewed`, not research findings.

The read-only intake command is:

```bash
python reproduction/analysis/validate_phase1_attempts.py --stage B_screen
```

It checks one-and-only-one attempt per manifest row, receipt identity, artifact
hashes, expected test-window shape, finite arrays, frozen split/key identity,
and `true.npy` alignment to raw dates with the train-only scaler. It does not
interpret model scores or replace scientific review.
