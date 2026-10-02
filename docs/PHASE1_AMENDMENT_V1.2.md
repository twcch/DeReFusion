# Phase 1 Protocol Amendment v1.2

- Status: **approved for the Stage A provenance disposition only**
- Protocol parent: `phase1-v1.1-2026-09-19`
- Effective decision time: `2026-09-28T15:25:06Z`
- Authorizing decision: the user approved disposition (b) in the Codex task on 2026-09-28. `PHASE1_STAGE_A_DEVIATION.md` §10 is cited only as the contemporaneous proposal and provenance record; it is not an authority. The operative authority is this amendment and its linked decision record.
- Decision record: [`PHASE1_EXECUTION_DECISION_V1.2.json`](PHASE1_EXECUTION_DECISION_V1.2.json)

## 1. Approved disposition

The 15 P4 Stage A packages under `reproduction/results/phase1/calibration/`
are the authoritative Stage A execution-calibration package set. They all
report `calibration_pass`, share the full source commit
`0d05f93198d07dcea64dbcc08e7a783a09152e6f`, and their receipt-listed artifact
hashes were verified on 2026-09-28 with zero mismatches.

The 14 CPU packages under `reproduction/results/phase1/calibration-cpu/` are
preserved as immutable historical execution evidence. They are excluded from
the active Stage A package set and from active attempt-uniqueness counting.
They must not be deleted, rewritten, or described as superseding the P4
packages. In particular, the CPU A05 `resource_blocked` receipt remains
unchanged; its historical status is not a gate on the selected P4 package set.

This is a transparent post-hoc provenance reconciliation, not a claim that the
P4 grid was within the original two-fit GPU execution authorization. The
expanded P4 execution remains the procedural deviation recorded in
`PHASE1_STAGE_A_DEVIATION.md`. The disposition is based on execution validity
and complete artifact coverage only; Stage A losses and model rankings were
not used.

## 2. Scientific scope unchanged

This amendment changes only which already-existing Stage A execution package
set is authoritative. It does not change the research question, estimand,
target population, assets, cohort labels, models, horizons, seeds, split
definitions, prediction keys, training configuration, metrics, multiplicity
plan, or stage order in v1.1. Stage A remains execution calibration only; its
losses are not evidence for or against a model.

The dataset contract, split manifest, and prediction-key universe remain those
identified by the v1.1 preflight artifacts. No raw dataset, receipt, checkpoint,
prediction, target, or metric file was modified to implement this decision.

## 3. Stage B--D authorization remains separate

This amendment does **not** authorize Stage B, C, or D training. Before any such
run, the execution plan's remaining controls must be closed and a separate,
stage-specific execution authorization must bind the exact source commit,
environment fingerprint, GPU model, requested stage, and permitted run scope.
The required failure/retry lineage tests and worker-side preflight must pass.
No Stage B result may be called confirmatory before that authorization and the
pre-registered stage sequence are satisfied.

## 4. Execution provenance and resource boundary

The selected Stage A device is Tesla P4 (7,680 MiB); the saved P4 fingerprint
reports Python 3.11.15, torch 2.5.1+cu121, CUDA runtime 12.1, NumPy 2.1.2,
pandas 2.3.3, scikit-learn 1.7.2, and no version mismatches. It does not yet
record every environment-lock item (including immutable container digest and
GPU UUID/driver details); those must be captured before any new B--D run.

The 2,519-second Stage A P4 TimesNet h=24 fit is a single-run resource
observation, not a guaranteed per-fit duration. For planning only, 60 Stage B
TimesNet fits extrapolate to roughly 42 GPU-hours. Use one fit at a time on the
8 GB P4, bounded batches, and persistent sessions; leave a lease buffer longer
than the longest observed fit. A lease interruption during a fit must be
retained as an immutable failed attempt and separately adjudicated before any
retry.
