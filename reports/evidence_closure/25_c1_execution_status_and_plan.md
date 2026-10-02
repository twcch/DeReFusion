# 25 · C1 execution status, open problems and plan

**Date:** 2026-09-16 · **Scope:** execution status of the C1 panel and the receiving-side verification
only. This document contains **no analysis of C1 outcomes**: no headline statistic, no interaction
effect, no reading of either branch of the pre-committed interpretation. Blinding is preserved.

---

## 1. Where the panel stands

| Item | Value |
|---|---|
Target | 120 runs = 20 assets x 2 arms x 3 seeds |
Attempted so far | **87** |
Succeeded (`ok`) | **54** |
Failed (`stall`) | **33** |
Still un-attempted | 33 |
Lanes | 2 (a bounded third lane was tried and rolled back - see P2) |
Started | 2026-09-14 13:36, per-run wall clock ~70 min |
Handovers received | `BATCH-2021-DONE.md` (20 ok / 20 stall), `BATCH-2022-DONE.md` (24 ok / 16 stall) |
Final receipt | `007-T005-receipt.md` **not yet written** |

The distribution of the 87 rows is the single most important fact in this document:

| Arm | ok | stall |
|---|---|---|
DeReFusion (experimental) | **44** | **0** |
revin-DLinear (baseline) | **10** | **33** |

The baseline arm has produced 10 of its 60 runs. The experimental arm has produced 44 of its 60 and
has never stalled.

## 2. What the receiving side has already verified

Everything delivered so far passed independent intake verification (54/54):

- **Protocol conformance per run** - the stored command line carries every frozen value plus that
  run's own seed, model, `model_id` and data path.
- **Content integrity** - `pred.npy` / `true.npy` SHA-256 match the batch manifest; the executor also
  proved byte-level equality between its local originals and the shipped copies (6/6 files).
- **Metric integrity** - `metrics.npy` parses as the frozen six-value order, and all six values are
  reproducible from the delivered arrays using the pipeline's own metric functions (~1e-7 relative).
- **Ground-truth alignment** - for the first two runs, the delivered `true.npy` equals the target
  windows recomputed from the **locked cohort CSV** with the pipeline's own split and scaler
  conventions (max abs diff 2.3e-07 and 1.2e-07), and the array shape convention `(n, T, 1)` is
  identical to the project's existing runs.
- **Additional audits** - no duplicate arrays across runs, log tails belong to their directories,
  manifest and directory set are in bijection, and grid coverage matches expectation.
- **Handoff artefacts produced** - a **278-file SHA-256 inventory** and a handoff receipt recording
  `executor_reported_commit = d17822f`.

The 33 `stall` rows are reported by intake as manifest rows without a run directory. That is expected
- those runs were killed mid-training - but it is a **completeness gap**, not a corruption, and it is
systemic rather than random (P1).

## 3. Open problems

### P1 - Baseline-arm runs are killed by the liveness test (severity: blocking)

**Symptom.** Every one of the 33 stalls is on `revin-DLinear`; the experimental arm has none.

**Evidence it is a false kill rather than a real hang.** (a) the killed units left
`checkpoints/<setting>/checkpoint.pth` behind, i.e. they were actively training when killed;
(b) the reported stall values cluster in a narrow band (CPU frozen at roughly 50-80 s), which a
genuinely hung process would not do; (c) the only difference between the arms is model size.

**Mechanism.** Liveness is judged from an aggregate of the process tree CPU summed by parent-process
id. On the smaller-model path that aggregate evidently stops tracking the real work, so after 12
minutes of apparent CPU stasis the tree is killed. Each killed unit is retried up to three times, so
33 stall rows represent roughly three kills each.

**Consequence if left alone.** The baseline arm can never complete; C1 would lose on the order of 40%
of its control evidence, which would gut the paired comparison the study exists to make. The operator
had already decided that baseline stalls would be retried after the 120, but retrying under the same
criterion would simply reproduce the same kills.

**Ruling (made and communicated).** A minimal **runner** change is approved: judge liveness **by
artifact progress** (checkpoint / results writes) instead of process-tree CPU. The approval is bounded
by: no change to protocol, data, seeds, hyperparameters, windows, splits or metrics; **stall detection
is replaced, not removed** (a tree is killed only when artifact progress **and** CPU growth are both
absent for N minutes, so a genuinely hung run is still caught); every kill is logged with both
criterion values so the killed set is auditable; and the 54 delivered rows and arrays are not touched.

### P2 - The bounded third-lane trial is uninformative and was rolled back (severity: medium)

The trial tripped its circuit-breaker (two stalls) and was rolled back to two lanes as specified. The
executor also disclosed that during the trial window the machine was actually running **four**
concurrent runs - two orphans from the two-lane era plus two newly launched ones - so the window
measured **oversubscription**, not a clean three-lane configuration. The clean question ("does a third
lane help, and with how many threads per lane?") therefore moves to the out-of-campaign tuning A/B,
which will run on an idle machine after the panel completes. Until then the campaign stays at two
lanes.

### P3 - A recurring operational failure class: self-matching process filters (severity: medium)

Two independent incidents share one root cause: a filter that matches strings appearing in the
inspecting process's own command line. In the stability drill a counting rule produced a **false
"5/5 PASS"** that the executor itself retracted; during the rollback a process filter matched its own
shell and **killed it**. Both were reported rather than hidden, and the corrected pattern (run the
action from a script file, exclude `$PID`) is now in use. This class deserves to be on the checklist
for any future long-running operator.

### P4 - Smaller open items

| # | Item | Status |
|---|---|---|
a | `wall_clock_min` is estimated for orphan-ingested rows | accepted, real timing required for normal-path rows |
b | `Test-Recorded` only saw lane 0/1 CSVs, so a third lane could re-run a completed run | fixed (wildcard `lane*_runs.csv`) |
c | Two different handover documents share the number `018` | cosmetic; ledger keys on paths, not numbers |
d | `007-T005-receipt.md` not yet produced | pending panel completion |

## 4. Plan

| Step | Owner | Action | Done when |
|---|---|---|---|
S1 | executor | Apply the artifact-progress liveness criterion | kill events logged with both criterion values; a dry-run shows a healthy run no longer killed |
S2 | executor | Idempotently re-run the 33 stalled runs and finish the 33 un-attempted ones | 120 rows accounted for: per arm, 60 completed runs |
S3 | executor -> receiver | Hand over per seed batch as they close | each handover carries manifest, artefacts and the execution revision |
S4 | receiver | Intake verification per batch | protocol, hashes, metrics, audits pass; inventory and receipt updated |
S5 | receiver | Assemble the handoff: 120 runs, 240 prediction/ground-truth arrays, manifest, frozen revision, per-file SHA-256 inventory, executor's local-origin hash confirmation | all six items present and mutually consistent |
S6 | operator / independent third party | Blind recomputation of the headline statistic from the raw artefacts | difference list returned; discrepancies reported as findings |
S7 | operator | Apply the pre-committed interpretation of whichever branch the outcome falls into | one of the two pre-committed branches recorded |

**Time estimate.** About 66 runs remain (33 re-runs + 33 new). At ~70 min per run and two lanes that
is roughly **1.5-2 days**, excluding any further defect discovery.

## 5. Boundaries in force

- No change to protocol, data, seeds, hyperparameters, windows, splits, metrics or capacity settings.
- Stall detection is replaced rather than disabled; every kill is logged and auditable.
- The campaign stays at two lanes; the A/B waits for an idle machine and its numbers never enter the
  C1 evidence chain.
- Anomalies are reported, never silently fixed; negative or inconvenient findings are reported with
  the same prominence as convenient ones.
- Blinding is preserved: whatever this document does not contain, it omits deliberately.
