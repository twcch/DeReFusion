# Evidence index / 证据索引

The numbered files in `evidence_closure/` are an append-only decision history.
Later reports may supersede an operational status, but do not silently rewrite
what was known when an earlier decision was made.

## Current conclusions

| Topic | Authoritative report | Status |
|---|---|---|
| Repository-wide audit | [`../docs/REPOSITORY_AUDIT_2026-09-18.md`](../docs/REPOSITORY_AUDIT_2026-09-18.md) | current |
| Gate A structural validation | [`14_structural_validation_result.md`](evidence_closure/14_structural_validation_result.md) | **FAIL** |
| Post-gate boundaries | [`16_post_gate_experiment_program.md`](evidence_closure/16_post_gate_experiment_program.md) | binding boundaries; opening status text is historical |
| F1 breadth panel | [`21_f1_result.md`](evidence_closure/21_f1_result.md) | **SUCCESS with 3/7 unstable caveat** |
| C1 pre-commit interpretation | [`24_c1_precommitted_interpretation.md`](evidence_closure/24_c1_precommitted_interpretation.md) | frozen |
| C1 blind specification/lock | [`24a`](evidence_closure/24a_c1_blind_recomputation_spec.md), [`24b`](evidence_closure/24b_c1_blind_handoff_lock_table.md) | frozen |
| C1 result and audit | [`26_c1_result_and_blind_audit.md`](evidence_closure/26_c1_result_and_blind_audit.md), [`27`](evidence_closure/27_c1_rule_provenance_and_blind_audit_scope.md), [`28`](evidence_closure/28_c1_raw_replay_and_data_quality.md), [`29`](evidence_closure/29_c1_independent_blind_replay.md) | **FAIL**; independent blind replay complete; no material numerical discrepancies |
| Literature and next plan | [`../docs/LITERATURE_AND_NEXT_PLAN.md`](../docs/LITERATURE_AND_NEXT_PLAN.md) | current |
| Phase 1 modern-baseline programme | [`phase1/phase1-experiment-data-and-progress-20260930.md`](phase1/phase1-experiment-data-and-progress-20260930.md) | Stage A calibration 15/15; Stage B artifacts cover 300/300 across separate P4/RTX 8000 tracks; Stage C 360/360 with descriptive analysis only; Stage D gated |

Phase 1's detailed stage reports, CSV summaries and reproduction scripts are
indexed in the linked report. The 2026-09-29 artifact-inventory files are
historical snapshots; the latest 2026-09-30 Stage B intake and closeout reports
supersede their earlier 25/55 RTX 8000 progress count.

## Chronology

- `00`–`09`: repository inventory, initial experiments, integrity review,
  N=10 exploration, diagnosis and the original conditional gate.
- `10`–`15`: structural-validation blocker, pre-registration, frozen decision
  rules, two external reviews and Gate A result.
- `16`–`19`: post-gate plan, NS feasibility boundary, agent review and runner
  design. These are planning records, not evidence that NS was implemented.
- `20`–`22`: F1 pre-registration, result and corrected count convention:
  63 paired settings = 126 arm fits = 792 state rows.
- `23`–`25`: C1 protocol, lock table, pre-committed interpretation, blind
  recomputation spec and an execution-in-progress snapshot.
- `26`–`29`: analyst C1 outcome, partial audit history, rule provenance,
  owner-side raw replay and completed independent blind replay. Both replays
  support FAIL; the independent review found no material numerical discrepancy.

## Historical-warning labels

- `25_c1_execution_status_and_plan.md` is intentionally retained as the
  2026-09-16 snapshot at 54/120 successful runs. It is **not current**; the
  panel later completed 120/120 after the watchdog liveness fix.
- `16_post_gate_experiment_program.md` begins from the state before Gate A and
  F1 completed. Its scientific prohibitions remain relevant, while its progress
  paragraph is superseded by reports 14 and 21.
- reports 17/18 are hypotheses/design drafts. They must not be cited as empirical
  operator or NS results.

## Minimum reading order

1. repository audit;
2. 14 (why Gate A failed);
3. 21 (what F1 did and did not rescue);
4. 24, 24a, 24b, 27–29 (C1 rules, audit scope, and independent replay);
5. 26 (analyst result and the partial independent audit as recorded at that time);
6. literature and next plan.

Negative results, execution anomalies and frozen decisions are retained because
they are evidence. Redundant briefs and superseded top-level roadmaps were
removed during the 2026-09-18 cleanup.
