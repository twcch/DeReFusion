# 18b · Response to the requirement left for the operating agent (NS benchmark runner)

**Date:** 2026-09-13 20:15 · **Actor:** operating agent (OpenClaw) · **Branch:** draft work on
`draft/ns-benchmark-runner`; this record lives on `main`.

## 1. The requirement, as written by the runner's author

In `reproduction/README.md` (commit `80e6c5e`, external Codex clone) the runner's usage section says:

> 执行前，**OpenClaw 需要**把模板替换成已锁定的 manifest，提供完整资产列表与 SHA-256、
> `C1_REPLICATED` 资格证据报告路径及事先写定的证伪文字。

i.e. the operating agent is asked to (a) fill the manifest template into a **locked** manifest with
a complete asset list and per-file SHA-256, (b) supply the `C1_REPLICATED` eligibility-evidence
report path, and (c) supply the pre-written falsification text.

## 2. What was provided

| Asked for | Status |
|---|---|
| complete asset list | ✅ provided in `reproduction/configs/ns_hypothesis_benchmark.cohort-scaffold.json` (the four structural-validation cohort assets) — **labelled a placeholder, not a pre-registered cohort** |
| per-file SHA-256 | ✅ **computed and embedded** (BYD `9ffb1f8e…`, BOE `7139cb2e…`, EASTMONEY `adbab122…`, YANGHE `fa3a9c83…`) |
| pre-written falsification text | ✅ copied verbatim from the template (utility / mechanism / scope) |
| locked manifest | ❌ **deliberately not produced** — see §3 |
| `C1_REPLICATED` evidence path | ❌ **deliberately not asserted** — see §3 |

**Read-only preflight executed** (no model, dataset or output written; exit 0 in both cases):

```
template -> PRECHECK_OK: manifest is structurally valid. No model, dataset, or output was written.
            protocol_id=REPLACE_WITH_LOCKED_PROTOCOL_ID assets=['REPLACE_WITH_PRE_REGISTERED_ASSET']
            Execution remains blocked until status=LOCKED, C1_REPLICATED evidence, hashes,
            and report path are present.
scaffold -> PRECHECK_OK ... protocol_id=BLOCKED_PENDING_C1__NOT_A_PREREGISTRATION
            assets=['BYD','BOE','EASTMONEY','YANGHE']
            Execution remains blocked ...
```

So the instrument is exercised and **refuses to run** — the intended behaviour.

## 3. What was refused, and why

The scaffold sets `status = BLOCKED_PENDING_C1` and
`eligibility_evidence = NOT_AVAILABLE_GATE_A_FAILED`.

Asserting `eligibility_evidence = C1_REPLICATED` would be **fabricating eligibility evidence**:
Gate A is `FAIL` (`reports/evidence_closure/14_structural_validation_result.md`), and under the frozen
rule a `FAIL` closes the structural-bridge direction, so **C1 (the independent prospective cohort)
does not run**. The condition the runner demands cannot be legitimately satisfied in this line, and
writing the flag anyway would defeat the very gate the runner was built to enforce.

## 4. Where the runner now lives (operator decision)

- `main` is unchanged: **no runner code, no constraint-list change**. The standing prohibition on
  implementing NS modules remains in force for `main` and for every other file.
- The runner, its configs, its README section, the scaffold manifest and a point-in-time snapshot of
  the `20_project_evidence` handoff package live on **`draft/ns-benchmark-runner`**, whose
  `DRAFT-BRANCH-NOTES.md` records: the owner's artefact-specific constraint lift, the dormancy
  reason, the conditions for ever running it, and the compliance statement.
- `RESEARCH-LINE.md` §5 carries a one-line pointer so future readers can find the draft without
  mistaking it for part of the active line.

## 5. If a future verdict ever unlocks it

Required before any execution: its own pre-registration with a real asset cohort and hashes (the
current list is a placeholder), a genuine `C1_REPLICATED`-equivalent result, a **leakage audit** of
the runner (not yet performed), execution in a clone with the project virtualenv, and reporting
against the frozen metrics and the decision rules in `18_ns_operator_design_draft.md`.
