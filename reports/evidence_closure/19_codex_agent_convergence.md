# 19 · Convergence between the operating agent and the external Codex agent

**Date:** 2026-09-13 20:20 · **Authority:** this document records the *shared* position. Where the
two sides differed, the resolution follows from the already-frozen rules (`11`, `12a`, `12b`), not
from preference. Neither side edits the other's records; disagreements are recorded, not erased.

## 0. State of the two clones (why the divergence appeared)

| | External Codex clone | Operating agent's clone (`main`) |
|---|---|---|
| HEAD | `80e6c5e` | `9737395` |
| Base | `5267c06` (**pre-Gate-A**) | includes `54bba97` (Gate A) and `9737395` |
| Has `14_structural_validation_result.md` (Gate A = FAIL)? | **no** | yes |
| Has `18_*` (NS operator design) / `18b_*` (runner-requirement response)? | **no** | yes |
| `origin/main` ref | stale at `5267c06` | current |

**Action for the Codex side:** `git fetch origin && git rebase origin/main` (its single commit
`80e6c5e` applies cleanly; the runner files are byte-identical to what now sits on
`draft/ns-benchmark-runner`).

## 1. Agreed (both sides already state this)

1. **Gate A = FAIL** (recorded in `14_*`): the structural-bridge direction is closed and
   `NS_BENCHMARK_ELIGIBLE` is downgraded from `CONDITIONAL`.
2. The runner is an **instrument, not evidence** — its own docstring says *"It is not evidence that
   an NS-inspired structure is useful."* It must never be cited as support for the NS frame.
3. **No fabricated eligibility.** `eligibility_evidence = C1_REPLICATED` cannot be asserted: under a
   `FAIL`, C1 does not run, so that evidence cannot exist.
4. The runner must not execute in this line. If it is ever executed: `status = LOCKED`, real
   per-asset hashes, its own pre-registration, a **leakage audit**, and execution in a clone with
   the project virtualenv.
5. **`16` §3's own stopping rule** agrees with the dormant disposition: if the structural claim
   fails, structural-candidate and NS-operator work stops.
6. **The twelve structural states are evaluation strata, not training multipliers**: F1 is
   `7 × 3 widths × 3 seeds × 2 operators = 126 fits` (≈ 5–7 min), not 1,512.
7. **Parameter matching holds only at width 23** (9,240 vs 9,431); at 64/128 the comparison is
   capacity-varying and must be described that way.

## 2. The one requirement that is superseded

The runner's README asks the operating agent to

> 把模板替换成已锁定的 manifest，提供完整资产列表与 SHA-256、`C1_REPLICATED` 资格证据报告路径及事先写定的证伪文字。

**Superseded by the Gate A verdict.** Agreed replacement wording:

> A manifest may be locked only when an eligibility report exists that is **not** derived from this
> closed line — i.e. a genuinely independent prospective structural validation under its own
> pre-registration. Until then the runner stays dormant and any manifest stays
> `BLOCKED_PENDING_C1` (as in `ns_hypothesis_benchmark.cohort-scaffold.json`).

What was still delivered under the old requirement, because it is legitimate: real SHA-256 hashes,
the falsification block verbatim, and a **read-only preflight** proving the instrument refuses to
run (`PRECHECK_OK … Execution remains blocked until status=LOCKED, C1_REPLICATED evidence, hashes,
and report path are present.`).

## 3. Agreed disposition of the artefact

- Runner, configs, README section, scaffold manifest and the `20_project_evidence` snapshot live on
  **`draft/ns-benchmark-runner`**; `main` carries no runner code and only a one-line pointer.
- The standing constraint list (`RESEARCH-LINE.md` §7, `PROJECT-BRIEF.md` §7) is **unchanged**; the
  owner's lift applied to this artefact only.
- `main` remains the single source of truth for evidence.

## 4. Agreed next experiment (both documents already support it)

**F1 — capacity-controlled breadth panel on the remaining original assets**, ordinary operators only
(no NS, no router, no new definitions):

- prerequisite "`14_*` records FAIL" ✅ satisfied;
- asset list and file hashes locked **before** execution;
- identical preprocessing / optimiser / stopping to E4–E5; idempotent output naming that cannot
  overwrite existing raw results;
- 126 fits ≈ 5–7 min on this host; all data already on disk; no network dependency;
- criterion as tightened in `16a` §4: success requires **≥ 2 additional assets** with preferences
  stable across all three seeds **and** both widths 64 and 128 (the ETH/BTC pair already exists, so
  merely re-finding it proves nothing);
- if it fails, the asset-heterogeneity claim itself is recorded as unreproduced and structural /
  NS directions stop.

**This is an owner decision** — neither agent runs it unilaterally.

## 5. Not negotiable from either side

The frozen Gate rules (`11`, `12a`, `12b`), the Gate A verdict, the constraint lists, and the
requirement that raw results be retained. Disagreement, if it recurs, gets a dated note rather than
an edit to history.
