# 16a · Agent review notes on the post-Gate program (16)

Review of `16_post_gate_experiment_program.md` by the operating agent (2026-09-13).
**No rule, threshold or result is changed here.** These are corrections/additions the plan needs
before its branches can be executed: measured resource numbers the plan explicitly leaves open,
one unmet prerequisite, and one falsification criterion that is already satisfied by existing data.

## 1. VERIFIED: the plan is docs-only and consistent with the frozen rules

- The document alters no result, threshold, definition, seed, metric, split, horizon, window,
  capacity or decision rule (checked against `11`, `12a`, `12b`).
- Its two branches and its NS feasibility boundary match the constraints already in force
  (no router/gate/MoE/attention/pressure/NS implementation; terminology discipline).
- The edits it made to `PROJECT-BRIEF.md`, `RESEARCH-LINE.md`, `reports/README.md`, `docs/ROADMAP.md`
  are additive pointers; all four files verify as valid UTF-8 with no replacement characters.

## 2. UPDATE: branch C1's prerequisite #2 is currently unmet (data)

C1 requires *"one source can supply complete, consistently adjusted 2016–2025 OHLC data for the
entire new cohort; no composite-provider cohort"*. On this host that is **not** satisfiable today:

| Source | State from this host |
|---|---|
| Yahoo (`query1`/`query2`) | **HTTP 403** (edge block; TLS itself verified fine) — see `15_*` |
| Stooq | HTTP 200 but a **JS anti-bot challenge** page, not CSV |
| Sohu | **works**, but A-shares/ETFs only → cannot supply a non-A-share cohort |
| East Money | `RemoteDisconnected` (anti-bot) |

**Consequence:** C1 is, as of now, **blocked on data acquisition**, not on method. The most
practical route is the one already in motion: have an external agent with independent egress
(e.g. Codex in its cloud sandbox) fetch the cohort and **commit it to a branch**, from which the
working clone pulls. Operations must then re-verify: single provider, identical adjustment policy,
row counts, date coverage, no duplicates, and a pre-registered universe/exclusion list (C1's
prerequisites 2–3).

## 3. UPDATE: measured resource envelope (the plan leaves this to be estimated)

Measured on this host (CPU-only, 2 concurrent runs):

| Quantity | Measurement | Extrapolation |
|---|---|---|
| Proxy operator fit (linear map / MLP, 12 states, one seed) | the `_sv` capacity run completed 4 assets × 2 widths × 3 seeds × 2 operators = 48 fits in ≲2 min | **≈2–3 s per fit** ⇒ branch **F1's 1,512 fits ≈ 1–1.5 h** single-threaded; parallelisable |
| Framework run (DeReFusion or revin-DLinear, `T=24`, seed fixed) | measured pairs: BYD 17:14→17:47, BOE 17:47→18:22, EASTMONEY 18:48→~19:25 | **≈30–35 min per pair** (2 concurrent) ⇒ **≈15–17 min per run effective** |
| Branch **C1** (15–20 assets × 2 arms × 3 seeds = 90–120 runs) | — | **≈25–35 h wall-clock** at 2-way parallelism on this host |

**Operational caveat for C1:** detached batches on this host have died silently several times
(three occurrences on 2026-09-13). Long campaigns must therefore run under an **idempotent,
self-healing launcher** (see `run_sv_batch_guard.ps1` + the scheduled task) and must be verified by
**process CPU time**, not by log output. Budget for restarts.

## 4. CORRECTION: F1's falsification criterion is already satisfied — it must be tightened

The plan states that the reproducibility claim fails if the completed panel contains *"no pair of
assets with opposite mean ΔMSE signs that are each consistent across all three seeds and both widths
64 and 128"*.

That pair **already exists** in the current three-asset panel:

- ETHUSD — nonlinear-favoured, **36/36** state observations at widths 64 **and** 128, 3 seeds;
- BTCUSD — linear-favoured, **36/36** at every width, 3 seeds.

So as written, F1 would "pass" without adding any new information. F1's actual content is
**breadth**, not the existence of a pair. Recommended tightening (to be pre-registered before F1
runs, i.e. now, while only the 3-asset panel exists):

1. Success requires **at least two *additional* assets** with preferences that are stable across
   all three seeds **and** both widths 64 and 128 (one joining each class, or two joining either) —
   i.e. breadth beyond the three assets that generated the claim.
2. If an asset's mean ΔMSE sign differs between widths 64 and 128, or across seeds, it is recorded
   as **unstable** and does not count toward either class (unchanged from the plan).
3. Every eligible asset is reported, including ties and failures (unchanged from the plan).
4. F1 remains an **ordinary-operator stability** experiment: it does **not** test `|ACF1|` and must
   not be used to resurrect a structural explanation.

## 5. Note on the NS boundary in the plan

The narrow form the plan describes (nonlinear flux difference + fixed-form second difference +
ordinary residual forcing; no router/gate/MoE/attention/pressure; explicitly *not* a physical claim)
is consistent with `reports/ns_hypothesis_charter.md`, and its caveats match: it may degenerate into
an ordinary temporal convolution, and any gain may come from capacity, optimisation or
regularisation rather than the bias. The stated boundary — *possible in principle, scientifically
unqualified at present; eligibility ≠ evidence* — is the correct one and is adopted as-is.
