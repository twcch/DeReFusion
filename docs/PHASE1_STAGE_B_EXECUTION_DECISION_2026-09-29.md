# Phase 1 Stage B execution decision

- Status: **approved -- bounded Stage B run on the P4 worker**
- Protocol parent: `phase1-v1.1-2026-09-19`
- Amendment in force: [`PHASE1_AMENDMENT_V1.2.md`](PHASE1_AMENDMENT_V1.2.md)
- Decision time: `2026-09-29T00:30Z`
- Authorizing decision: the user directed the P4 worker to begin Stage B in the Claude Code task on 2026-09-29, with a wall-clock budget of roughly eight hours.
- Machine-readable authorization: `authorization_stage_b.json`, written on the worker.

## 1. Why the machine-readable authorization is not committed

`validate_authorization` (`reproduction/batches/run_phase1_stages.py:208`) requires the
authorization's `git_commit` to equal the worker's current `HEAD`. A file that is itself
committed moves `HEAD` and invalidates its own binding, so the authorization is a
**runtime credential** held on the worker and deliberately untracked. The worker's
fingerprint hash and source commit are recorded below and re-verified by that check at
launch; this document is the durable, reviewable half of the same decision.

## 2. Bound scope

| Field | Value |
|---|---|
| Source commit | bound at launch; the worker refuses any other `HEAD` |
| Environment | the frozen `derefusion-p1` fingerprint (`python 3.11.15`, `torch 2.5.1+cu121`, `torch_cuda 12.1`, `numpy 2.1.2`, `pandas 2.3.3`, `scikit-learn 1.7.2`) |
| Device | `cuda`, Tesla P4 (7,680 MiB) |
| Stage | `B_screen` only |
| Run scope | **one wall-clock budget of approximately eight hours** |

## 3. This is a partial execution, and must be reported as one

Stage B is pre-registered as 300 fits: 30 assets x 5 models x 2 horizons x seed 2021.
No eight-hour budget on one 7,680 MiB P4 can finish it. The observed Stage A TimesNet
fit ran 2,230--2,519 s, and v1.2 §4 already extrapolates the 60 Stage B TimesNet fits to
roughly 42 GPU-hours on their own.

The budget therefore orders the queue by cost rather than by manifest order: the 240
non-TimesNet fits (10--20 s each on this device) run first, then TimesNet fits until the
budget floor is reached. Every fit that is not reached keeps the `planned` status it has
in `B_screen.manifest.csv`.

**Consequence for reporting.** Whatever this run produces is a *partial* Stage B. It
must not be described as "Stage B complete", and no screening decision may be taken from
it while fits remain `planned`, because the pre-registered multiplicity plan covers the
full 300-fit panel.

## 4. Interruption is not failure, and does not create a retry

A fit stopped by the budget floor is recorded as `interrupted`, and continuing it is
*not* a retry under v1.1 §4.2: the fit did not fail and its metrics were never
consulted. It continues **in the same attempt**, from its own per-epoch save point, so
no second attempt is created and no `retry_authorization` is needed. The interrupted
receipt is preserved as `receipt.interrupted-NN.json` rather than overwritten.

The save point is verified, not assumed: `tests/verify_phase1_recovery.py` runs a real
fit twice -- once straight through with no save point, once killed at an epoch boundary
and restarted -- and requires `pred.npy` and `true.npy` to agree byte for byte. A resume
that produced a different trajectory would be a second, unrecorded experiment, and that
test is what rules it out.

## 5. Not authorized

- **Stage C and Stage D remain locked.** This decision covers `B_screen` only; each of
  the others needs its own authorization under v1.2 §3.
- No recomputation, deletion, or rewriting of any existing attempt, receipt, or
  packaged artifact, in either Stage A result root.

## 6. Controls still open at the time of this decision

These are recorded as open rather than asserted as closed:

- **Immutable container digest (v1.2 §4) is still missing.** The worker exposes no
  Docker or Podman interface -- its root volume is managed by containerd
  (`PHASE1_P4_HOST_IDENTITY_SUPPLEMENT_2026-09-28.json`) -- so the digest is not
  capturable from inside the instance. What *is* bound is the GPU UUID and driver
  (`550.54.15`), the complete Python/PyTorch lock, and the source commit. This is a
  knowing, recorded shortfall, not a satisfied requirement.
- **Cross-root attempt visibility (deviation §9)** was scoped by v1.2 §1: the CPU
  Stage A root is historical evidence and is excluded from attempt-uniqueness counting.
  Stage B additionally writes to its own root, `reproduction/results/phase1/attempts/`,
  which is new and single, so no cross-root lineage question arises within it.
- `worker-side preflight` and the failure/retry lineage tests are satisfied by
  `reproduction/analysis/build_phase1_preflight_artifacts.py` and
  `tests/test_phase1_stage_runner.py` respectively, both green at this commit.
