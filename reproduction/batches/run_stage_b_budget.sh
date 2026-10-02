#!/usr/bin/env bash
# Phase 1 Stage B under one wall-clock budget.
#
# Stage B is 300 fits and one ~8 h budget on a single P4 cannot finish it: the
# 60 TimesNet fits alone extrapolate to ~42 GPU-hours (v1.2 §4).  The budget is
# therefore spent in cost order -- the 240 cheap fits first, then TimesNet until
# the floor is reached.  Fits the budget never reaches keep `planned` status in
# B_screen.manifest.csv; this is a partial execution and is reported as one.
#
# Work is durable at two levels, so an interruption costs at most one epoch of
# one fit:
#   * a completed fit is packaged and hash-verified as its own attempt;
#   * a fit cut off mid-training resumes from its own per-epoch save point, in
#     the same attempt -- not a retry, because it never failed.
#
# Re-running this script continues where it stopped.
set -u

REPO=/data/coding/DeReFusion
PYTHON=/data/miniconda/envs/derefusion-p1/bin/python
AUTH="$REPO/reproduction/handoff/phase1_outsourcing/authorization_stage_b.json"
FPRINT="$REPO/reproduction/handoff/phase1_outsourcing/environment_fingerprint.json"
CONTROL=/data/phase1_control
BUDGET_HOURS="${BUDGET_HOURS:-8}"
FAST_MODELS="revin-DLinear,DeReFusion,revin-PatchTST,revin-iTransformer"

mkdir -p "$CONTROL"
cd "$REPO" || exit 1

COMMON=(--stage B_screen --execute --device cuda --enable-recovery
        --authorization "$AUTH" --environment-fingerprint "$FPRINT")

START=$(date +%s)
echo "=== Stage B budget run: ${BUDGET_HOURS} h, started $(date -u +%FT%TZ) ==="
echo "    commit $(git rev-parse --short HEAD)   host $(hostname)"

echo
echo "--- phase 1/2: the 240 affordable fits ---"
# --resume-interrupted matters here too: without it a fit cut off by the
# timeout would block the whole phase on the next run instead of continuing
# its own attempt from its save point.
"$PYTHON" reproduction/batches/run_phase1_stages.py "${COMMON[@]}" \
    --models "$FAST_MODELS" --resume-interrupted \
    --fit-timeout-hours 0.5 \
    --stop-below-seconds 300
echo "--- phase 1 exit=$? elapsed=$(( $(date +%s) - START ))s ---"

ELAPSED=$(( $(date +%s) - START ))
REMAIN=$("$PYTHON" -c "print(max(0.0, ${BUDGET_HOURS} - ${ELAPSED}/3600.0))")
echo
echo "--- phase 2/2: TimesNet for the remaining ${REMAIN} h ---"

# Below ~45 min there is no room to finish a 42-minute fit, so starting one
# would waste the remainder.  --resume-interrupted picks up a fit an earlier
# budget cut off, continuing it rather than beginning a second attempt.
if "$PYTHON" -c "import sys; sys.exit(0 if ${REMAIN} > 0.85 else 1)"; then
    "$PYTHON" reproduction/batches/run_phase1_stages.py "${COMMON[@]}" \
        --models revin-TimesNet --resume-interrupted \
        --max-hours "$REMAIN" \
        --fit-timeout-hours 1.5 \
        --stop-below-seconds 2700
    echo "--- phase 2 exit=$? elapsed=$(( $(date +%s) - START ))s ---"
else
    echo "--- phase 2 skipped: ${REMAIN} h left, not enough for a TimesNet fit ---"
fi

echo
echo "=== budget run finished $(date -u +%FT%TZ), $(( ($(date +%s) - START) / 60 )) min total ==="
echo "Unreached fits keep status=planned. Re-run this script to continue."
