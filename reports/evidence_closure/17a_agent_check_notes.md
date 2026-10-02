# 17a · Agent check notes on the Astra NS-direction assessment (17)

**Archived verbatim:** `17_ns_directions_astra.md` (18 KB, unmodified — one word was not changed).
**Purpose of this file:** compliance check against the project's hard constraints, two factual
corrections verified in code, and what (if anything) is folded back into the plan.
**No experiment is authorised here. Nothing in `11`/`12a`/`12b` is altered.**

## 1. Compliance check — clean, with one wording note

| Constraint | Result |
|---|---|
| Does not assume NS works | ✅ opens with *"可用性判断：低"* and labels every direction *"假设设计，不是有效性证据"* |
| Does not request experiments now | ✅ explicitly *"不是立即执行的实验方案"*; gives paper commitments only |
| Does not touch frozen rules | ✅ states the Gate-A correlation thresholds **must not** be transplanted into new module criteria |
| No router / gate / MoE / attention / pressure | ✅ excludes them; also excludes free forcing networks and "flux coupling" renaming of input-dependent reweighting |
| Terminology | ✅ *temporal NS-inspired nonlinear operator*; and it explicitly retires "markets are fluids"-type inference by noting the blowup theorem provides no forecasting consequence |
| Falsifiable statements | ✅ every direction carries a falsification rule; unfalsifiable constraints (e.g. arbitrary forcing absorbing error, unmeasured conservation) are rejected outright |
| Negative evidence honoured | ✅ keeps the no-discriminative-power jump ratio and the routing negative result intact |

**Wording note (no action needed):** it argues that a *fixed linear smoothing* cannot claim
independent value because ordinary convolution realises it. That matches our own reading; it is
recorded as agreement, not as a new claim.

## 2. Corrections verified in code (these change the plan's numbers)

### 2.1 The twelve structural states are an **evaluation stratification**, not a training multiplier

Astra flagged that "12 states" cannot be treated as a 12× training count. Verified in
`reproduction/analysis/structure_routing_capacity_check.py`:

```
for seed in S.SEEDS:                       # 3 seeds
    train_linear_once(...)                 # 1 linear fit per (asset, seed)
    for w in WIDTHS:                       # 3 widths
        train_mlp_width(...)               # 1 MLP fit per (asset, seed, width)
        for sname, mask in states.items(): # <- states are EVALUATION masks over the same fit
```

and in the raw output (`operator-regime-capacity_sv.csv`, 306 rows): every
`(asset, width, seed)` group contains 12–13 state rows sharing the same `params_L/params_N`.

**Consequence — the F1 resource estimate in `16` must be corrected:**

| Plan figure | Correct value |
|---|---|
| `16` §3 "7 × 3 × 3 × 12 × 2 = **1,512 operator fits**" | **7 × 3 × 3 × 2 = 126 fits** (12 states are eval strata) |
| measured cost ≈2–3 s per fit (48 fits ≲2 min) | **126 fits ≈ 5–7 min** single-threaded |

So branch F1 is *cheap* (minutes), not a 1.5-hour job. It remains a breadth test, not a
computation bottleneck. `16a` §4's tightening of the F1 criterion is unaffected — with the panel
this cheap, requiring **≥2 additional seed- and width-stable assets** is entirely affordable.

### 2.2 Parameter matching is exact only at width 23

The linear map (9,240 params) is parameter-matched to the MLP **only at width 23** (9,431); at
widths 64 (26,200) and 128 (52,376) it is not. Astra's note is correct and is now recorded
explicitly: any future claim must say "capacity-matched at width 23 / capacity-varying at 64 and
128" rather than implying a matched comparison at every width.

## 3. What I fold back into the plan (and what I do not)

**Fold in (documentation only):**

1. the F1 figure correction (126 fits, minutes) in `16`;
2. the parameter-matching caveat (2.2);
3. Astra's framing sentence for what a *mechanism* claim would require — a pre-fixed, strictly-past
   structural marker that explains **when** the gain appears, plus a removal test (remove the
   claimed structure and the gain disappears), and reproduction attempts by ordinary
   convolution/SSM given equal features and budget;
4. its exit wording for total failure (recorded as the project's retirement language):
   > *"在本项目的冻结预测任务、数据及已覆盖预算范围内，所列 NS-inspired 结构未提供超出同参数量
   > MLP、普通时序卷积和 SSM 的可验证预测价值。停止将 NS-inspired 作为本项目后续架构选择和结果
   > 解释的依据。"*

**Do not fold in:** none of the six directions is promoted to a planned experiment. The only
direction Astra itself would keep is the second-order multi-scale quantity
`q = C(x²) − (Cx)²` — and it immediately notes that ordinary convolution plus squaring computes it.
It therefore enters the record as **the cheapest-to-falsify question**, not as a commitment.

## 4. Status

- Astra's usability verdict (**LOW**) is consistent with the frozen gate: the NS question remains a
  downstream hypothesis, and the immediate next step is still the Gate-A verdict plus (if
  CONDITIONAL) independent prospective validation.
- No experiment is authorised. Any later proposal must arrive as its own pre-registration, separate
  from `11`, with the four bridge elements (marker, prospective detector, capacity-matched
  baseline, falsification rule).
