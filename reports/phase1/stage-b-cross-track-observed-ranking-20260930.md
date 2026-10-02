# Stage B 全量已交付结果排序（跨轨描述性视图，2026-09-30）

## 用途与解释

按用户要求，把冻结清单中的 300 个已交付 Stage B attempt 放在一张表中，对**实际训练产物（模型 + 执行轨道）**作统一的观察值排序。本报告不声称硬件等价，也不把它解释为纯模型架构效应、确认性排名或显著性结论；它是现有实验执行结果的描述性对照。

这是用户本轮明确要求生成的独立描述性视图，不回写或追认原始 intake/authorization 中 `ranking_authorized=false` 的字段，也不修改冻结协议。原授权的跨轨推断限制继续有效。

这里不把 P4 和 RTX8000 视为可交换执行环境。四个非 TimesNet 模型的 240 项都来自 P4；TimesNet 的 60 项中，5 项来自 P4、55 项来自 RTX8000。RTX8000 与 P4 没有相同的 TimesNet 逻辑设置可用于直接估计这两条轨道的 TimesNet 训练差异。因此完整榜单虽然可以按相同目标键和 MSE 算术汇总，但 TimesNet 的名次仍可能受到执行轨道影响。

## 验证和计算口径

- 固定使用全部 300 个冻结逻辑设置，没有按分数选择资产、模型、horizon 或 attempt。
- 300/300 receipt 均为 `completed_unreviewed`；逐个重验 `pred.npy`、`true.npy`、`metrics.npy` 的 SHA-256，并从数组重算 MAE/MSE/RMSE，与保存的指标前三项在 `rtol=2e-5, atol=2e-7` 内一致。
- 对 60 个 asset × horizon 组，五模型均有一项；prediction-key SHA、true SHA、dataset SHA、split ID 和测试元素数逐组一致。
- 在每个冻结 cohort × horizon 单元内，先逐资产计算 fit 的归一化测试 MSE，再对资产等权取均值。cohort/horizon 不合并；Stage B 只有 seed 2021。
- 排名仅是四个分层单元内平均 MSE 的排序。未计算 p 值、置信区间或校正后的 inferential ranking。

复现入口：[phase1_stage_b_cross_track_observed_ranking.py](../../reproduction/analysis/phase1_stage_b_cross_track_observed_ranking.py)。它产生逐资产指标、全量观察值排序、成对描述统计及 TimesNet 敏感性 CSV。执行轨道标签保留在输出中。

## 全量观察值排序

分数为每资产等权平均标准化测试 MSE，越低越好。TimesNet 的 P4/RTX8000 构成列在表后；所有其它模型行均为 P4。

| Cohort | Horizon | Rank | Model | Mean asset MSE | Assets | P4 / RTX8000 fits |
|---|---:|---:|---|---:|---:|---:|
| c1-20 | 1 | 1 | revin-PatchTST | 0.019661 | 20 | 20 / 0 |
| c1-20 | 1 | 2 | revin-iTransformer | 0.024705 | 20 | 20 / 0 |
| c1-20 | 1 | 3 | revin-TimesNet | 0.032996 | 20 | 3 / 17 |
| c1-20 | 1 | 4 | DeReFusion | 0.046804 | 20 | 20 / 0 |
| c1-20 | 1 | 5 | revin-DLinear | 0.048731 | 20 | 20 / 0 |
| c1-20 | 24 | 1 | revin-PatchTST | 0.158807 | 20 | 20 / 0 |
| c1-20 | 24 | 2 | DeReFusion | 0.179203 | 20 | 20 / 0 |
| c1-20 | 24 | 3 | revin-DLinear | 0.179487 | 20 | 20 / 0 |
| c1-20 | 24 | 4 | revin-iTransformer | 0.179569 | 20 | 20 / 0 |
| c1-20 | 24 | 5 | revin-TimesNet | 0.182391 | 20 | 2 / 18 |
| original-10 | 1 | 1 | revin-PatchTST | 0.020603 | 10 | 10 / 0 |
| original-10 | 1 | 2 | revin-iTransformer | 0.025981 | 10 | 10 / 0 |
| original-10 | 1 | 3 | revin-TimesNet | 0.044630 | 10 | 0 / 10 |
| original-10 | 1 | 4 | DeReFusion | 0.051074 | 10 | 10 / 0 |
| original-10 | 1 | 5 | revin-DLinear | 0.051677 | 10 | 10 / 0 |
| original-10 | 24 | 1 | revin-PatchTST | 0.185569 | 10 | 10 / 0 |
| original-10 | 24 | 2 | revin-DLinear | 0.209801 | 10 | 10 / 0 |
| original-10 | 24 | 3 | DeReFusion | 0.214574 | 10 | 10 / 0 |
| original-10 | 24 | 4 | revin-iTransformer | 0.226294 | 10 | 10 / 0 |
| original-10 | 24 | 5 | revin-TimesNet | 0.234772 | 10 | 0 / 10 |

PatchTST has the lowest observed mean in all four strata. This is a point-estimate pattern over one training seed, not evidence that PatchTST is statistically superior. The primary DeReFusion–DLinear ordering also varies by stratum: DeReFusion is slightly lower for c1-20 at both horizons and original-10 at horizon 1, while DLinear is lower for original-10 at horizon 24. These are descriptive MSE means only.

## TimesNet rank sensitivity to an unknown RTX8000 shift

To show exactly how much the mixed-track TimesNet line would need to move to tie the best observed P4 model (PatchTST), let `δ` be a hypothetical multiplicative change applied only to RTX8000 TimesNet MSEs. P4 TimesNet observations are held fixed. The break-even value solves:

`[sum(P4 TimesNet MSE) + (1 + δ) × sum(RTX8000 TimesNet MSE)] / asset_count = mean(PatchTST MSE)`.

| Cohort | Horizon | RTX8000 TimesNet fits | Best other model | Hypothetical RTX8000 MSE shift for TimesNet to tie |
|---|---:|---:|---|---:|
| c1-20 | 1 | 17 | revin-PatchTST | -44.1% |
| c1-20 | 24 | 18 | revin-PatchTST | -13.7% |
| original-10 | 1 | 10 | revin-PatchTST | -53.8% |
| original-10 | 24 | 10 | revin-PatchTST | -21.0% |

These are break-even scenarios, not estimated GPU effects, correction factors, or equivalence limits. The separate 34-pair P4–RTX5060 bridge does not connect to the RTX8000 TimesNet runs and cannot supply an RTX8000 correction. In particular, the c1-20 horizon-24 order is sensitive to a hypothetical shift smaller than the largest difference seen in that unrelated bridge; that bridge maximum is not a valid RTX8000 bound.

## Data outputs and remaining limit

- [All 300 asset-level metrics and track provenance](stage-b-cross-track-asset-metrics-20260930.csv)
- [Four-stratum observed ranking](stage-b-cross-track-observed-ranking-20260930.csv)
- [All model-pair asset-level descriptive differences](stage-b-cross-track-pairwise-descriptive-20260930.csv)
- [TimesNet break-even sensitivity](stage-b-cross-track-timesnet-sensitivity-20260930.csv)
- Existing P4-only four-model analysis remains available in [the P4 subpanel review](stage-b-p4-subpanel-descriptive-review-20260930.md).

This provides one unified view of all 300 observed artifacts while preserving the hardware labels. A model-only five-way conclusion remains unidentified from these data because TimesNet execution track is entangled with model for 55 of its 60 settings, and no RTX8000/P4 same-setting TimesNet bridge was run. No additional training was performed.
