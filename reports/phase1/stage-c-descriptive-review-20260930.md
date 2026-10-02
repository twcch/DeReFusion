# Stage C 描述性点估计复核（2026-09-30）

## 结果范围

本文件报告冻结 Stage C 配对 estimand `MSE_DeReFusion - MSE_revin-DLinear` 的描述性点估计，按冻结 cohort、预测 horizon、训练 seed 分层。负值表示该单元中 DeReFusion 的测试 MSE 点估计较低；此符号说明不作优越性推断。

本分析未合并 horizon 或 seed，也未计算 bootstrap 区间、p 值或多重性结论。冻结预注册要求 asset-cluster bootstrap，但初始预注册提交只要求不可变 seed，没有固定 seed 或重采样次数。因此下列值不能作为预注册的 Stage C 确认性主结论。

## 数据和配对核验

- 输入为冻结的 `C_confirmation.manifest.csv`，360行、360个唯一逻辑设置。清单 SHA-256：`bda784f0dd816939cdfd02e30109369df6a3e2e76aecd79aa8ee87b3910e6bfa`，与 Stage C 授权绑定一致。
- 使用已摄入的 RTX 5060 Ti `confirmatory_local_stage_c` 轨道。既有 artifact inventory 记录360份 receipt、360个唯一 receipt hash 被 consumer 接收、115个 intake pass、0 issues。
- 本次复核读取全部360组预测、真值和指标数组：均通过有限值与形状核对；1,080个 `pred.npy`、`true.npy`、`metrics.npy` 文件的实际 SHA-256 与 receipt 声明一致；重算 MSE 与 `metrics.npy` 中的 MSE 一致。
- 对30资产 × 2 horizon × 3 seed 的180个配对，模型两臂的 prediction-key hash 和真值文件 SHA 一致，载入的真值数组也逐元素相等。
- receipt 中的运行命令未含 `--inverse`（其默认值为 false）；结合冻结数据契约的 train-only `StandardScaler`，这里报告的是训练标准化目标空间的测试 MSE。
- cohort 内每个资产等权。每行先在相同 horizon 与 seed 下对每个资产计算两个模型的 MSE 差，再对该 cohort 的资产差取算术平均。资产方向计数以资产为单位，不是逐测试窗口的胜率。
- 可复现入口：[phase1_stage_c_descriptive_summary.py](../../reproduction/analysis/phase1_stage_c_descriptive_summary.py)。使用 Stage C 锁定环境执行：`python reproduction/analysis/phase1_stage_c_descriptive_summary.py`；脚本验证 manifest/receipt 绑定、三个数值文件的实际哈希、配对键/真值与 MSE，随后写出下方两份 CSV。

## 描述性点估计

| Cohort | Horizon | Seed | Assets | Mean paired delta | Assets with delta < 0 | Assets with delta > 0 |
|---|---:|---:|---:|---:|---:|---:|
| original-10 | 1 | 2022 | 10 | -0.009315 | 10 | 0 |
| original-10 | 1 | 2023 | 10 | -0.015518 | 10 | 0 |
| original-10 | 1 | 2024 | 10 | -0.021454 | 10 | 0 |
| original-10 | 24 | 2022 | 10 | -0.005482 | 7 | 3 |
| original-10 | 24 | 2023 | 10 | 0.004682 | 6 | 4 |
| original-10 | 24 | 2024 | 10 | 0.000922 | 4 | 6 |
| c1-20 | 1 | 2022 | 20 | -0.009161 | 19 | 1 |
| c1-20 | 1 | 2023 | 20 | -0.011696 | 20 | 0 |
| c1-20 | 1 | 2024 | 20 | -0.018900 | 20 | 0 |
| c1-20 | 24 | 2022 | 20 | -0.008638 | 10 | 10 |
| c1-20 | 24 | 2023 | 20 | -0.010120 | 13 | 7 |
| c1-20 | 24 | 2024 | 20 | -0.002854 | 11 | 9 |

包含两模型的 cohort 平均 MSE 与未舍入差值的逐层明细见 [cohort 汇总 CSV](stage-c-descriptive-estimates-20260930.csv)。180个资产 × horizon × seed 配对、各对的 prediction-key hash 与真值 SHA 见 [资产级配对 CSV](stage-c-asset-paired-estimates-20260930.csv)。本表保留 horizon 和 seed 分层，避免引入预注册未说明的合并权重。

## Gate table

| Check | Input | Status | Evidence | Next permitted action |
|---|---|---|---|---|
| Stage C 身份、覆盖与运行轨道 | 冻结清单、Stage C 授权、已审计 intake | eligible | 360/360设置；授权清单 SHA 一致；RTX 5060 Ti 单一确认性执行轨道；intake 0 issues。 | 可继续在该轨道做冻结目标所许可的复核。 |
| 配对点估计与 target 对齐 | 360组 `pred.npy`、`true.npy`、`metrics.npy` 和 receipts | eligible | 180个配对均有相同 prediction-key hash、真值 SHA 和逐元素相同真值；重算指标与保存 MSE 一致。 | 仅报告分层点估计和明确标记的描述性统计。 |
| 确认性区间与主结论 | `PHASE1_PREREGISTRATION.md` 及其初始 Git 提交 `6bbb045ef8da34fb6be1b05222347167859b9104` | requires-resolution | 方案要求 asset-cluster bootstrap CI，但未固定 seed 或重采样次数；本次不选择事后参数。 | 先查找仓库外可验证的事前记录。若不存在，保留本次结果为描述性结果；任何新确认性协议需透明修订并由独立确认支持。 |

## 解释边界

这些分层均值和资产方向计数是 Stage C 输出的描述性摘要。它们没有回答预注册的区间判据是否成立，也不构成模型优越性结论。Stage D 仍是独立的后续阶段，不能用于改写 Stage C 的解释。
