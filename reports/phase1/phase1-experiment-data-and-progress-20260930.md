# Phase 1 实验数据与进度总览（2026-09-30）

## 摘要

本报告将 Phase 1 的冻结清单、已验收产物、分轨运行时间、Stage C 配对点估计和后续门控汇总到一个可审计入口。Stage A 的 15 项 GPU 校准完成；Stage B 的 300 个冻结逻辑设置均已有合格产物，但来自 P4 主轨和 RTX 8000 非确认性补充轨道，不能据此形成跨硬件统一模型排名；Stage C 的 360 项确认性训练与原始数组复核完成，当前只报告分层描述性点估计；Stage D 的 144 项仍为 planned，且没有专属执行授权。

本报告不包含 Stage B 测试误差比较，不把 Stage C 点估计写成优越性结论，也不以跨硬件等效为前提。原始预测、真值和 checkpoint 留在本地的 attempt 目录；远程仓库保存小型清单、报告、摘要 CSV 和可复现分析脚本，不把大体积原始结果作为本次提交对象。

## 阶段进度

| 阶段 | 冻结规模 | 当前核实状态 | 可支持的结论 |
|---|---:|---|---|
| Stage A calibration | 15 fits | GPU calibration receipts 15/15 `calibration_pass` | 执行校准完成；校准损失不作科学比较。另有 CPU 旁路校准 13 pass、1 resource-blocked、1 项无 receipt，不计入正式 GPU 校准。 |
| Stage B screen | 300 settings | P4 245/300 + RTX 8000 补充 55/55；跨轨重叠 0；并集 300/300；intake issues 0 | 逻辑身份与产物覆盖齐全；P4 仍按原授权标记 partial，RTX 8000 仍是单独非确认性补充轨道。无统一硬件完整面板，不作跨轨排名或等效结论。 |
| Stage C confirmation | 360 fits | RTX 5060 Ti 单一执行轨道 360/360；intake issues 0 | 输入与配对数据通过完整性检查；本报告只给按 cohort、horizon、seed 分层的描述性点估计。 |
| Stage D temporal robustness | 144 fits | 冻结 manifest 144/144 `planned`；未发现 D attempt 或专属授权 | 未运行；须独立设门并取得执行授权。 |

## 实验数据和完整性

### Stage B

- 冻结 `B_screen.manifest.csv`：300 行、300 个唯一逻辑设置；SHA-256 `4bc275cf58922d18cab6840d31adb59368aea9304964b54d3f2a3e156ba63ea5`。
- P4 intake：245/300 eligible、0 issues。其授权仍是部分执行；不把 RTX 8000 补充任务改写成 P4 已完成。
- RTX 8000 补充 manifest：55 个唯一的剩余 TimesNet 设置；SHA-256 `927f7d486b32e2069f97cc985c32784c0ca9a3b94b8a9e06df15b024cab3f24c`。最新 intake 55/55 eligible、0 issues，来源为独立非确认性轨道。
- 300 份 receipt 与 `run.log` 按 identity 对齐；300 个日志当前 SHA-256 与 receipt/intake 声明一致；每份日志恰有一个可解析的 `train_time`。运行时间仅按 P4 和 RTX 8000 分开汇总。
- 冻结 B 清单、分轨覆盖和许可结论见 [Stage B 覆盖审计](stage-b-coverage-audit-20260930.md)；训练时长与逐设置哈希见 [技术收口报告](stage-b-technical-closeout-20260930.md)、[分轨时长 CSV](stage-b-runtime-by-track-20260930.csv) 和 [逐设置 ledger](stage-b-runtime-setting-ledger-20260930.csv)。

### Stage C

- 冻结 `C_confirmation.manifest.csv`：360 行、360 个唯一逻辑设置；SHA-256 `bda784f0dd816939cdfd02e30109369df6a3e2e76aecd79aa8ee87b3910e6bfa`，与执行授权绑定一致。
- 读取并核验 360 组 `pred.npy`、`true.npy` 和 `metrics.npy`：1,080 个文件实际 SHA-256 与 receipt 声明一致；重算 MSE 与保存值一致。
- 180 组 DeReFusion/DLinear 配对的 prediction-key hash 和真值 SHA 一致，真值数组逐元素相同。报告量为 train-only 标准化目标空间的测试 MSE 差 `MSE_DeReFusion - MSE_revin-DLinear`。
- 已按 `original-10`、`c1-20`、horizon 和 seed 分层计算点估计；没有合并 horizon/seed，没有 bootstrap 区间或 p 值。Stage C 的预注册未固定 bootstrap seed、重采样次数、区间算法和跨 horizon/seed 主汇总规则，因此目前不能声称确认性区间判据成立。
- 12 个分层点估计见 [Stage C 描述性报告](stage-c-descriptive-review-20260930.md) 和 [cohort 汇总 CSV](stage-c-descriptive-estimates-20260930.csv)；180 个资产级配对与 hash 见 [资产级 CSV](stage-c-asset-paired-estimates-20260930.csv)。

### Stage D 与本机 GPU

- `D_temporal_robustness.manifest.csv` 有 144 行，均为 `planned`；SHA-256 `04ce9d9bba9ccc3590dbdbb63698395a7a5b6c7169f7ffd99fc92011b62f0f7b`。仓库中未找到 Stage D 专属授权或 attempt。
- 当前 Windows 主机枚举到 Intel Arc 140T。隔离环境中的 PyTorch `2.13.0+xpu` 能发现该设备，并完成张量前向、反向和同步；这只是运行时探针，不是 Phase 1 模型校准。现有训练入口仍只实现 CUDA/MPS/CPU，旧 RTX 5060/8000 授权不能用于 Intel Arc。
- XPU 可行性与适配门见[本机 XPU 核查](stage-b-local-intel-xpu-readiness-20260930.md)。

## Stage C 描述性点估计摘要

负值表示该冻结层内 DeReFusion 的 MSE 点估计较低；仅为描述，不代表优越性推断。以下保留 horizon 和 seed，不作事后合并：

| Cohort | Horizon | Seed 2022 | Seed 2023 | Seed 2024 |
|---|---:|---:|---:|---:|
| original-10 | 1 | -0.009315 | -0.015518 | -0.021454 |
| original-10 | 24 | -0.005482 | 0.004682 | 0.000922 |
| c1-20 | 1 | -0.009161 | -0.011696 | -0.018900 |
| c1-20 | 24 | -0.008638 | -0.010120 | -0.002854 |

资产方向计数、逐层两模型均值和原始精度值保存在 Stage C CSV 中。不能把上述点估计改写成 CI、p 值或确认性模型优越结论。

## 报告和复现入口

| 内容 | 仓库文件 |
|---|---|
| Stage B 身份/覆盖/许可审计 | `reports/phase1/stage-b-coverage-audit-20260930.md` |
| Stage B 运行时收口与逐设置证据 | `reports/phase1/stage-b-technical-closeout-20260930.md`、`stage-b-runtime-by-track-20260930.csv`、`stage-b-runtime-setting-ledger-20260930.csv` |
| Stage C 配对点估计与完整性复核 | `reports/phase1/stage-c-descriptive-review-20260930.md`、`stage-c-descriptive-estimates-20260930.csv`、`stage-c-asset-paired-estimates-20260930.csv` |
| 后续门控与本机 XPU 状态 | `reports/phase1/phase1-next-steps-gate-20260930.md`、`stage-b-local-intel-xpu-readiness-20260930.md` |
| 可复现分析脚本 | `reproduction/analysis/phase1_stage_b_technical_closeout.py`、`phase1_stage_c_descriptive_summary.py` |
| 历史 artifact inventory 快照 | `reports/phase1/phase1-artifact-inventory-20260929T160534Z.md`、`phase1-artifact-inventory-20260929T161118Z.md`；后者是 16:11 UTC 的较新快照，但其中 RTX 8000 仅摄入 25/55，已由 2026-09-30 的 55/55 收口报告取代。 |

复算命令（在对应锁定环境、且本地原始产物目录可用时）：

```powershell
python reproduction/analysis/phase1_stage_b_technical_closeout.py
python reproduction/analysis/phase1_stage_c_descriptive_summary.py
```

分析脚本校验 manifest/receipt identity、输入哈希及数值一致性；Stage B 脚本不读取测试误差作排名。原始 attempt、checkpoint、预测和真值数组不由本次报告或摘要文件替代。

## 下一步

1. 查询是否存在早于 Stage C 结果查看、且可验证来源的 bootstrap/aggregation 记录；若没有，当前 Stage C 保持描述性，不从已观察结果事后挑选统计参数。
2. 如需确认性结论，另行预先固定 estimand 汇总权重、asset-cluster bootstrap 配置、精度/成功判据和独立确认方案；功效、精度和资源阈值当前均为 `TBD`。
3. Stage D 等 Stage C 解释边界、独立授权和资源方案齐备后再运行。
4. 若计划使用本机 Intel Arc 重做完整 Stage B，先为 XPU 增加独立、版本化训练后端，完成 Stage A 校准和资源核查；新结果按新轨道保存，不与已有 NVIDIA/P4 分数池化。

## 解释边界

本报告只汇总仓库内实际冻结清单、intake、receipt、日志和数值数组核验结果。Stage B 的 300/300 表示逻辑身份和合格产物覆盖完整，不表示单一硬件面板完整；Stage C 的 12 个数值是描述性分层点估计；Stage D 尚未执行。任何更强结论都需要现有材料之外的事前统计规则和相应证据。
