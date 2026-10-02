# Phase 1 后续工作 Gate（2026-09-30）

本备忘基于仓库内冻结方案、现有 intake、授权和本地运行产物。它记录可继续的动作与需要新证据的决策；不修改 active protocol，不授权新训练。

## Gate table

| Check | Input | Status | Evidence | Next permitted action |
|---|---|---|---|---|
| Stage B 设置身份和产物覆盖 | 冻结 B manifest、P4 intake、RTX8000 完整 intake | eligible | 300/300 唯一逻辑 ID 由 P4 245 + RTX8000 55 恰好覆盖；两轨重叠0。 | 保留为覆盖完成记录，按执行轨道分开陈述。 |
| Stage B 跨硬件筛选与统一排名 | Stage B 决策、GPU bridge 方案、两个 intake 的 pooling/ranking 字段 | requires-resolution | P4 与 RTX8000 属于不同硬件/软件执行轨道；intake 禁止 pooling/ranking；等价精度界限为 `TBD`。 | 不做跨轨模型排序。若需要共同科学筛选，须先有明确授权和事前可比性方案；既有混合轨结果不自动满足这一条件。 |
| Stage C 描述性复核 | 冻结 C manifest、授权、360个 receipt 与预测/真值/指标数组 | eligible | 已核验360项、180组配对、1,080个数值文件 SHA-256；逐对 prediction keys、真值和 MSE 对齐。分层点估计见 `stage-c-descriptive-review-20260930.md`。 | 继续保存和复核描述性输出，不将其写成确认性模型优越结论。 |
| Stage C 确认性区间 | 预注册初始提交 `6bbb045ef8da34fb6be1b05222347167859b9104`、Stage C 执行决策 | requires-resolution | 预注册要求 asset-cluster bootstrap 和 immutable seed，但没有给出 seed、重采样次数、CI 算法或 across-horizon/seed 的主汇总规则。 | 不从当前已打开的结果中选统计设置后声称确认性。若存在可核验的事前记录，可按其复核；若没有，将当前分析保持描述性，并为新的独立确认预先写全estimand汇总、bootstrap配置、决策规则。 |
| Stage D 执行与分析 | `D_temporal_robustness.manifest.csv`、Stage C 决策、授权目录、Phase 1 产物根目录 | requires-resolution | 冻结 D manifest 为144行、144行均为 `planned`；SHA-256 `04ce9d9bba9ccc3590dbdbb63698395a7a5b6c7169f7ffd99fc92011b62f0f7b`。授权记录中未找到 Stage D 专属授权，结果目录中也未见 D attempt。Stage C 执行决策明确 D 另行设门，预注册将 D 排在 A–C 之后。 | 不启动 Stage D。待 Stage C 解释边界及 D 的独立授权、资源方案与完整预检具备后再运行。 |
| 当前本机 GPU 是否可用于已有 CUDA 轨道 | Windows GPU 枚举、隔离 XPU 探针、训练设备代码、5060 授权 | requires-resolution | 本机枚举为 Intel Arc 140T；独立 PyTorch `2.13.0+xpu` 环境已通过设备发现及张量前后向探针，但仓库训练入口仅实现 CUDA/MPS/CPU。`.venv` 仍为 CPU PyTorch；既有本机授权绑定的是另一块 RTX 5060 Ti 和 CUDA 环境。详见 `stage-b-local-intel-xpu-readiness-20260930.md`。 | 不启动现有 RTX 授权轨道。下一步是版本化 XPU 后端适配、Stage A 保真校准和资源核查，再评估新的、分轨非确认性 Stage B 运行。 |

## 建议推进顺序

1. 将 Stage B 覆盖审计与 Stage C 描述性结果作为当前只读结项证据，保留两轨标识和描述性限定。
2. 对现有 Stage C，若能找到在结果查看前形成的 bootstrap / aggregation 记录，登记来源并复现；若找不到，不再尝试从现有结果构造确认性区间。
3. 若研究目标仍需要确认性主张，准备独立确认方案。方案至少固定：每个资产内 horizon/seed 的汇总权重、每 cohort 的 asset-cluster bootstrap 算法、seed、重采样次数、区间类型和成功判据。现有冻结文件未支持这些数值，全部先标为 `TBD`，不能从其他研究脚本借值。
4. 对独立确认先做精度/功效与资源分析，再取得新的执行授权。最小有意义效应或目标区间宽度也未在当前预注册中明确，不能诚实地从现有结果倒推出所需样本量或运行预算。
5. Stage D 暂不启动；它有独立144-fit网格，且缺少专属执行授权和产物。
6. 本机 Intel Arc 不套用 RTX 5060/8000 授权；隔离 XPU 设备探针已通过，但完成后端适配与 Stage A 校准前，不提交 Phase 1 GPU 训练。

## 明确不在本次范围内

- 不重跑或改写任何 Stage A–C 实验 attempt、receipt、checkpoint、预测/真值数组。
- 不将 RTX8000 的 Stage B 补充结果视作 P4 的延续，也不 pooled ranking。
- 不补造 Stage C bootstrap 参数，不以未预注册的跨 horizon/seed 合并均值代替主 estimand。
- 不根据已观察到的 Stage C 描述性点估计选资产、模型、参数或 Stage D 资源。
