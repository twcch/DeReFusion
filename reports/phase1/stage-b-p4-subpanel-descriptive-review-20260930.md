# Stage B P4 四模型子面板描述性复核（2026-09-30）

## 目的与边界

复用既有 P4 Stage B 产物，描述四个非 TimesNet 模型在 P4 同轨子面板上的测试误差。该复核覆盖固定的全部 240 个设置（30 个冻结资产 × 4 个模型 × 2 个 horizon，seed 2021），不按结果筛选资产或模型。

这是限定执行轨道的描述性补充，不替代预注册的 300-fit 五模型 Stage B，不合并 RTX8000 结果，不作显著性检验、确认性优越性结论或资源调整依据。未汇总 horizon 或 cohort；cohort 内按资产等权。差值方向为 `model A MSE - model B MSE`，负值表示 A 的点估计较低。

## 数据与复算核验

- 输入为冻结 `B_screen.manifest.csv` 中的 240 个非 TimesNet 设置及 P4 intake 已接收的 attempt-01 包。
- 240/240 receipt 身份、`completed_unreviewed` 状态、dataset SHA 和 prediction-key SHA 均符合固定设置；`pred.npy`、`true.npy`、`metrics.npy` 的实际 SHA-256 与 receipt 一致。
- 60 个 asset × horizon 配对组中，全部 60 组均有四个模型；prediction-key SHA、true.npy SHA、数据 SHA、split ID 和测试数组形状逐组一致。
- 240/240 组预测与真值数组均为有限值；从数组重算的 MAE、MSE、RMSE 与保存 `metrics.npy` 的前三项在容差 `rtol=2e-5, atol=2e-7` 内一致。
- `run.py` 中 `--inverse` 默认为 false，模型按 train-only 标准化目标输出评估。这里使用每个 fit 的标准化目标空间 MSE/MAE/RMSE；不使用可能受标准化后目标接近零影响的 MAPE/MSPE。

可复现脚本：[phase1_stage_b_p4_subpanel_descriptive.py](../../reproduction/analysis/phase1_stage_b_p4_subpanel_descriptive.py)。脚本只读原始数组与 receipt，输出以下两个文件：

- [逐资产指标与来源哈希](stage-b-p4-subpanel-asset-metrics-20260930.csv)
- [cohort/horizon 描述统计及完整成对差值](stage-b-p4-subpanel-descriptive-20260930.csv)

复算命令：`.venv\Scripts\python.exe reproduction/analysis/phase1_stage_b_p4_subpanel_descriptive.py`。

## 每资产等权的标准化测试 MSE 均值

每行在冻结 cohort 内对资产等权平均，cohort/horizon 单独报告。下表是描述性点估计，未构造区间。

| Cohort | Horizon | revin-DLinear | DeReFusion | revin-PatchTST | revin-iTransformer | Assets |
|---|---:|---:|---:|---:|---:|---:|
| original-10 | 1 | 0.051677 | 0.051074 | 0.020603 | 0.025981 | 10 |
| original-10 | 24 | 0.209801 | 0.214574 | 0.185569 | 0.226294 | 10 |
| c1-20 | 1 | 0.048731 | 0.046804 | 0.019661 | 0.024705 | 20 |
| c1-20 | 24 | 0.179487 | 0.179203 | 0.158807 | 0.179569 | 20 |

在这四个固定 cohort × horizon 单元中，PatchTST 的资产等权 MSE 点估计均最低。此描述不表示统计显著或跨执行栈普遍优越；本面板只有一个训练 seed，且 Stage B 的原始用途包括运行可行性与 gross-failure screen。

## 主模型对的 P4 seed-2021 描述性差值

| Cohort | Horizon | Mean asset-level `DeReFusion − DLinear` MSE | DeReFusion lower | DLinear lower | Tied |
|---|---:|---:|---:|---:|---:|
| original-10 | 1 | -0.000602 | 5/10 | 5/10 | 0 |
| original-10 | 24 | +0.004774 | 8/10 | 2/10 | 0 |
| c1-20 | 1 | -0.001928 | 13/20 | 7/20 | 0 |
| c1-20 | 24 | -0.000285 | 9/20 | 11/20 | 0 |

符号按冻结 MSE 差值定义。该表不合并 Stage C seeds；Stage C 使用另一执行轨道和 seeds 2022–2024，不能用两套结果的方向差异单独归因于随机 seed 或硬件。

## 当前可用结论

1. 这 240 个 P4 fit 不需要为了弥补配对或目标对齐问题而重跑；原始计算可直接复用。
2. P4 轨道本身可形成四模型、两 horizon、两 cohort 的完整描述性误差摘要。当前结果给出了有用的 baseline 上下文，但不构成正式模型排名。
3. RTX8000 的 55 个 TimesNet fit 仍只用于独立补充轨道；本复核没有读取或合并其误差。
4. 若要把本结果提升到确认性筛选，必须另行冻结比较 family、Holm 家族范围、估计量汇总规则和区间方法；不得利用本表模式反向挑选资产、模型、horizon 或后续计算资源。

## 统计门控

| 检查 | 状态 | 下一步 |
|---|---|---|
| 固定 240-fit P4 身份及产物 | eligible | 作为只读、同轨的描述性面板存档 |
| 四模型预测键、目标、split 配对 | eligible | 可以做上述限范围 MSE 描述 |
| 多重性校正与确认性比较家族 | requires-resolution | 既有 Stage B 覆盖的是五模型计划；本补充不改写原 family，不报告检验或排名 |
| 对全五模型 Stage B 作单轨比较 | requires-resolution | P4 缺 55 个 TimesNet 设置；保留两条轨道，除非有新的事前可比性设计或完整共同轨道 |
| Stage C 确认性 bootstrap | requires-resolution | 仍需找回可验证的事前参数记录；若不存在，Stage C 继续限定为描述性点估计 |
