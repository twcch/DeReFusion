# Phase 1 运算产物索引与差异盘点

- 快照时间：2026-09-29T16:11:18.171793+00:00
- 仓库 HEAD：`f6093f1d85f13613dfe1a3ac0d25b554ca83ccd3`
- 范围：本地 Phase 1 收据化 attempt、已有摄入/审计摘要及 TSLib `results/` 原始输出目录数。
- 安全边界：只读取已发布 attempt；不扫描 `incoming/`、不改写/移动/删除 attempt、receipt、checkpoint 或预测数组。
- 哈希边界：CSV 记录 receipt 的 SHA-256 及 receipt 声明的 artifact 哈希；本次不重复读取/重哈希大型二进制。下表引用已完成的 intake/审计核验。

## 来源目录清单

| 来源 | 计算轨道 | receipts | 状态计数 |
|---|---|---:|---|
| `reproduction/results/phase1/calibration` | RTX 8000 / GPU calibration | 15 | calibration_pass: 15 |
| `reproduction/results/phase1/calibration-cpu` | CPU calibration | 14 | calibration_pass: 13, resource_blocked: 1 |
| `reproduction/results/phase1/p4-partial-20260929/attempts` | P4; partial B_screen | 245 | completed_unreviewed: 245 |
| `reproduction/results/phase1/local-5060-bridge-v1` | RTX 5060 Ti; nonconfirmatory bridge | 34 | completed_unreviewed: 34 |
| `reproduction/results/phase1/local-5060-stage-c-v1` | RTX 5060 Ti; C_confirmation | 360 | completed_unreviewed: 360 |
| `reproduction/results/phase1/rtx8000-intake-20260929-v2/attempts` | RTX 8000; supplemental B_screen | 25 | completed_unreviewed: 25 |

## 冻结清单与完成度对照

| 阶段/队列 | 冻结行数 | 产物快照/审计证据 | 差异或状态 |
|---|---:|---|---|
| A calibration（GPU） | 15 | 15 receipts；calibration_pass: 15 | GPU 标定批齐全；该目录的 file SHA 本轮未重复重算 |
| A calibration（CPU 对照） | 15 | 14 receipts；calibration_pass: 13, resource_blocked: 1 | 少 1 个有效 calibration_pass；resource_blocked 保留为资源阻断，不计成功 |
| B_screen（P4） | 300 | intake `completed_descriptive_only`；245 包通过既有摄入 | 55 行缺失；既有 P4–5060 审计见下 |
| B_screen（5060 bridge） | 34 | 34 fits 已验证 | 与 P4 有 34 个跨来源逻辑 ID 重叠（桥接/对照，不能相加为独立设置） |
| B_screen（RTX8000 supplement） | 55 | 25 本地已发布 receipts；25 个唯一 receipt hash 已摄入 | 30 行尚未本地摄入；远端状态取守护程序快照 |
| C_confirmation（5060） | 360 | 360 receipts；360 个唯一 receipt hash 被 consumer 接收 | consumer passes=115；issues=0 |
| D_temporal_robustness | 144 | 本索引来源中无 D attempt | 尚未看到 D 产物；不代表仓库外不存在 |

## 已存在的独立完整性证据

- P4 + RTX 5060 bridge：既有机器报告记载 P4 `245`、bridge `34` 验证通过，完整性问题 `0`；报告核对 artifact SHA、target/split、finite arrays 和重算 metrics。详见 [`P4-RTX5060-bridge-audit-20260929.md`](P4-RTX5060-bridge-audit-20260929.md)。
- Stage C 5060：已有 consumer 跨 `115` 个 pass 确认 `360` 个唯一 receipt，issues=0；每份被摄入的 package 由 intake 校验，训练/使用 GPU=False。
- RTX8000：已有 consumer 跨 `26` 个 pass 确认 `25` 个唯一 receipt，issues=0；摄入仍随远端新产物增长。
- 固定 checkpoint forward audit：状态 `completed_descriptive_only`，设置数 `34`；是推理复算而非新训练。详见 [`checkpoint-forward-audit-20260929.md`](checkpoint-forward-audit-20260929.md)。
- Stage A 两个 calibration 根目录合计 `29` 个 receipts；本轮核对 receipt 状态、SHA 声明和文件在场情况，但未重复重哈希约 1 GiB 文件。

## 实时队列快照（可能继续变化）

- 本机 supervisor 状态：`running`，观测时间 `2026-09-29T16:11:05.365556+00:00`。
- 最近收到/摄入计数：25/55 received，25/55 intake-acknowledged。
- 远端最后同步到本机的 runner 进度：25/55，当前运行 `B_screen_JPM_revin-TimesNet_h1_s2021`，状态 `running`，快照时间 `2026-09-29T15:51:55.811561+00:00`。此字段是最近一次同步快照，不保证等于阅读时的即时远端状态。

## 原始 TSLib 输出与重叠说明

仓库顶层 `results/` 目前有 562 个原始运行目录。它们是传统训练输出目录，不等同于本次的 immutable attempt receipt；本报告不移动、不删除，也不把目录数直接加到 receipt 总数。跨来源逻辑 ID 有 48 个重叠；来源组合计数：stage_a_cpu + stage_a_gpu: 14, stage_b_bridge_5060 + stage_b_p4: 34。P4 与 bridge 的重叠是预期配对，不是额外独立 setting。

## 差异清单与建议

1. **Stage B 仍未闭合**：P4 245/300；RTX8000 supplemental 55 行队列仍在运行，当前本地已摄入快照未满 55。保留两条硬件轨道，不跨轨道合并排名。
2. **CPU calibration 有 1 个资源阻断**：保留原 receipt 与 attempt，不把 blocked 改写成成功，也不在这个 inventory 里自动重跑。
3. **Stage D 没有本地 attempt**：manifest 仍有 144 planned 行；不得把缺少文件解释为失败或结果。
4. **旧 `results/` 562 个目录尚未逐一连接到本次 attempt**：它们单独保留；若后续需要归档/清理，应先建立明确的 run_id 对照并单独复核，不在本次进行删除。
5. **未发现文件缺失**：所有已索引 receipt 声明的 artifact 文件均在场。

逐 attempt 索引：[`reproduction/results/phase1/artifact-index-20260929T161118Z.csv`](../../reproduction/results/phase1/artifact-index-20260929T161118Z.csv)（693 行；包含 receipt SHA-256、硬件/阶段/模型、artifact 声明与文件在场状态）。
