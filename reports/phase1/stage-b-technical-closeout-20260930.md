# Stage B 技术收口记录（2026-09-30）

## 结论范围

300个冻结逻辑设置均有通过 intake 的最终产物，技术覆盖层面已齐全。原始 P4 执行仍是授权中的部分运行（245/300）；另55项通过独立的 RTX8000 非确认性补充轨道完成。没有单一硬件轨道执行完整300项，因此本记录不把两条轨道合成统一模型面板。

Stage B 预注册用于检查运行可行性与明显失败，不用于更换主要模型对或筛选资产。本收口只报告覆盖、intake 和各轨训练耗时；不提取或比较测试误差，不做 pooled ranking、硬件等效或模型优越性结论。

## Gate table

| Check | Input | Status | Evidence | Next permitted action |
|---|---|---|---|---|
| 冻结逻辑设置覆盖 | `B_screen.manifest.csv`、P4/RTX8000 intake manifests 和报告 | eligible | 冻结清单300个唯一ID；P4 245项 + RTX8000 55项；跨轨重叠0，并集等于冻结清单。 | 可将逻辑身份与最终产物覆盖标记为完成。 |
| P4 主执行批次 | `intake-p4-20260929-v1/report.json` | partial by authorization | 245/300 eligible、0 issues；P4 有240个非 TimesNet 设置和5个 TimesNet 设置，后者为 h=1 三项、h=24 两项。该批授权是预算受限的 P4 执行。 | 保留P4批次为partial；不把补充轨迹改写为P4完成。 |
| RTX8000 补充轨道 | `rtx8000-intake-review-20260930-complete/report.json`、本地 supervisor snapshot | eligible | 最新 intake 55/55 eligible、0 issues；supervisor 快照为 `queue_completed`、55/55，观测于 `2026-09-30T00:44:03Z`。这55项均为补充 TimesNet 任务：h=1 27项、h=24 28项。 | 按单独的非确认性补充轨道存档。 |
| 完成包和运行时日志 | 两份 intake 报告、300份最终 receipt 与 `run.log` | eligible | 300个 intake ID 与各自 selected manifest 精确一致；300份 receipt 均为 `completed_unreviewed`；300份 `run.log` 的当前 SHA-256 与 receipt/intake 声明一致；每份日志恰有一个 `train_time` 字段可解析；两份 intake issues 均为0。 | 可按硬件轨道分别报告训练耗时分布。 |
| 同一硬件的完整 Stage B 面板 | P4 授权及两份 intake 的 execution-track provenance | requires-resolution | P4 仅完成245项；缺项由 RTX8000 执行。没有单一轨道覆盖全部300项。 | 不产生统一完整面板的跨模型结果。若以后需要该类结果，需新研究/执行授权，并预先确定同栈面板或获批的数值可比性设计。 |

## 各轨训练耗时

耗时取训练日志 `train_time` 字段，单位秒。每张表只汇总本轨数据；不同轨道的行不构成跨硬件比较。

### P4 主轨（245项）

| Model | Horizon | Eligible fits | Median seconds | P90 seconds | Min seconds | Max seconds |
|---|---:|---:|---:|---:|---:|---:|
| revin-DLinear | 1 | 30 | 10.285 | 11.253 | 9.14 | 13.61 |
| revin-DLinear | 24 | 30 | 8.480 | 10.037 | 3.61 | 14.59 |
| DeReFusion | 1 | 30 | 21.750 | 23.752 | 18.95 | 30.03 |
| DeReFusion | 24 | 30 | 18.000 | 24.232 | 8.77 | 25.18 |
| revin-PatchTST | 1 | 30 | 19.940 | 31.012 | 12.87 | 36.50 |
| revin-PatchTST | 24 | 30 | 16.835 | 29.151 | 10.18 | 33.73 |
| revin-iTransformer | 1 | 30 | 15.740 | 26.439 | 7.97 | 29.58 |
| revin-iTransformer | 24 | 30 | 16.900 | 25.596 | 7.67 | 33.20 |
| revin-TimesNet | 1 | 3 | 2,869.080 | 7,034.104 | 2,688.24 | 8,075.36 |
| revin-TimesNet | 24 | 2 | 2,554.325 | 2,762.017 | 2,294.71 | 2,813.94 |

### RTX8000 补充轨道（55项）

| Model | Horizon | Eligible fits | Median seconds | P90 seconds | Min seconds | Max seconds |
|---|---:|---:|---:|---:|---:|---:|
| revin-TimesNet | 1 | 27 | 1,363.970 | 1,839.508 | 539.21 | 2,264.96 |
| revin-TimesNet | 24 | 28 | 666.640 | 1,160.844 | 491.76 | 1,695.85 |

## 复现与许可边界

- 复现脚本：[phase1_stage_b_technical_closeout.py](../../reproduction/analysis/phase1_stage_b_technical_closeout.py)
- 汇总数据：[按轨运行耗时 CSV](stage-b-runtime-by-track-20260930.csv)
- 逐设置 ledger：[300项训练耗时和日志哈希 CSV](stage-b-runtime-setting-ledger-20260930.csv)
- 复现命令：`python reproduction/analysis/phase1_stage_b_technical_closeout.py`
- 脚本验证 manifest identity、intake eligibility、receipt 与日志哈希，再提取 `train_time`；不解析、汇总或比较模型测试误差。
- RTX8000补充运行可用于补全逻辑覆盖及本轨技术记录，但授权禁止把它们当成P4分数面板的延续。跨硬件等价阈值仍是 `TBD`。
