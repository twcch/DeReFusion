# Stage B 覆盖审计（2026-09-30）

## 结论

身份与已审计产物覆盖齐全：冻结的 Stage B 清单包含 300 个唯一逻辑设置；P4 intake 的 245 个设置与 RTX8000 补充批次的 55 个设置互不重叠，二者并集恰好等于这 300 个设置。P4 和 RTX8000 的 intake 均为 eligible，未报告 issue。

这只证明设置身份及产物覆盖齐全。它不证明 P4 与 RTX8000 数值等效，也不授权将指标合并作统一排名、模型优劣或确认性 Stage B 结论。Stage B 仍按两个执行轨道分别保留。

## Gate table

| Check | Input | Status | Evidence | Next permitted action |
|---|---|---|---|---|
| 冻结 Stage B 身份集合 | `B_screen.manifest.csv`，协议 `phase1-v1.1-2026-09-19` | eligible | 300 行、300 个唯一逻辑 ID；文件 SHA-256 为 `4bc275cf58922d18cab6840d31adb59368aea9304964b54d3f2a3e156ba63ea5`。 | 将该清单作为本次覆盖核验的固定全集。 |
| P4 主轨产物 intake | `intake-p4-20260929-v1/report.json` 及 `selected-manifest.csv` | eligible | 245/300 eligible；245 个报告 ID 唯一且与 intake 清单一致；0 issue。其 intake 清单 ID 均属于冻结的 Stage B 清单。 | 作为 P4 轨道的已审计覆盖记录；保留硬件和来源标签。 |
| RTX8000 补充清单身份 | `supplemental-manifests/rtx8000-timesnet-remaining-v1.csv` | eligible | 55 行、55 个唯一逻辑 ID；清单 SHA-256 为 `927f7d486b32e2069f97cc985c32784c0ca9a3b94b8a9e06df15b024cab3f24c`。这些 ID 正好是冻结总清单减去 P4 intake 的 55 项。 | 按补充轨道分别报告。 |
| RTX8000 全量产物 intake | `rtx8000-intake-review-20260930-complete/report.json`（最新全量55份报告，完成时间 `2026-09-30T01:11:09Z`）及同目录 `selected-manifest.csv` | eligible | 55/55 eligible；55 个报告 ID 唯一且与 RTX8000 补充清单一致；缺少设置 0、issue 0、`input_immutability_verified=true`。更早的 `rtx8000-intake-review-20260930/report.json` 仅覆盖25/55，不作为最新覆盖状态。 | 接受本次完整 intake 快照；若有后续新 attempt，另做不可变快照和 intake，不覆盖本报告证据。 |
| 两轨身份并集 | 冻结清单、P4 intake、RTX8000 全量 intake 的逻辑 ID | eligible | P4 245 + RTX8000 55 = 300；跨轨重叠 0；并集与冻结清单完全相等。独立复算逻辑 ID：`B_screen_<asset>_<model>_h<horizon>_s<seed>`。 | 可陈述“身份与产物覆盖齐全”。 |
| 跨硬件数值等价与合并排名 | 两份 intake 报告、Stage B 执行决策、GPU bridge 校准方案 | requires-resolution | 两份 intake 均 `pooling_authorized=false`、`ranking_authorized=false`、`numerical_equivalence_claim=false`；RTX8000 与 P4 为不同执行轨道。等价容忍阈值 `TBD`，没有预注册判定规则。 | 不合并指标、不 pooled ranking、不声称硬件等效或模型优越。任何跨轨推断须先有事前批准的精度/等价规则和相应研究设计；阈值在批准前为 `TBD`。 |
| Stage B 科学筛选结论 | `PHASE1_PREREGISTRATION.md`、`PHASE1_STAGE_B_EXECUTION_DECISION_2026-09-29.md`、以上 intake | requires-resolution | Stage B 的多重性方案针对完整300项面板；P4 执行决策将 RTX8000 的55项列为技术补充轨道，不构成 P4 轨道的追溯延伸。覆盖完成本身不解除跨轨限制。 | 不从混合轨道产生全 Stage B 模型排名或优劣结论。可继续做各轨独立的完整性和描述性汇总；要作统一科学筛选，须有获批的共同执行栈完整面板，或另行批准并事前定义可比性设计和规则。 |
| Stage C 独立分析准入 | `C_confirmation.manifest.csv`、`authorization-local-5060-stage-c-v1.json`、`PHASE1_STAGE_C_EXECUTION_DECISION_2026-09-29.md`、Stage C 产物索引快照、预注册历史 | requires-resolution | Stage C 的360行冻结清单 SHA-256 `bda784f0dd816939cdfd02e30109369df6a3e2e76aecd79aa8ee87b3910e6bfa` 与 RTX 5060 Ti 本地执行授权绑定一致；既有产物索引记录360份 receipt、360个唯一 receipt hash 被 intake 接收、115个 pass、0 issues。预注册初始提交 `6bbb045ef8da34fb6be1b05222347167859b9104`（2026-09-19）只写“immutable bootstrap seed”和等资产权重的 asset-cluster bootstrap，没有给出 seed 或重采样次数；执行决策也未补值。已按描述性边界完成分层点估计，全部数值文件 SHA 与 receipt 声明复核通过，见 `stage-c-descriptive-review-20260930.md` 及其可复现脚本。 | 描述性点估计可按明确边界报告。先核实仓库外是否存在可验证的事前 bootstrap 记录；若没有，不用事后参数形成确认性区间，需透明修订方案并说明确认性地位，独立确认要求按治理规则处理。Stage D 仍单独设门。 |

## 边界与许可动作

- 本审计只读取冻结清单、intake 报告和授权/方案记录；没有修改实验产物，也没有计算或比较 Stage B 的分数。
- 当前可以报告 Stage B 的逻辑身份与 intake 产物覆盖完整；不能把这句话延伸为跨硬件可合并的统一科学结论。
- 冻结方案允许的运行可行性/明显失败检查与分轨训练耗时已另行收口，见 [Stage B technical closeout](stage-b-technical-closeout-20260930.md)。该报告不提取或合并模型测试误差。
- RTX8000 最新全量报告只替代先前25/55的覆盖快照；它没有改变原补充轨道的用途、授权或硬件身份。
- Stage C 是另一项本地确认性研究。执行许可和360份产物覆盖均有记录；分层描述性点估计见独立复核报告。预注册没有固定 bootstrap seed 与重采样次数，因此尚无确认性区间或模型效应结论。
- 研究问题与 Stage C 配对 MSE estimand 已冻结。跨硬件等价精度阈值为 `TBD`；Stage C bootstrap seed 与重采样数在仓库预注册历史中没有固定，均为 `TBD`。如需新研究设计，应分别说明是否需要研究问题、精度/功效或资源分析；本次身份覆盖审计不推导这些数值。
