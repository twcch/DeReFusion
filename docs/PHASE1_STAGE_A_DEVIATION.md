# Phase 1 Stage A 执行偏差记录

状态：**事实部分已冻结；§10 是处置决定的辅助记录，正式修订见 `PHASE1_AMENDMENT_V1.2.md`**
协议：`phase1-v1.1-2026-09-19`
上位文档：`docs/PHASE1_PREREGISTRATION.md`（规范）、`docs/PHASE1_OUTSOURCING_EXECUTION_PLAN.md`（执行控制）
记录日期：2026-09-28
记录时点：**Stage A 全部执行完毕之后**

## 1. 记录性质

本文是**事后偏差记录（post-hoc deviation record）**，不是事前协议修订。

区别是实质性的，不是措辞问题：预注册 §9（v1.1）的效力来自它写在任何产出生成**之前**；本文写于 Stage A 执行**之后**，因此**不具备** amendment 的地位。不得把本文表述为「预先声明」，也不得据此主张 Stage A 的执行路径获得了事前授权。

之所以仍可记录为无科学后果的偏差，唯一依据是预注册 §3 对 Stage A 的定性：

> This stage checks imports, tensor shapes, output retention, timing, and manifest generation. **Its losses are not compared or published as evidence.**

Stage A 不产生任何被比较或发表的损益。若同类设备变更发生在 Stage B/C/D，则属协议违背：应立即停止该阶段、保留并记录失败 attempt，在任何重跑前签发事前 amendment；随后只能按修订后的协议重新运行，不能事后补记来追认（见 §6）。

## 2. 授权基线

执行计划 `PHASE1_OUTSOURCING_EXECUTION_PLAN.md` 授权的 GPU 迁移范围是**两条** fit：

- §1：「A05 attempt 2、A15 attempt 1 与 Stage B 完整 300-fit 面板转移到同一型号 GPU、同一冻结环境。」
- §8 步骤 4：「仅运行 A15 attempt 1 和 A05 attempt 2，摄入并关闭 Stage A。」

即：13 个已通过的本机 CPU 包**原位保留为 Stage A 的正式产物**，GPU 只补 A05、A15 两个缺口。

同一文档 §3.1 另有一项约束：同 Stage 内的设备一致性要求，且「最低可执行单元的缩减必须先形成协议 amendment，**不能事后决定**」。

## 3. 实际执行

Stage A 在 P4（Tesla P4 7680 MiB）上以**完整 15-fit 网格**端到端重跑，而非仅补两个缺口。

| 结果根 | 设备 | 包数 | 状态分布 |
|---|---|---|---|
| `reproduction/results/phase1/calibration/` | `cuda` | 15 | 15 × `calibration_pass` |
| `reproduction/results/phase1/calibration-cpu/` | `cpu` | 14 | 13 × `calibration_pass` + 1 × `resource_blocked` |

15 个 GPU 包全部为 `__attempt-01`，`status` 唯一值为 `calibration_pass`，`supersedes_attempt`、`retry_reason`、`retry_authorization` 均为 `null`，`git_commit` 统一为 `0d05f931`，`created_at_utc` 跨 `2026-09-28T08:10:43Z`（A01 冒烟）至 `2026-09-28T10:18:10Z`（A15，与守护写 `STAGEA_DONE` 的 18:18:15+08:00 吻合）。

## 4. 偏差清单

**D1 — 执行范围扩大 13 个 fit。** 授权 2 个 GPU fit，实际 15 个。被额外重跑的 13 个中，有 **12 个此前的 CPU 状态是 `calibration_pass`**。

**D2 — 绕过了 attempt/授权机制。** 预注册 §4.2 规定「成功状态禁止为了获得更好指标而重跑。只有 `resource_blocked`、`failed`、`timed_out`、`artifact_invalid`、`infrastructure_interruption` 等技术失败可重试」，§4.3 要求每次重试记录新的 `retry_authorization`。实际做法是把 13 个 CPU 包移出 runner 的 `PACKAGE_ROOT`，使其对 `choose_attempt()` 不可见，从而以全新 `attempt-01` 执行，未产生任何 `retry_authorization`。

**D3 — 冻结文档与仓库历史互相矛盾。** 执行计划 §1/§8 仍描述「仅 A05/A15」；而 `0d05f93` 的提交信息描述的是全网格重跑。两者至今并存于仓库中。

**根因（D1/D2）**：`run_phase1_calibration.py:33` 的 `PACKAGE_ROOT` 是一个单一路径 `reproduction/results/phase1/calibration`；`choose_attempt()`（:88）与 `resource_blocked_models()`（:312）均只在该根下 glob。把 CPU 包 `git mv` 到 `calibration-cpu/` 后，runner 在 P4 上看到的是一个**空的 Stage A 根**，于是把 15 个 fit 全部选为待执行的 attempt-1。该行为是**刻意的**：`0d05f93` 的提交信息明确写明要在 P4 上端到端重跑完整 15-fit 网格，使其处于单一、同构的环境，并说明将 CPU 包移出 runner 可见根的理由；但这仍未获得协议中的事前授权，也未记入执行计划。

**动机澄清**：重跑的动机是让 Stage A 的执行环境保持一致，不是获取更好指标——Stage A 的损失本就不被比较（§3）。因此 D2 违反的是 §4.2「成功状态禁止重跑」的字面规定；本记录不把该规定改写成仅禁止指标择优，也不据动机将未授权重跑追认为合规。

## 5. 未偏离项

逐条核对，以下均未偏离：

- 模型网格、资产、horizon、seed、超参数、切分模式：与冻结配置一致（GPU 包 `config_fingerprint` 见 §8）。
- 结果数组的有限性与形状检查：`all_arrays_finite` 为真，形状与冻结契约一致。
- **未删除、未覆盖任何 attempt。** 13 个 CPU 包以 `git mv` 迁移并保留原始字节，迁移前后哈希均 13/13 通过；A05 的 CPU 包逐字节保留。
- 数据集字节：注册表 61/61 PASS，两套结果的 `dataset_sha256` 一致。
- 校准命令使用了 Phase-1 专属 `--result_log`，未写入遗留全局账本（§6）。
- 设备类别已进入产物身份：GPU 包 `environment.device = "cuda"`、`torch = "2.5.1+cu121"`、`cudnn = 90100`；CPU 包 `environment.device = "cpu"`、`torch = "2.5.1+cpu"`。执行计划 §5「输出身份必须包含设备类别」要求满足。

## 6. 科学后果评估

**结论：本次偏差不影响任何 Stage A 的科学主张，因为 Stage A 不承载科学主张。** 这是对科学后果的限定，不是对程序合规性的豁免。依据仅为预注册 §3：Stage A 的损失不被比较，也不作为证据发表。

但有一项**须显式声明的连带影响**：预注册 §5 把 `parameter count, training wall time, inference milliseconds per sample and peak memory` 列为次级产出。这些量随设备变化显著（见 §8），因此**引用 Stage A 标定数值时必须同时声明其来源设备与软件环境**，不得混用两套数字。

**假如同一变更发生在 Stage B/C/D**：那将是协议违背。正确处置是停止该阶段、记录失败 attempt、先签发事前 amendment，再依修订后的协议重跑，而不是事后补记。本次仅能依据 §3 记录为没有 Stage A 科学后果；这不构成对程序偏差的追认。

## 7. A05 判定校正（不修改任何 receipt）

CPU A05 的 `status = "resource_blocked"`，其 `status_reason` 为：

> `CPU calibration exceeded 205 seconds without completing epoch 1; model family paused before A10/A15`

该判定不成立，有两条独立证据：

1. **receipt 自证矛盾。** 同一份 receipt 的 `status` 为 `resource_blocked`，理由称 205 秒未完成 epoch 1；但 `termination` 字段为 `"manual controlled interrupt during backward pass"`，且 `observed.completed_epochs = 0`。status/理由声称资源阈值阻塞，termination 却记载人工中断；这两种原因不能同时作为该次终止的真实解释。
2. **实测证伪。** 同机同模型的 A10 在 CPU 上单 epoch 实测约 880 秒（总计 7285.16 s 完成早停）。205 秒仅相当于 epoch 1 的约 23%，属正常进度。且 A05 的 BOND10Y 训练段（1285 行）比 A10 的 BONDETF（1567 行）更小，只会更快。A05 在 GPU 上于 2230.97 s 内正常完成至早停。

**结论：A05 的真实性质是「人工可控中断」，`resource_blocked` 是误判。** 但 A05 的 CPU receipt 是 immutable 证据，**本次未作任何修改**。校正以本文记录。

**未决影响**：只要 CPU A05 的 receipt 保持 `resource_blocked`，`resource_blocked_models()`（`run_phase1_calibration.py:310`）在仅扫描 `PACKAGE_ROOT` 时就会持续把 `revin-TimesNet` 判为受阻模型并自动跳过。该状态的定性（记录性 vs 门控性）见 §10。

## 8. 连带证据：Stage A 标定实测

> **可比性限制**：下表两侧并非受控对照。CPU 列来自本机（ThinkBook 14+），GPU 列来自 P4；两者是不同机器、不同 torch 构建（`2.5.1+cpu` vs `2.5.1+cu121`），且 CPU 包产生于 5 个不同 commit（`f129cf3a` / `0f7c2c6` / `449290038e21` / `18b738a2` / `459c7742`；A05 的 receipt 记录的是其中 `449290038e21` 的短哈希 `4492900`），GPU 列统一为 `0d05f931`。CPU 侧另有本机并发负载。故倍数关系仅供量级参考，**不构成加速比的科学测量**。

| fit | 模型 | CPU 训练秒 | GPU 训练秒 | GPU/CPU |
|---|---|---:|---:|---:|
| A01 | revin-DLinear | 5.32 | 8.16 | **0.65** |
| A02 | DeReFusion | 43.34 | 14.05 | 3.09 |
| A03 | revin-PatchTST | 35.43 | 16.49 | 2.15 |
| A04 | revin-iTransformer | 33.42 | 20.90 | 1.60 |
| A05 | revin-TimesNet | 未完成 | 2230.97 | — |
| A06 | revin-DLinear | 5.84 | 10.57 | **0.55** |
| A07 | DeReFusion | 76.63 | 19.64 | 3.90 |
| A08 | revin-PatchTST | 30.74 | 12.86 | 2.39 |
| A09 | revin-iTransformer | 31.59 | 21.40 | 1.48 |
| A10 | revin-TimesNet | 7285.16 | 2519.21 | 2.89 |
| A11 | revin-DLinear | 2.69 | 4.28 | **0.63** |
| A12 | DeReFusion | 71.09 | 18.34 | 3.88 |
| A13 | revin-PatchTST | 43.43 | 15.14 | 2.87 |
| A14 | revin-iTransformer | 14.90 | 12.29 | 1.21 |
| A15 | revin-TimesNet | 未运行 | 2468.69 | — |

三条对 Stage B（300 fits）规划有直接后果的读数：

1. **最小的模型在 GPU 上更慢，三个资产全部如此**（DLinear：0.55–0.65×）。小模型被 kernel 启动开销支配，`deterministic: true` 又关闭了 cuDNN autotune（`run.py:181-187`），正是通常补偿该开销的机制。Stage B 把 DLinear 放到 GPU 上可能净亏。
2. **TimesNet 峰值显存 4674.9 MB**，三个资产完全一致；相对 7680 MiB 卡容量为 61%。这确认了同一张卡上无法安全并发两个 TimesNet fit，Stage B 的 TimesNet 必须串行。
3. **TimesNet 推理 71.79 ms/样本**（GPU）vs 212.84 ms/样本（CPU），2.96×。

## 9. 遗留完整性问题：attempt 谱系跨根不可见

§4.5 要求「每个 logical run 必须恰有一个经验证的有效成功 attempt：零个为未完成，多个为冲突并 fail-stop」。

当前 `A05_BOND10Y_revin-TimesNet_h24_s2021` 这一个 logical run 在仓库中**存在两个成功或半成功的 attempt**，且分处两个根：

- `calibration-cpu/A05_BOND10Y_revin-TimesNet_h24_s2021/` — 旧格式，无 `attempt` 字段，按 §4.1 兼容解释为 attempt 1，状态 `resource_blocked`。
- `calibration/A05_BOND10Y_revin-TimesNet_h24_s2021__attempt-01/` — 状态 `calibration_pass`，`supersedes_attempt = null`。

问题有两层：

1. **supersession 未被表达。** GPU 包的 `supersedes_attempt` 为 `null`，即产物层面**没有记录**它接替了 CPU 的 attempt-1。按 §4.5 字面，这是一个「多个成功」冲突，应 fail-stop。
2. **runner 无法核验。** `choose_attempt()` 只 glob `PACKAGE_ROOT`，对 `calibration-cpu/` 完全不可见，因此消费端无法独立发现该冲突。同理 `resource_blocked_models()` 也看不到另一根的状态。

A10、A15 同样存在 CPU/GPU 双包（A15 的 CPU 侧无包，故只涉及 A10 与 A05）。**该问题在 Stage B 之前必须关闭**，否则摄入工具（执行计划 §4.5）建立在一个不完整的可见域上。

## 10. 处置决定：辅助记录

> **本节性质：辅助内容。** 本节记录两个处置问题的推导与状态，供正式修订引用。
> **它本身不构成裁定，也不赋予任何事项 amendment 地位。** 依据是本文 §1：本文写于
> Stage A 执行之后，不具备事前效力；把结论写进事后记录不会让它获得事前效力。
> 处置的正式效力来自 `PHASE1_AMENDMENT_V1.2.md` 及其授权记录
> `PHASE1_EXECUTION_DECISION_V1.2.json`（由 Codex 会话形成，2026-09-28）。
> **本节不得被引为授权的来源。**

### 10.1 两个处置问题及其推导

**决策 1 — 哪一列是 Stage A 的正式产物。**
倾向方案：以 GPU 列（15/15，设备同构、单一 commit）为准，CPU 列整体转为历史证据并在摄入工具中显式排除。理由：CPU 列残缺（A05 受阻），GPU 列完整；且整列 GPU 恰好满足执行计划 §3.1 的设备一致性要求，无需再为「同列内 CPU/GPU 混用」作任何声明。一旦 Stage A 定为 GPU，则其标定数值全部来自 P4 环境，与 Stage B 若在同类 GPU 上执行的假设一致。

**决策 2 — 关闭 §9 的谱系冲突。**
候选做法：(a) 保留 CPU receipt 原字节，将本文对 A05 的判定作为外部审计说明；在统一可见域中纳入两根并明确记载 GPU 包与 CPU 失败 attempt 的谱系关系；(b) 明确将 CPU 列限定为历史证据并排除在 Stage A 的正式摄入与 §4.5 唯一性判定范围之外。**任一方案均须定义统一的 attempt 可见域及 `resource_blocked_models()` 的门控边界。** 本记录提出选项，不自行改写 receipt 中的状态或谱系字段。

### 10.2 已采纳的处置（陈述，非授权）

采纳的是候选处置 **(b)**：P4 `calibration/` 中 15 个 `calibration_pass` 包为 Stage A 权威执行标定包；CPU `calibration-cpu/` 中的 14 个包完整保留为历史证据，并排除出当前 Stage A attempt 唯一性判定范围。所有 receipt 与原始产物保持不变。

该处置确认的是**包的证据范围**：不追认最初 13 个额外 P4 fit 的事前程序授权，也不授权 Stage B--D。**其效力来源是 v1.2 amendment，不是本节。**

### 10.3 持续约束

不改动任何 receipt，不改动 runner，不推进 Stage B。v1.2 §2 明确未修改任何原始数据、receipt、checkpoint、预测、目标或指标文件；v1.2 §3 明确未授权 B--D。Stage B 任何训练启动之前，仍须关闭 v1.2 与执行计划中列出的剩余门禁，并签发单独的、绑定 commit/环境/设备/阶段的执行授权。

## 11. 需修订的冻结条款

| 文档 | 位置 | 现状 | 需修订为 |
|---|---|---|---|
| 执行计划 | §1、§8 步骤 4 | 旧文案仅授权 A05/A15 GPU 补跑 | 已更新为 P4 15/15 权威标定包，并引用 v1.2 amendment |
| 预注册 | §3 Stage A / §9 | v1.1 冻结原文 | 原文保持不变；本英文 v1.2 amendment 明确设备来源与包的纳入范围 |
| Stage A runner | `PACKAGE_ROOT` | 单根扫描 | 在 v1.2 下，该单根是正式 P4 包域；CPU 根明确定义为历史证据、不纳入唯一性计数 |

上述 v1.2 provenance amendment 已另行形成英文规范文件并经用户批准；它只裁定 Stage A 证据范围，不代表 B--D 的执行授权。

Stage B 任何训练启动之前仍须关闭执行计划中剩余工程/环境门禁，并签发单独的 Stage B 授权。

## 12. 本次连带关闭的既有缺陷

对照执行计划 §7 的九项：

- **#9（Windows 换行转换破坏冻结哈希）— 已关闭。** `.gitattributes` 为 `results/phase1/**/*.log` 与 `logs/phase1/**/*.log` 声明 `-text`；`git add --renormalize` 使 20 个既有 blob 与规则一致。验证：Windows 侧 `blob == receipt == worktree` 61/61；Linux 全新 checkout 侧 13/13 包哈希通过、`check-attr` 回报 `text: unset`。该缺陷若不修，**所有** receipt 在跨平台传输后都会校验失败。
- **#2（缺少不可覆盖的 attempt 机制）— 机制已生效。** Stage A 全程 15 个包均为 `attempt-01`，supervisor 的重试循环一轮未触发；A05 的 CPU 包在迁移前后逐字节保留。但如 §9 所述，跨根可见性仍不完整。
- **#8（setting 未覆盖设备，存在静默覆盖）— 已关闭。** `config_fingerprint` 纳入设备类别，CPU 与 GPU 的同一 logical run 落到互不冲突的 `results/<setting>/` 目录，`find_setting()` 的唯一性断言成立。

## 13. 证据位置

- GPU 结果：`reproduction/results/phase1/calibration/`（15 包，871 MB）
- CPU 结果：`reproduction/results/phase1/calibration-cpu/`（14 包，297 MB）
- 冻结提交：`0d05f93198d07dcea64dbcc08e7a783a09152e6f`
- P4 运行日志、守护脚本、环境指纹、安装日志：`C:\Users\26843\Documents\ChatGPT\p4_phase1_artifacts\`
- 两端哈希复算：`calibration` 15/15 `failures=0`；`calibration-cpu` 14/14 `failures=0`（本机复算，2026-09-28）
