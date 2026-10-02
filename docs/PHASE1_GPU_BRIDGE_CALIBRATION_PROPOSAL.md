# Phase 1 双硬件执行栈桥接校准（提案）

**状态：用户已授权分批执行；属于非确认性补充批次，不修改 `phase1-v1.1`、Stage B manifest 或既有结果。**  
**日期：2026-09-29**  
**目的：** 小规模评估既有 Tesla P4 结果与本地 RTX 5060 Ti 执行结果之间的可观察差异，为后续执行决策提供证据。该校准不是模型优劣检验，也不自动授权把两个硬件上的 Stage B 结果合并。

## 1. 先界定它能回答什么

当前两端不是单纯的 GPU 型号不同：P4 的冻结环境为 PyTorch `2.5.1+cu121` / CUDA runtime `12.1`；本地 5060 Ti 环境为 PyTorch `2.7.1+cu128` / CUDA runtime `12.8`。因此本提案估计的是**执行栈差异**（GPU、驱动、CUDA/cuDNN、PyTorch 与其余环境的组合），不能把差异单独归因于 GPU 硬件。

校准可以发现明显的执行栈漂移、模型×执行栈交互或结果不稳定；小样本下“未发现明显差异”不等于证明等价，也不能通过事后缩放、四舍五入或归一化把差异消掉。

## 2. 最小桥接面板

使用 P4 已完成且完整性验证通过的 Stage B attempts 作参考，只在 5060 Ti 上重跑对应配置。资产在查看校准输出前固定如下：

| 资产 | 预注册 cohort | 用途 |
|---|---|---|
| AAPL | c1-20 | 股票代表 |
| GSPC | original-10 | 指数代表 |
| EURUSD | original-10 | 外汇代表 |
| BTCUSD | original-10 | 加密资产代表 |

对四个资产均使用 `revin-DLinear`、`DeReFusion`、`revin-PatchTST`、`revin-iTransformer`，预测期 `1, 24`，seed `2021`：`4 × 4 × 2 = 32` 个配对设置。另在 AAPL 上加入 P4 已有完整参考包的 `revin-TimesNet` 两个预测期设置，共 **34 个配对设置**；每个配对设置有一份 P4 参考产物和一份 5060 Ti 新产物，故本地新增训练为 **34 fits**。这约为 Stage B 300 fits 的 11%，比在 5060 Ti 重跑整个面板小；代价是 TimesNet 的跨资产交互覆盖有限。

若任一指定 P4 参考包、receipt、冻结数据哈希或预测键缺失/校验失败，该逻辑设置标记为 `requires-resolution`，不得临时换资产或改预测期补齐。固定选择依据是 cohort 覆盖与 P4 参考包可用性，不使用任何预测分数。

## 3. 必须冻结的执行细节

1. P4 receipts 绑定源码提交 `bf738b9c42104197ee6204f9bcbd223fe0b3c3e2`。5060 Ti 使用本地执行器安全修订后的源码提交；已核验 `models/`, `exp/`, `run.py`, `data_provider/`, `layers/`, `utils/` 与 P4 提交内容一致，差异仅限 Phase 1 执行器及其测试。完整 commit 记录在本地授权与 receipt 中。
2. 对 P4 参考包先验证 receipt 中 `pred.npy`、`true.npy`、`metrics.npy`、`checkpoint.pth`、`run.log` 的 SHA-256，以及 prediction-key、manifest 行、数据文件 SHA-256。P4 原件归档在 `reproduction/results/phase1/p4-partial-20260929/`，不得放进新实验的活动输出根目录。
3. 两端保持相同的逻辑 run 配置：数据、split、seed、模型参数、训练上限、早停、指标和 test keys 均按冻结 Stage B 行执行；不调参、不重抽样、不删异常值。记录完整命令、源码 commit、GPU/driver、CUDA/cuDNN、Python、所有包版本、环境指纹和每文件哈希。
4. 5060 Ti 结果分别写入 `reproduction/results/phase1/local-5060-bridge-v1/` 和 `reproduction/results/phase1/local-5060-timesnet-remaining-v1/`；不能写回 P4 attempt、原 Stage B 活动目录或彼此的目录。每个新 attempt 不可覆盖；同批恢复必须匹配固定的批次元数据。
5. 以 `(asset, model, horizon, seed, prediction key)` 配对；首先检查数据/预测键一致、数组形状一致、有限值和 receipt 哈希。结构性校验失败是硬失败，不进入数值比较。

## 4. 预先指定的比较与决策

- 主校准量：每个逻辑设置的 normalized test MSE 两端绝对差与相对差；同时报告 MAE/RMSE、逐样本预测误差分布、参数量和运行诊断。Stage B 的既有多重性方案不在这份小校准上作显著性筛选。
- 汇总必须按模型、预测期和 cohort 分层；提供全部 34 个配对点与范围，不只报均值。对重叠时间窗口不把每个预测点误当独立资产样本。
- **数值接受界限目前为 TBD。** 在开启 5060 Ti 输出前，须由方案批准人给出与研究上最小有意义差异相联系的容忍界限和处理规则；不能看完结果再挑阈值。样本过小导致区间宽时，结论记为 `inconclusive`，不能把“未显著”写作“等价”。
- 若未通过或不确定：不得合并跨栈 Stage B 结果；要么在同一执行栈完成一个新的完整面板，要么将其明确列为不同执行批次且不作模型排名结论。
- 即使桥接表现稳定，本 34 对设置也只支持“所抽取配置上的执行栈敏感性诊断”。它**不自动验证**未抽样的资产/模型组合，更不自动解除“Stage B 使用统一硬件与环境”的已批准边界。任何合并或调整 Stage B 统计解释都需另行、事前、版本化批准。

## 5. 阶段门与允许动作

| 检查 | 输入 | 状态 | 证据 | 下一步允许动作 |
|---|---|---|---|---|
| P4 阶段是否完整 | 既有 Stage B manifest、授权和日志 | partial | P4 决策仅授权 Tesla P4 上的约 8 小时 B_screen；记录为 245 个已打包 receipt，剩余项未完成 | 把既有批次标为 partial；不可声称 Stage B 完成或作筛选结论 |
| P4 原件是否已本地留存 | P4 attempts、授权、环境指纹、budget log | copied; hash check passed | `reproduction/results/phase1/p4-partial-20260929/`；245 个 receipts，receipt 所列产物 SHA-256 全部匹配 | 可继续做只读审计与桥接面板覆盖核对 |
| 5060 Ti 是否属于原授权 | 执行决定及机器授权 | no; separate auth | 原授权绑定 `bf738b9…`、CUDA、Tesla P4 和 P4 环境指纹 | 仅使用新的、hash-bound 本机非确认性授权 |
| 两批 manifest 与 P4 来源是否核验 | P4 archive receipts、manifest 和数据键 | pass | 245 份 P4 receipts、产物 SHA mismatch=0；桥接参考覆盖 34/34；remaining TimesNet=55 | 按锁定清单运行，缺项不替换 |
| 5060 Ti dry-run | RTX 5060 Ti 环境及两份独立 manifest | pass | bridge selected=34；TimesNet selected=55；CUDA available；无训练启动 | 正式授权生成后依次运行 |
| 数值等价与池化判据 | 小样本桥接设计 | not claimed | 用户批准的是描述性诊断；不设结果后挑选阈值，不声称等价 | 两批永不合并，不产生跨栈 Stage B 排名 |

## 6. 批次授权和执行顺序

用户于 2026-09-29 确认“本地校准 + 5060 Ti 继续未跑部分”。范围落实为：

1. `local-5060-bridge-v1`：34 个配对设置，只做描述性执行栈敏感性诊断。
2. `local-5060-timesnet-remaining-v1`：55 个 P4 尚未运行的 TimesNet 设置，独立技术补充。

先跑桥接，再跑 55 个 TimesNet；不并行争用同一张 GPU。两批使用独立 manifest、授权、输出目录和 batch metadata。授权均明确禁止跨栈 pooling、排名及等价声明。结果统计、审查和最终说明可以在 GPU 任务结束后完成。

## 7. 必须保留的限制

本桥接实验不取代 Stage B，也不把 P4 部分结果“校正成”5060 Ti 结果。若目标是得到可直接用于 Stage B 模型间比较的完整统一面板，最干净的路径仍是：在一个冻结执行栈上完成全部 300 个设置，并将 P4 部分面板作为历史/执行证据保留。若只完成这份小桥接，Stage B 仍不得宣称完整。

## 8. 执行后审查

每份 receipt 都要复核状态、数据/预测键、数组形状、有限值及 SHA-256。桥接报告给出 34 个配对点和分层差异，只能称为描述性敏感性诊断；55 个 fit 报告覆盖及未完成项。不得声称 Stage B 完成：245 P4 + 55 local 的混合栈组合不能替代同一执行栈的完整 300-fit 面板。

## 9. RTX 8000 执行调整（2026-09-29）

用户明确要求将尚未完成的 TimesNet 计算转交线上 RTX 8000，并尽快自动化。原 5060 Ti bridge 已完成 34/34 fits（约 29 分钟）；不再重跑，也不把它当作 RTX 8000 的校准证据。`rtx8000-timesnet-remaining-v1` 是新的、独立的非确认性执行批次，覆盖原计划中 55 个尚未运行的 `revin-TimesNet` B_screen 行；数据、切分、键、模型配置及指标保持原冻结定义。

此调整不改变 `phase1-v1.1`、正式 B_screen manifest 或既有 P4 receipts。RTX 8000 批次具有自己的 manifest、授权、环境指纹、attempt、进度与输出目录；授权仅限 Quadro RTX 8000 上 CUDA 执行，绑定完整代码 commit、manifest 哈希、GPU UUID 和以下执行栈：Python `3.11.15`、PyTorch `2.5.1+cu121` / CUDA runtime `12.1`、NumPy `2.1.2`、pandas `2.3.3`、scikit-learn `1.7.2`。如果这些精确版本无法建立，fail closed；不得降级为 Python 3.10 或借用 5060 授权。

即使软件栈与 P4 对齐，GPU 型号、driver/cuDNN 等仍不同。因此这 55 个结果只补充执行覆盖和技术诊断：**不与 P4 或 RTX 5060 Ti 结果池化，不填充为统一硬件的 B_screen 面板，不用于模型排名/优劣结论，也不构成等价性证明。** RTX 8000 上不另跑 34-fit bridge；如果未来要提出跨硬件可比/池化主张，必须另立事前桥接方案、给出精度容忍标准并获得批准。Stage C/D 仍不在本次授权范围。
