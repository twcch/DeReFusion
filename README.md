# DeReFusion：金融时间序列预测研究项目

> 给第一次接触本项目的计算机专业读者：本 README 先解释项目在解决什么问题，再介绍模型、代码、数据、实验结果、复现方式和当前限制。除专有名称、命令与代码标识外，正文使用中文。

- **项目性质：** 基于公开 DeReFusion 与 THUML Time-Series-Library 代码的独立研究分支，不是论文作者发布的官方代码包。
- **报告更新：** 2026-10-02；实验进度依据仓库中截至 2026-09-30 的阶段报告。
- **主要语言与框架：** Python、PyTorch。
- **研究主题：** 金融 OHLC 时间序列的多步预测，以及预测模型复杂度是否带来稳定收益。

参考论文：Chih-Chien Hsieh、Mu-Yen Chen，*DeReFusion: A Controlled Comparison of Soft Computing Fusion Strategies for Financial Time Series Forecasting via a Decomposition-Residual Architecture*，Applied Soft Computing 203 (2026), 116252，[DOI](https://doi.org/10.1016/j.asoc.2026.116252)。使用原论文方法时请引用原论文，并同时说明本仓库是独立研究分支。

## 1. 一分钟了解项目

金融市场每天会产生按时间排列的数据。例如某只股票每天有开盘价、最高价、最低价和收盘价。我们希望用过去一段时间的数据，预测未来若干天的收盘价。这属于**时间序列预测**。

这个仓库研究一个具体问题：把一个简单、可解释的线性预测分支，与一个更复杂的神经网络分支组合，能否比简单模型稳定地预测得更好？仓库既包含模型代码，也保存数据清单、实验规则、分析程序和结果审计记录。它的重点不是宣称某个模型“永远最好”，而是让读者能够追查每个结论是如何得到的、结论又有哪些边界。

对第一次阅读的人，可以先记住三点：

1. **模型主意：** DeReFusion 先用 DLinear 得到一个基础预测，再让 LSTM 与 Transformer 组成的残差分支预测基础模型没有解释的部分，最后将两个预测相加。
2. **科学结论需要克制：** 现有结果没有证明 DeReFusion 对所有资产、预测期限或环境都更好；若干探索性假设没有通过预先规定的检验。
3. **“跑完实验”不等于“证明假设”：** 仓库严格区分程序运行完成、数据与产物核验通过，以及统计结论得到支持这几件事。

## 2. 问题背景与基本术语

### 什么是 OHLC 数据？

OHLC 是一段时间内四种价格的缩写：Open（开盘）、High（最高）、Low（最低）、Close（收盘）。本项目的数据主要是日频金融数据。模型在每个预测窗口中读入四个价格字段，并以 Close（收盘价）作为预测目标。数据身份、日期范围、行数和文件哈希以 [`reproduction/results/dataset_registry.csv`](reproduction/results/dataset_registry.csv) 为准。

### 什么是训练、验证和测试？

为了评价模型是否能预测未来，时间数据按先后顺序切分：

- **训练集**用于调整模型参数；
- **验证集**用于选择训练轮数或检查训练过程；
- **测试集**模拟模型从未见过的未来数据，只在训练完成后用于最终评估。

如果未来测试数据提前参与训练、归一化参数计算或模型选择，就会产生数据泄漏，让成绩看起来比真实情况好。本项目的冻结方案要求按时间切分，并且只使用训练部分拟合标准化器。历史审计也明确记录了早期流程的局限，不会把旧结果包装成完全没有这些问题。

### 什么是 MSE？

MSE（均方误差）是预测值与真实值之差的平方平均值，越小代表该组数据上的误差越小。MSE 对大误差比较敏感。项目还会记录 MAE、RMSE 等指标。不同资产、不同预测期限或不同执行环境的 MSE 不能不加说明地混在一起比较；指标低也不自动代表统计显著、未来可获利或普遍优越。

## 3. DeReFusion 模型如何工作

一个输入窗口经过以下步骤：

1. **RevIN 归一化：** 根据当前输入窗口的统计量调整数据尺度，降低不同时间段数值水平变化造成的影响；预测后再把结果转换回原来的尺度。
2. **基础分支：** DLinear 对输入序列作分解，并给出一个相对简单的预测。
3. **残差分支：** 模型计算基础预测没有解释的剩余部分，再用线性层、LSTM 和 Transformer 建模其中的时间关系。
4. **融合：** 将基础预测与残差预测直接相加，得到最终预测。

```text
过去的 OHLC 窗口
       │
   RevIN 归一化
       ├─────────────┐
       ▼             ▼
   DLinear 基础预测   LSTM + Transformer 残差预测
       └────── 相加 ─┘
              │
       RevIN 还原尺度
              │
         未来收盘价预测
```

这里的“残差”是基础模型与实际序列之间未解释的部分，并不表示代码错误。LSTM 是处理序列信息的循环神经网络；Transformer 使用注意力机制处理序列中不同位置之间的关系。仓库还包含去除某个组件的消融模型、若干可学习融合门控变体、RevIN 包装的基线模型和可选的零样本基础模型适配器。

## 4. 项目结构导航

| 路径 | 内容 | 初学者可以怎样理解 |
|---|---|---|
| [`run.py`](run.py) | 训练与评估命令行入口 | 运行一次模型实验的总入口 |
| [`models/`](models/) | DeReFusion、消融、门控、基线和适配器 | 模型结构定义 |
| [`layers/`](layers/) | RevIN、注意力、嵌入等组件 | 神经网络积木 |
| [`exp/`](exp/) | 训练、验证和测试流程 | 实验循环与模型调用 |
| [`data_provider/`](data_provider/) | 数据集读取、切分和批次加载 | 将 CSV 转成模型输入 |
| [`dataset/`](dataset/) | 已纳入版本管理的 OHLC 数据文件 | 本项目使用的数据样本 |
| [`requirements/`](requirements/) | 分组的软件依赖 | 安装运行环境所需的 Python 包 |
| [`reproduction/analysis/`](reproduction/analysis/) | 数据检查与结果复算脚本 | 审核数据、重算已保存的统计结果 |
| [`reproduction/results/`](reproduction/results/) | 小型表格、数据注册表和分析摘要 | 可版本管理的结构化结果 |
| [`reproduction/c1_handoff/`](reproduction/c1_handoff/) | C1 阶段交接、清单和复算资料 | 该实验的审计材料 |
| [`docs/`](docs/) | 研究审计、预注册与方法说明 | 项目的规范和深入说明 |
| [`reports/`](reports/) | 阶段报告和证据链 | 结果、审查与决策记录 |
| [`reports/evidence_closure/`](reports/evidence_closure/) | 编号的历史证据链 | 按时间追溯探索、规则和结论 |
| [`research-ops-site/`](research-ops-site/) | 研究运行状态展示网站的源代码 | 独立的前端项目，不是模型训练入口 |

大型训练日志、模型检查点和部分原始预测产物可能仅保存在运行机器，不一定随 GitHub 仓库发布。远程仓库中的报告会注明结论所依据的产物和限制；如报告需要未随仓库发布的本地文件，克隆代码仓库本身并不足以完整重现该项实验。

## 5. 数据范围与注意事项

仓库数据注册表列出 **61 个日频 OHLC CSV 文件、合计 149,474 行**。每个文件的哈希、行数、日期范围和质量审查可在数据注册表与历史审计报告中核查。数据来源、分组和使用边界也应结合具体报告阅读；不能因为某个 CSV 存在于目录中，就默认它获准进入每一项正式比较。

已记录的数据质量问题包括：

- 五份外汇数据存在少量 OHLC 高低价包络不一致。原始数据保留，没有悄悄修改；
- WTI 原油数据保留了 2020 年 4 月的非正价格。涉及对数收益的计算在这些位置没有定义，不能通过加常数或删行来隐去；
- 部分数据被标为 supporting-unregistered（仅供支持用途、未注册为正式样本），不应事后加入确认性分析；
- 数据质量问题可能影响特定特征或目标定义，需要由相应实验报告逐项说明，不能只看总体模型分数。

详细说明：[`reproduction/results/dataset_registry.csv`](reproduction/results/dataset_registry.csv)、[`docs/REPOSITORY_AUDIT_2026-09-18.md`](docs/REPOSITORY_AUDIT_2026-09-18.md)。

## 6. 研究问题、实验和当前结论

### 6.1 早期融合与结构探索

- **学习式融合门控：** 在已测试设置下，波动率门控没有优于直接相加；这不代表所有可能的门控都已被排除。
- **按样本动态路由：** 没有获得足够稳定的支持，因此当前不能声称已经找到可靠的逐窗口模型选择器。
- **Gate A（结构特征检验）：** 按事先冻结的规则判定为 **FAIL**。其中 `|ACF1|` 与算子偏好的探索性联系不能作为已验证的选择规律。
- **F1（更多资产上的宽度检验）：** 7 个追加资产中，4 个符合预先定义的种子与模型宽度稳定条件，3 个不符合。结果只支持有限的资产异质性倾向，不能说每个资产都具有固定的算子偏好。
- **C1（前瞻性检验）：** 120/120 次运行完成，但预先指定的候选关系未能复现，结果为 **FAIL**（报告记录 `rho=-0.1519, p=0.5227`）。独立盲复算支持失败结论的稳健性，同时指出交接规则曾有缺口；详细范围见编号报告 26–29。

以上 FAIL 表示对应的预先声明假设没有通过规则，并不是“项目代码运行失败”。否定结果与执行异常被保留下来，因为它们是研究记录的一部分。

### 6.2 DeReFusion 与现代基线：Phase 1

Phase 1 的主要问题是：在固定数据、切分与训练条件下，DeReFusion 是否比 RevIN-DLinear 基线持续带来足够收益，以证明较复杂残差分支值得承担其成本？方案将工作分成校准、筛查、主要配对比较和时间稳健性四个阶段。

| 阶段 | 计划规模 | 截至 2026-09-30 的核实进度 | 可以怎样解读 |
|---|---:|---|---|
| Stage A：运行校准 | 15 次 | 15/15 正式 GPU 校准收据通过 | 说明规定任务能运行、张量形状和产物流程可用；按方案，校准分数不是模型效果证据。 |
| Stage B：现代基线筛查 | 300 项 | P4 轨道 245 项 + RTX 8000 补充轨道 55 项，合计覆盖 300/300 个设置 | 两条硬件/软件执行轨道不得合并排名或声称数值等价。P4 四模型子面板有独立描述性复核，但不是完整五模型确认性排名。 |
| Stage C：主模型配对比较 | 360 次 | 360/360 次；30 个资产、2 个预测期限、3 个随机种子 | 预测键、目标数组与文件哈希配对核验通过；现有报告只给分层描述性点估计。 |
| Stage D：时间稳健性 | 144 次 | 0/144，清单仍为 planned | 尚未运行；没有本阶段专属执行授权。 |

Stage C 的指标是 `DeReFusion MSE − RevIN-DLinear MSE`，按样本组、期限和随机种子分开报告；负值表示该分层中的 DeReFusion 平均点估计较低。现有 12 个分层点估计从大多数层为负，到部分层接近零或为正，因而不宜用单一平均数概括。更关键的是，冻结方案要求资产聚类 bootstrap 区间，但没有在结果产生前固定所需的随机种子、重抽样次数、区间算法和跨期限/种子汇总规则。报告因此**没有**事后补造这些参数，也**没有**声称 DeReFusion 已得到确认性优越结论。

读者应区分：产物覆盖完整、预测数组配对通过，说明数据与文件链条经核验；它们本身不能替代已完整预先规定的统计判据。Stage B 的跨硬件排名、Stage C 的确认性区间、Stage D 的启动均受报告中的门控限制。

- [Phase 1 总进度和结果边界](reports/phase1/phase1-experiment-data-and-progress-20260930.md)
- [下一步门控与允许动作](reports/phase1/phase1-next-steps-gate-20260930.md)
- [冻结预注册方案](docs/PHASE1_PREREGISTRATION.md)
- [Phase 1 描述性证据索引](reports/README.md)

### 6.3 关于 Navier–Stokes 的边界

Navier–Stokes 是描述流体运动的一组方程。有过把其算子结构作为神经网络架构灵感的研究设想，但本仓库没有证明金融 OHLC 数据满足相应物理状态变量、守恒律、边界条件或偏微分方程残差。因此，Navier–Stokes 相关模型目前是**未经测试的想法**：既没有被本仓库验证，也不能说已经被本仓库否定。不得把它描述成金融市场的真实物理定律或本项目已经实现的成果。

## 7. 安装环境与运行一个示例

建议使用 Python 3.11，并根据自己的硬件安装匹配的 PyTorch。依赖分为核心、CUDA 和可选基础模型几组，不需要为了运行核心模型而一次安装全部大型可选依赖。

```bash
# 先安装与你的系统及硬件相匹配的 PyTorch。
# 以下文件示例对应 CUDA 12.1；CPU 或其他 GPU 环境请选用适配版本。
pip install -r requirements/torch-cu121.txt

# 安装核心模型与分析依赖
pip install -r requirements/core.txt
```

在仓库根目录运行以下单次训练示例（需要已安装匹配的 PyTorch，并可访问示例数据）：

```bash
python run.py \
  --task_name long_term_forecast --is_training 1 \
  --model_id GSPC_96_24 --model DeReFusion \
  --data custom --root_path ./dataset/ --data_path GSPC-2016-2025.csv \
  --features MS --target Close --freq b \
  --seq_len 96 --label_len 48 --pred_len 24 \
  --enc_in 4 --dec_in 4 --c_out 1 \
  --d_model 32 --moving_avg 25 --train_epochs 30 --batch_size 32 \
  --learning_rate 0.0001 --patience 5 --lradj cosine \
  --rand_seed 2021 --no_use_gpu
```

参数含义：`seq_len=96` 表示输入过去 96 个时间步；`pred_len=24` 表示预测未来 24 个时间步；`label_len` 是解码器使用的历史部分长度；`--no_use_gpu` 强制使用 CPU。数据参数的准确含义由 `run.py` 与相应数据加载器实现。这个示例用于说明如何调用代码，并不等于任何冻结研究方案的完整复现，也不能直接与报告中的确认性阶段互换。

Chronos、Moirai、TimesFM 等零样本模型需要可选依赖：

```bash
pip install -r requirements/foundation.txt
```

Linux/CUDA 环境下的 Mamba 依赖也单独列出。依赖版本、平台差异和隔离环境建议见 [`requirements/README.md`](requirements/README.md)。在 Windows CPU 环境可将上面的反斜线续行改成 PowerShell 反引号，或将命令写在一行。

## 8. 检查数据与复算已有分析

以下命令用于核对数据或根据已留存的小型结果表复算摘要；它们不会替代模型训练，也不会自动授权新的正式实验：

```bash
# 审查 61 份数据的结构、日期和哈希
python reproduction/analysis/dataset_audit.py

# 根据保留的 F1 结果表复算摘要
python reproduction/analysis/f1_analysis.py

# 验证 Phase 1 冻结清单（不启动训练）
python reproduction/analysis/build_phase1_manifest.py

# 审计冻结的日期切分（不启动训练）
python reproduction/analysis/audit_phase1_date_splits.py
```

Phase 1 训练启动器默认是只读预览；真正训练还需要与代码提交、实验阶段和设备绑定的独立版本化授权以及通过的环境指纹。不要把修改参数后重跑得到的结果称为同一冻结方案的复现。

## 9. 建议阅读顺序

如果你刚克隆项目，建议按这个顺序阅读：

1. 本 README，了解项目问题、模型、数据和结论边界；
2. [`docs/REPOSITORY_AUDIT_2026-09-18.md`](docs/REPOSITORY_AUDIT_2026-09-18.md)，了解仓库审计、数据质量和历史方法问题；
3. [`reports/README.md`](reports/README.md)，查看权威结论索引与证据时间线；
4. [`reports/phase1/phase1-experiment-data-and-progress-20260930.md`](reports/phase1/phase1-experiment-data-and-progress-20260930.md)，了解最近阶段进度；
5. [`docs/PHASE1_PREREGISTRATION.md`](docs/PHASE1_PREREGISTRATION.md)，了解哪些比较在看到结果前已冻结；
6. [`docs/LITERATURE_AND_NEXT_PLAN.md`](docs/LITERATURE_AND_NEXT_PLAN.md)，了解相关工作与后续研究边界；
7. 对某个历史假设感兴趣时，再沿 `reports/evidence_closure/` 的编号报告从预注册读到结果与审计。

较早报告可能记录当时的历史状态，后续报告会修正或补充它们。请优先看 `reports/README.md` 指向的当前权威报告，不要把历史计划或旧的运行快照误读为当前结论。

## 10. 可复现性与结论边界

本项目尽量保存冻结配置、代码版本、数据 SHA-256、预测键、原始预测/真值、指标与收据，从而让其他人检查样本是否配对、数组是否对应同一预测目标、保存指标能否由数组复算。小型摘要和注册表随 Git 管理；检查点、完整训练产物等大型文件可能留在本地执行目录，不承诺仅凭 GitHub 克隆即可重建每一次训练。

复现时特别留意：

- 不要把训练完成数当作模型显著优越的证据；
- 不要合并不同硬件轨道并称它们数值等价；
- 不要从已经观察到的测试结果挑资产、特征、阈值或资源预算；
- 不要把描述性点估计改称置信区间、显著性检验或确认性结论；
- 不要静默改写数据、隐藏负结果或把未测试想法写成验证成果；
- 对新的实验，先固定问题、数据、切分、比较对象、统计规则和资源安排，再开始查看结果。

研究流程的更严格约束见 [`docs/EXPERIMENTAL_DOCTRINE.md`](docs/EXPERIMENTAL_DOCTRINE.md) 与 [`docs/PHASE1_PREREGISTRATION.md`](docs/PHASE1_PREREGISTRATION.md)。

## 11. 引用、来源与许可

引用 DeReFusion 方法时请引用上方 2026 年论文，并明确说明代码来自独立研究分支。代码许可见 [`LICENSE`](LICENSE)；THUML Time-Series-Library 的来源记录在相关源文件头部和 Git 历史中。数据和预训练模型可能有各自的使用条款，使用前需核对对应来源。

---

如果你是第一次读这个仓库，最重要的入口是本页的项目结构、当前结论和报告索引。项目代码回答“如何运行模型”，冻结方案与审计报告回答“实验可以支持什么结论”。
