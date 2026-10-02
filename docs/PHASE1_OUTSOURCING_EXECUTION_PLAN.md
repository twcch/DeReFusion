# DeReFusion Phase 1 GPU 外包与执行控制计划

状态：执行准备中，不构成 Stage B--D 开跑授权  
协议：`phase1-v1.2-2026-09-28`（仅 Stage A 证据范围修订；科学问题与 B--D 门禁不变）
制定日期：2026-09-28

## 1. 决策摘要

依据 `docs/PHASE1_AMENDMENT_V1.2.md`，P4 `calibration/` 下 15/15 个
`calibration_pass` 包是 Stage A 权威执行标定包；CPU `calibration-cpu/` 下 14 个包
原样保留为历史证据，不纳入当前 Stage A 有效包与 attempt 唯一性范围。Stage A 已按
执行证据收口；不得再按旧计划补跑 A05/A15，也不得把标定损失用于模型优劣判断。

Stage B--D 仍受预注册 §8 门禁约束。在以下条件全部满足前，任何 Stage B--D 训练均
未获授权：

1. Stage A 的 15 个 logical runs 均由 v1.2 指定的 P4 权威包覆盖且完整性校验通过；
2. rolling/date split 的参数传递、边界语义和回读验证经过仓库审查；
3. 外包包、环境指纹、摄入工具和幂等重试测试通过；
4. 冻结 commit 与数据逐文件 SHA-256 已记录。

## 2. 当前证据状态

- P4 权威包集：15/15 `calibration_pass`，对应 receipt 列出的产物哈希已复核无差异；
  该状态仅表示执行标定与产物完整性通过，不表示科学结果通过。
- CPU 历史包集：14 个包，13 个 `calibration_pass`、1 个 A05
  `resource_blocked`；完整保留，不能与 P4 包合并计为当前有效 attempt。
- Stage A 的 P4 TimesNet 单次耗时与完整环境指纹见 v1.2 amendment 和证据目录；这些
  运行仅用于资源与执行链标定，不构成模型优劣证据。
- 现有 B/C/D manifest 分别冻结为 300、360、144 fits；状态均为 `planned`。
- 其余 744 fits 应称为“非 TimesNet fits”，不能笼统称为 DLinear 系。

## 3. 外包执行边界

### 3.1 设备一致性

- Stage B 的 300 fits 必须在同一 GPU 型号、同一容器镜像、同一 Python/PyTorch
  环境中完成。不得只迁移 TimesNet 而把同一 Stage B 的其他模型留在 CPU。
- Stage C/D 可以使用 CPU，但每一个预注册配对内的两条模型臂必须具有相同设备、
  软件环境和数据字节；设备类别进入 config fingerprint 和 receipt。
- 预算受限时，最低可执行单元是 TimesNet + DLinear 的完整锚定面板，不是孤立的
  TimesNet runs；该缩减必须先形成协议 amendment，不能事后决定。

### 3.2 环境锁

预钉核心环境：

```yaml
python: 3.11.15
torch: 2.5.1+cu121
torch_cuda: "12.1"
numpy: 2.1.2
pandas: 2.3.3
scikit_learn: 1.7.2
container: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
```

远端首次启动只执行环境探测，不执行训练。合格条件包括
`torch.__version__ == "2.5.1+cu121"`、`torch.version.cuda == "12.1"`、
`torch.cuda.is_available() is True`。随后冻结镜像 digest、GPU 名称/UUID（如可用）、
驱动、cuDNN、操作系统、CPU、RAM 和完整 Python 包清单。后续不一致立即 fail-stop。
系统 `nvcc` 版本不作为 PyTorch CUDA runtime 的判据。

Stage A 收口后的 P4 主机身份观测见
`docs/PHASE1_P4_HOST_IDENTITY_SUPPLEMENT_2026-09-28.json`：GPU UUID、驱动、操作系统
及冻结训练解释器均已从原 P4 实例读取。该补充是运行后主机级佐证，不回写原始 receipt。
P4 平台未向实例暴露正在运行镜像的不可变 digest；该项仍是 B--D 前置门禁，环境锁中的
镜像 tag 不能替代 digest。

## 4. Attempt 与重试规则

1. logical run 与物理 attempt 分离。所有新目录使用
   `<logical_run_id>__attempt-NN`；历史无后缀目录兼容解释为 attempt 1，原位保留。
2. 成功状态禁止为了获得更好指标而重跑。只有 `resource_blocked`、`failed`、
   `timed_out`、`artifact_invalid`、`infrastructure_interruption` 等技术失败可重试。
3. 每次重试必须记录新的 `retry_authorization`，并引用前一 attempt、技术失败原因、
   批准人和 UTC 时间；首次执行授权不能自动扩展到后续重试。
4. 默认最多两个 attempts。第三次及以后须单独升级审批，并形成可审计说明。
5. 摄入工具不得直接选“最大 attempt”。每个 logical run 必须恰有一个经验证的有效
   成功 attempt：零个为未完成，多个为冲突并 fail-stop。旧成功只有在重试授权之前
   被明确判定无效时，才可由新成功替代。
6. 任何 attempt 都不得覆盖此前日志、数组、checkpoint、metrics 或 receipt。

每份 receipt 至少包含：`logical_run_id`、`attempt`、`attempt_id`、
`supersedes_attempt`、`retry_reason`、`retry_authorization`、`created_at_utc`、
`protocol_version`、冻结 commit、config fingerprint、数据 SHA-256、完整命令、环境
指纹、数组形状、有限值检查、诊断值及逐文件 SHA-256。

## 5. Runner 与防覆盖设计

- 采用一个共享执行引擎 `phase1_runner.py`，B/C/D 入口只负责选择冻结 manifest；不得
  复制三套训练、打包、重试或校验逻辑。
- runner 必须逐行读取 manifest，不硬编码资产、行数或 Stage A 路径。
- `split_mode=dates` 时必须显式传递 `train_end`、`val_end`、`test_end`，并从运行回执
  回读核对；任一字段缺失或下游不支持即拒绝执行，绝不回退到 ratio split。
- 输出身份必须包含协议版本、设备类别、attempt 和规范化 config SHA。学习率、batch
  size、epochs、patience、dropout 或设备变化必须产生不同输出目录。
- 设备采集可以由 runner 完成，不改变模型数学与指标定义；若训练模块无法提供必需的
  实测值，允许增加纯仪表字段，但不得借机改变数据、模型或评价逻辑。
- dry-run、单行 smoke test、断点续跑、超时和网络中断均需测试；只有完整成功包可以
  进入摄入清单。

## 6. 外包包结构

仓库内权威源放在 `reproduction/handoff/phase1_outsourcing/`。压缩包由
`make_package.py` 从冻结 commit 确定性导出，不把二进制压缩包提交进 Git；导出后
记录包本身的 SHA-256。

```text
phase1_outsourcing/
  README.md
  PROTOCOL_FREEZE.md
  FROZEN_COMMIT.txt
  package_manifest.json
  environment/
    ENVIRONMENT_LOCK.md
    env_fingerprint.py
  manifests/
    B_screen.manifest.csv
    C_confirmation.manifest.csv
    D_temporal_robustness.manifest.csv
  data/
    DATA_MANIFEST.csv
    verify_data.py
  runners/
    phase1_runner.py
    run_stage_b.py
    run_stage_c.py
    run_stage_d.py
  tools/
    ingest_check.py
  make_package.py
```

每个成功 attempt 的最小交付物为 `pred.npy`、`true.npy`、`metrics.npy`、
`checkpoint.pth`、`run.log`、`receipt.json`。包内清单覆盖所有源文件和数据文件的原始
字节 SHA-256。

## 7. 已确认、必须关闭的九项缺陷

1. 失败 receipt 被当成完成；
2. 缺少不可覆盖的 attempt 机制；
3. calibration runner 对 Stage A 硬编码；
4. Stage D 日期切分参数未传递；
5. 缺少 GPU/驱动/CUDA/cuDNN 指纹；
6. `--no_use_gpu` 被硬编码；
7. 缺少 Phase 1 通用摄入和清单工具；
8. setting 未覆盖关键超参数与设备，存在静默覆盖；
9. Windows 换行转换可能破坏冻结哈希。

其中 1、2、8 必须通过单元测试验证失败/重试/多成功冲突；3--7 必须通过 B、C、D
各一条 dry-run 和不产生科学结果的 smoke test；9 必须通过 fresh clone 的字节哈希
复核。仅有代码存在不等于缺陷关闭。

## 8. 执行顺序

1. 允许当前 A10 完成，落盘后停止 CPU 队列；
2. 修 runner、外包包、摄入校验与测试；
3. 在租用 GPU 上做环境探测并冻结完整指纹；
4. Stage A 已依 v1.2 收口；不再启动 A05/A15 补跑。后续工作从滚动/日期切分审查开始，
   不得据此自动启动 Stage B；
5. 审查 rolling/date split，实现合成边界测试与回执回读；
6. 形成门禁审查记录；
7. 门禁通过后另行签发 Stage B 执行授权；
8. Stage B 完整完成、摄入、分析计划检查后再推进 C；随后才是 D。

任何步骤发生异常时只保留证据并停住，不自动改协议、不修数据、不删除失败 attempt、
不根据中间指标改变模型或样本。
