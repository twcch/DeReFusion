# 本机 GPU 执行可行性核查（2026-09-30）

## 结论

用户确认可以使用本机 GPU。当前 Windows 环境实际枚举到 Intel Arc 140T（16 GB），没有可由现有 CUDA 流程访问的 RTX 5060 Ti 或 RTX 8000。现有 Phase 1 训练入口只选择 CUDA、MPS 或 CPU；本地状态服务也将 Intel XPU 标记为未启用。因此用户允许使用 GPU，不等于现有 RTX 5060 授权或训练入口可以直接用于这块 Intel GPU。

本次只做了硬件、解释器和代码路径核查，没有新建 attempt、改写冻结方案或启动训练。

## Gate table

| Check | Input | Status | Evidence | Next permitted action |
|---|---|---|---|---|
| 本机 GPU 枚举 | Windows `Win32_VideoController` | observed | 枚举到 `Intel(R) Arc(TM) 140T GPU (16GB)`，驱动 `32.0.101.8864`；列出的 NVIDIA SMI 查询不可用，命令返回权限错误。 | 将当前机器资源记录为 Intel XPU；不把 NVIDIA SMI 错误解释成“无 GPU”。 |
| 仓库 Python 环境 | `.venv` 内 PyTorch 与 CUDA 探针 | requires-resolution | `.venv` 为 `torch 2.5.1+cpu`，`torch.cuda.is_available()` 为 false，设备数为0。 | 不用此环境启动 GPU 训练。 |
| 全局 Intel PyTorch 环境 | Python 3.13 的 `torch 2.13.0+xpu` | requires-resolution | 系统 Python 原有 wheel 导入在加载 `c10_xpu.dll` 时因缺失依赖失败；`pip show` 显示其声明依赖（oneAPI/SYCL/TBB 等）未安装。 | 不修改系统 Python；隔离环境另行验证。 |
| 隔离 Intel XPU 设备探针 | `%LOCALAPPDATA%\\Programs\\redefusion-intel-xpu-probe`，官方 PyTorch XPU wheel | eligible for device access | 隔离安装 `torch 2.13.0+xpu` 及 wheel 声明依赖成功；`torch.xpu.is_available()=true`、设备数1、名称为 Intel Arc 140T；128×64 输入的 Linear 前向、反向与同步成功。该探针不是 Phase 1 模型训练，也不验证仓库算子覆盖或实验保真。 | 可在隔离环境继续检查模型依赖与后端适配；不得因此复用 RTX 授权或跳过 Stage A。 |
| 训练后端支持 | `exp/exp_basic.py`、`run.py`、`exp/exp_long_term_forecasting.py`、`utils/recovery.py` | requires-resolution | 设备选择只有 `cuda`、`mps`、`cpu`；训练入口调用 `torch.cuda` 的随机种子、AMP、同步和内存接口。直接把 device 字符串改成 `xpu` 不足以执行完整流程。 | 为 XPU 另建版本化适配分支/执行轨道，并先完成 Stage A 校准与实现保真核查。 |
| 现有 RTX 5060 补充授权 | `authorization-local-5060-timesnet-remaining-v1.json` | inapplicable | 授权绑定 RTX 5060 Ti UUID、CUDA、PyTorch `2.7.1+cu128` 环境哈希及 TimesNet 55 行 manifest；本机枚举到的是 Intel Arc，仓库 `.venv` 指纹也不匹配。 | 不沿用旧授权、不向其输出目录写入 Intel 产物。 |
| Intel XPU 官方支持状态 | PyTorch Intel GPU 安装文档 | documented, prototype | PyTorch 文档称 Windows 客户端 Intel GPU 支持处于 prototype 阶段，并要求安装 Intel GPU 驱动及匹配的 XPU 运行时。此说明不证明本仓库模型代码已兼容。 | 先建立可验证环境与模型算子兼容记录；支持状况依据 [PyTorch Intel GPU 指南](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)。 |

## 下一步

1. 隔离 XPU 基础运行时已安装并通过设备/张量探针；不改现有 `.venv` 和已冻结的 CUDA 环境。
2. 对训练设备、AMP、随机数状态、内存统计、同步及缓存接口做 XPU 适配清单和实现保真审查。
3. 固定新的执行轨道 ID、环境指纹、输出根目录及仅限技术可行性/耗时的用途；不得将 Intel XPU 数值并入 P4、RTX 8000 或 RTX 5060 结果，也不据此排名。
4. 通过 Stage A 校准和资源核查后，再决定是否值得在该轨道重做完整 300-fit Stage B 屏幕。每 fit 的资源需求为 `TBD`，当前不从其他硬件的耗时外推。

Stage B 既有覆盖结论仍是：冻结逻辑设置 300/300 已有产物，但分布于 P4 与 RTX 8000 两轨；这份核查没有改变既有覆盖或授权状态，也没有生成单一硬件的完整 Stage B 面板。
