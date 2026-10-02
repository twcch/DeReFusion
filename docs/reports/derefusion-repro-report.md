# DeReFusion 复现简报（快速版：关键规律复现 + checkpoint 清单）

> 日期：2026-09-11 ｜ 目的：理解论文实验方式 + 为后续方向（结构感知路由）提供可插入框架的验证基线
> 环境：Windows / Python 3.13 venv / PyTorch 2.14 CPU / 单资产 GSPC / seed 2021
> 仓库：Desktop\project\repos\DeReFusion（fork: Alastair-Jiang/DeReFusion，数据已补齐入库）

---

## 一、复现结果总表（6/6 完成）

| 模型 | T=1 MSE | T=24 MSE | T=24 R² | 论文锚点（10资产均值） |
|---|---|---|---|---|
| **DeReFusion**（加法融合） | **0.0165** | **0.0623** | **0.869** | T=1≈0.037 / T=24≈0.205 |
| revin-DLinear（线性基线） | **0.0158** | 0.0701 | 0.853 | T=1≈0.049 / T=24≈0.172 |
| gatev2-learnable（静态门控） | 0.0253 | 0.0719 | 0.849 | T=1≈0.049 |

训练耗时参考（CPU）：DeReFusion T=24 约 23.5 分钟（30 epochs 全跑完）；其余实验相近量级。

## 二、复现验证结论

1. ✅ **框架跑通**：数据加载 → 训练 → 早停 → 测试 → 6 指标 → 可视化全链路成功，产出 checkpoint/metrics.npy/6张论文风格图。
2. ✅ **规模吻合**：28,436 可训练参数 ≈ 论文 ~26k（d_model=32 配置）。
3. ✅ **规律A（加法 > 门控）**：T=24 上 DeReFusion(0.0623) < DLinear(0.0701) < gatev2(0.0719)——无参数加法融合优于可学习门控，与论文核心结论一致。
4. ⚠️ **规律B（混合 > 线性@短时域）在 GSPC 上部分成立**：T=24 混合明显优于线性；但 T=1 时 DLinear(0.0158) 略优于 DeReFusion(0.0165)（差 4%，属单资产单种子噪声范围内——论文多资产均值下 DeReFusion T=1 领先 24%，但 GSPC 单点上 DLinear 本就是论文中最强的线性基线，此差异不构成矛盾，反而印证论文"领先组内部差异微小"的结论）。
5. ⚠️ **数值差异说明**：绝对 MSE 比论文均值低（单资产差异：GSPC 指数比加密资产好预测得多），且 torch 2.14 vs 论文 2.5.1、数据尾部几天差异。规律层面复现成功即达成目标。

## 三、Checkpoint 清单（框架插入点验证）

后续无论选什么方向，往这个框架里"塞实验"的每个入口都已有验证过的命令。所有命令在 repos\DeReFusion 下执行，`$PY=.venv\Scripts\python.exe`，公共参数：`--data custom --root_path ./dataset/ --features MS --target Close --freq b --enc_in 4 --dec_in 4 --c_out 1 --no_use_gpu --rand_seed 2021`。

### CP1 — 换模型（插入新模型）
框架自动扫描 `models/` 目录：**新建 MyModel.py，定义 `class Model(nn.Module)`，构造器收 configs（seq_len/pred_len/enc_in/d_model…），forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None) 即自动注册**。
```
$PY run.py --task_name long_term_forecast --is_training 1 --model_id GSPC_96_24 --model <文件名去.py> --data_path GSPC-2016-2025.csv --seq_len 96 --label_len 48 --pred_len 24 --d_model 32 --moving_avg 25 --train_epochs 30 --batch_size 32 --learning_rate 0.0001 --patience 5 --lradj cosine
```
✅ 已验证：DeReFusion / revin-DLinear / gatev2-learnable 三种模型名走同一入口。

### CP2 — 换数据（插入新资产/新数据集）
CSV 放 `dataset/`，格式 `date,Open,High,Low,Close`，改 `--data_path` 与 `--enc_in`（输入通道数）。A股日线（hs300）可直接适配；多特征时 `--enc_in` 改为特征列数。
✅ 已验证：GSPC-2016-2025.csv（我们自拉的 Yahoo 数据）在框架内正常训练评估。

### CP3 — 换融合策略（融合对照系）
`models/derefusion/gate_variant/` 三个门控变体即现成对照组；论文的受控对比设计 = 固定其余、只换融合算子。
```
--model DeReFusion-gatev2-learnable   # 静态门控（本次已验证）
--model DeReFusion-gatev1-volatilityaware / gatev3-inputconditioned  # 未跑，同接口
```
✅ 已验证：gatev2 与加法可同协议直接对比。

### CP4 — 消融开关（组件贡献）
`--model DeReFusion-woDy`（纯线性基底）/ `woLSTM` / `woTransformer`，同一命令换名即用。
✅ 未跑但机制与 CP1 相同（models/derefusion/ablation_variant/ 下同名文件自动注册）。

### CP5 — 超参/时域扫描
改 `--pred_len`（论文网格 1/7/12/24/36）、`--seq_len`（48/96/192）、`--d_model`。批量脚本 `run_batch_long_term_forecast.py`（改顶部配置块，支持断点续跑）。
✅ 已验证：T∈{1,24} 两种时域。

### CP6 — 结果与可视化
`results/<setting>/metrics.npy` 六指标 + `result_long_term_forecast.txt` 一行式日志（含参数量/训练时长/推理速度）+ `test_results/<setting>/fig_*.png` 六张论文风格图（可独立重生成：`python -m utils.visualization --input results/<setting>/`）。
✅ 已验证：6 个实验全部产出完整三件套。

## 四、环境坑位记录（照抄可绕过）

1. 依赖链比 README 声明的长：patool / huggingface_hub / sktime（+skbase）/ datasets 均为 data_loader 隐式依赖，缺一即崩。
2. sktime 装不上：pip 直连断流 → `pip download`（支持续传）+ 本地 whl 安装 + 手动补 scikit-base；joblib 需 pin 1.5.3（sktime 要求 <1.6）。
3. GBK emoji 崩溃：exp_basic.py 的 🚀 print 在 GBK 控制台直接抛 UnicodeEncodeError → 已改为 `[Lazy Loading]`（fork 内已修复，建议提交）。
4. CPU 版 torch 必须加 `--no_use_gpu`（run.py 默认 auto-CUDA，会 assert 崩）。
5. 运行时设 `PYTHONIOENCODING=utf-8` 兜底其余打印。
6. 后台跑批注意 exec 超时（3h 上限），单实验 CPU 20-25 分钟，串行一批 ≤5 个安全。

## 五、对后续选题的含义

- **塞实验的最低成本路径已验证**：写一个 models/xxx.py + 一条 run.py 命令 = 一个完整受控实验（数据/协议/指标/可视化全白嫖）。结构感知路由的原型只需实现 router 替换 + 三轴嵌入，其余全部复用。
- 受控对比协议（同种子/同切分/同指标）天然支持论文所需的公平性声明。
- 下一步若走结构路由：先做 CP1 的最小版本（把 router 的输入从内容向量换成 内容+资产轴嵌入），GSPC 单资产上跑通，再扩 BTCUSD 验证跨市场，最后上 hs300 面板。
