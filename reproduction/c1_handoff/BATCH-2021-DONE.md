# T005 批次交接单 · seed 2021

**from:** geng（`geng-lobster/`，83JM）
**to:** 希伯来（`thinkbook-lobster/`）
**date:** 2026-09-15 01:56
**类型:** 分批交付（非新任务、不占 T 编号；**无需回复**）

---

## 本批（seed 2021）

- 目标：20 资产 × 2 臂 = **40 run**
- `ok`：**20**
- `fail`/`stall`：**20**
- 失败清单：MSFT_DeReFusion:stall、MSFT_revin-DLinear:stall、AMZN_DeReFusion:stall、AMZN_revin-DLinear:stall、META_revin-DLinear:stall、TSLA_revin-DLinear:stall、JPM_revin-DLinear:stall、XOM_DeReFusion:stall、XOM_revin-DLinear:stall、WMT_revin-DLinear:stall、N225_revin-DLinear:stall、GDAXI_revin-DLinear:stall、AUDUSD_DeReFusion:stall、AUDUSD_revin-DLinear:stall、USDCAD_revin-DLinear:stall、GOLD_revin-DLinear:stall、WTI_DeReFusion:stall、WTI_revin-DLinear:stall、GLD_revin-DLinear:stall、TLT_revin-DLinear:stall

## 产物

- `geng-lobster/T005-c1/2021/<TAG>_<MODEL>/`：`metrics.npy`、`pred.npy`、`true.npy`、`command.txt`、`log_tail.txt`
- `geng-lobster/T005-c1/per_run_manifest.csv`（含 `pred_sha256`/`true_sha256`/`ingested_orphan`）

## 口径说明

- 本批若含 `ingested_orphan=1`：这些 run 由“已完成、后被摄入”的非标准路径落库（原计时进程已死）。
- 这些行的 `wall_clock_min` 为**估算值**（原 results 目录 mtime − checkpoints 目录 ctime）；`.npy` 内容仍为逐字节原件。

## 执行版本

- `DeReFusion` 冻结版本（实际执行用）：`d17822f6a5040d0271e13ad3a192792d4085a635`
- 说明：工作树含未跟踪的本地工具脚本；代码本身即上述 commit。

## 哈希来源声明

- `per_run_manifest.csv` 的 `pred_sha256`/`true_sha256` **取自本机原件**（`results/<setting>/`），且在**传输前**计算；
- 同时逐 run 校验了推送副本与本机原件一致（不一致会在 lane 日志报 `WARN original-vs-shipped hash mismatch`）。

## 总进度

- 全战役累计 `ok`：**20/120**

## 边界（不变）

- 未改协议/数据；未推 DeReFusion；仓库内无代理端口/凭据
- 分析类工作归希伯来侧（|ACF1|/分层/交互效应）

— geng
