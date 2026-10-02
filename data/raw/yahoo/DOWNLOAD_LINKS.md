# 手动取数清单（拿回来我直接清洗入库）

**目标目录**：`C:\Users\26843\Desktop\project\repos\DeReFusion\data\raw\yahoo\`
**命名**：就用列表里的 `存为` 名字（如 `AAPL.csv`）
**格式**：任意（Yahoo 的 7 列或 Stooq 的 5 列都行），我负责清洗成 `date,Open,High,Low,Close`

时间窗口已固定：**2016-01-01 → 2025-12-31**（`period1=1451606400`，`period2=1767225600`）

---

## 方案 A：Yahoo 直接下载（推荐，最省事）

在能正常上网的设备上依次点开，浏览器会直接下载 CSV：

| 资产 | 存为 | 链接 |
|---|---|---|
| AAPL | `AAPL.csv` | https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| MSFT | `MSFT.csv` | https://query1.finance.yahoo.com/v7/finance/download/MSFT?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| AMZN | `AMZN.csv` | https://query1.finance.yahoo.com/v7/finance/download/AMZN?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| META | `META.csv` | https://query1.finance.yahoo.com/v7/finance/download/META?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| TSLA | `TSLA.csv` | https://query1.finance.yahoo.com/v7/finance/download/TSLA?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| JPM | `JPM.csv` | https://query1.finance.yahoo.com/v7/finance/download/JPM?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| XOM | `XOM.csv` | https://query1.finance.yahoo.com/v7/finance/download/XOM?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| WMT | `WMT.csv` | https://query1.finance.yahoo.com/v7/finance/download/WMT?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| 日经225 | `N225.csv` | https://query1.finance.yahoo.com/v7/finance/download/%5EN225?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| 德国DAX | `GDAXI.csv` | https://query1.finance.yahoo.com/v7/finance/download/%5EGDAXI?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| 恒生指数 | `HSI.csv` | https://query1.finance.yahoo.com/v7/finance/download/%5EHSI?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| 富时100 | `FTSE.csv` | https://query1.finance.yahoo.com/v7/finance/download/%5EFTSE?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| 罗素2000 | `RUT.csv` | https://query1.finance.yahoo.com/v7/finance/download/%5ERUT?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| 英镑美元 | `GBPUSD.csv` | https://query1.finance.yahoo.com/v7/finance/download/GBPUSD%3DX?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| 澳元美元 | `AUDUSD.csv` | https://query1.finance.yahoo.com/v7/finance/download/AUDUSD%3DX?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| 美元加元 | `USDCAD.csv` | https://query1.finance.yahoo.com/v7/finance/download/USDCAD%3DX?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| 黄金期货 | `GOLD.csv` | https://query1.finance.yahoo.com/v7/finance/download/GC%3DF?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| WTI原油 | `WTI.csv` | https://query1.finance.yahoo.com/v7/finance/download/CL%3DF?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| 黄金ETF | `GLD.csv` | https://query1.finance.yahoo.com/v7/finance/download/GLD?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |
| 长债ETF | `TLT.csv` | https://query1.finance.yahoo.com/v7/finance/download/TLT?period1=1451606400&period2=1767225600&interval=1d&events=div%2Csplit |

## 方案 B：Stooq（无需登录、无需 cookie，最不容易被拦）

浏览器直接打开就下载 CSV（**给你那边用**；我这台机器被 JS 挑战挡着）：

| 资产 | 存为 | 链接 |
|---|---|---|
| AAPL | `AAPL.csv` | https://stooq.com/q/d/l/?s=aapl.us&i=d |
| MSFT | `MSFT.csv` | https://stooq.com/q/d/l/?s=msft.us&i=d |
| AMZN | `AMZN.csv` | https://stooq.com/q/d/l/?s=amzn.us&i=d |
| META | `META.csv` | https://stooq.com/q/d/l/?s=meta.us&i=d |
| TSLA | `TSLA.csv` | https://stooq.com/q/d/l/?s=tsla.us&i=d |
| JPM | `JPM.csv` | https://stooq.com/q/d/l/?s=jpm.us&i=d |
| XOM | `XOM.csv` | https://stooq.com/q/d/l/?s=xom.us&i=d |
| WMT | `WMT.csv` | https://stooq.com/q/d/l/?s=wmt.us&i=d |
| 标普500 | `SPX.csv` | https://stooq.com/q/d/l/?s=%5Espx&i=d |
| 纳斯达克100 | `NDQ.csv` | https://stooq.com/q/d/l/?s=%5Endq&i=d |
| 道指 | `DJI.csv` | https://stooq.com/q/d/l/?s=%5Edji&i=d |
| 欧元美元 | `EURUSD.csv` | https://stooq.com/q/d/l/?s=eurusd&i=d |
| 英镑美元 | `GBPUSD.csv` | https://stooq.com/q/d/l/?s=gbpusd&i=d |
| 黄金 | `GOLD.csv` | https://stooq.com/q/d/l/?s=gc.f&i=d |
| WTI原油 | `WTI.csv` | https://stooq.com/q/d/l/?s=cl.f&i=d |

## 优先级（时间紧就只拿这些，≥6 个即可开跑）

`AAPL` `MSFT` `AMZN` `META` `TSLA` `JPM` `XOM` `WMT` —— 8 个美股，约 **8–12 分钟**点完。

## 拿回来之后

放到 `data/raw/yahoo/`，告诉我一声。我会：① 清洗成统一 schema；② 校验行数与日期覆盖（2016-01-01→2025-12-31）；③ 计算 |ACF1| 并排序确认自相关跨度；④ 再按新的同一来源队列跑独立验证。
