# Asset-Dependence Exploration（探索性，最终版）

> 定位：**Exploratory Asset-Dependence Analysis**——只检查 interaction 符号与资产结构特征的关系，
> 不验证 volatility 假设、不声称因果。所有 association 仅为 exploratory。

## 1. 资产级表（n=10 个资产有分层结果；GSPC 使用 seed 2021，seed 稳健性见 §4）

| 资产 | seeds | 样本 | Δ_low | Δ_high | **Δ_interaction** | 交互 CI | 描述（方向 vs 显著性）| 高波动侧胜率 | RV | RelVol | \|ACF1\| | Jump | Trend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EURUSD | 1 | 497 | -0.00178 | -0.13888 | **-0.13710** | [-0.16421,-0.11061] | directionally negative and statistically supported | 70.6% | 0.00422 | 1.022 | 0.066 | 0.0105 | 0.63 |
| GSPC | 2 | 479 | -0.00047 | -0.01773 | **-0.01726** | [-0.02719,-0.00876] | directionally negative and statistically supported | 80.5% | 0.00799 | 1.046 | 0.076 | 0.0105 | 1.07 |
| USDJPY | 1 | 497 | +0.00490 | -0.00727 | **-0.01218** | [-0.02606,+0.00226] | directionally negative but statistically inconclusive | 39.9% | 0.00615 | 1.017 | 0.072 | 0.0105 | 0.73 |
| ETHUSD | 1 | 572 | +0.00198 | -0.00745 | **-0.00943** | [-0.01881,+0.00033] | directionally negative but statistically inconclusive | 50.3% | 0.03583 | 1.006 | 0.053 | 0.0105 | 0.84 |
| DJI | 1 | 479 | -0.04435 | -0.05270 | **-0.00835** | [-0.01939,+0.00266] | directionally negative but statistically inconclusive | 83.7% | 0.00749 | 1.065 | 0.074 | 0.0105 | 0.92 |
| TM | 1 | 479 | +0.07983 | +0.07178 | **-0.00805** | [-0.02759,+0.01129] | directionally negative but statistically inconclusive | 10.9% | 0.01737 | 1.034 | 0.069 | 0.0105 | 0.66 |
| SOX | 1 | 479 | +0.01603 | +0.02043 | **+0.00439** | [-0.00527,+0.01435] | directionally positive but statistically inconclusive | 35.1% | 0.02109 | 1.039 | 0.114 | 0.0105 | 0.68 |
| BTCUSD | 1 | 707 | -0.01274 | +0.00228 | **+0.01502** | [+0.00615,+0.02378] | directionally positive and statistically supported | 35.7% | 0.02542 | 0.983 | 0.084 | 0.0105 | 0.83 |
| BABA | 1 | 479 | -0.02964 | +0.01081 | **+0.04045** | [+0.02155,+0.05986] | directionally positive and statistically supported | 32.2% | 0.02546 | 1.052 | 0.082 | 0.0105 | 0.66 |
| NVO | 1 | 479 | +0.05966 | +0.15723 | **+0.09758** | [+0.04636,+0.14937] | directionally positive and statistically supported | 30.5% | 0.02204 | 1.107 | 0.156 | 0.0105 | 1.05 |

## 2. A. 符号分布（严格区分方向与显著性）

- 负号（Δ_interaction<0，非线性优势随波动上升）：**6** 个；正号：**4** 个
- 其中**显著**：负 2 个 / 正 3 个；CI 含 0：5 个
- 符号是否跨零：是

## 3. B. 结构特征 ↔ Δ_interaction（Spearman，探索性）

| 特征 | ρ | p（仅探索） | N | 方向 |
|---|---|---|---|---|
| realized_vol | +0.661 | 0.038 | 10 | 正 |
| relative_vol | +0.358 | 0.310 | 10 | 正 |
| acf1_abs | +0.733 | 0.016 | 10 | 正 |
| jump_ratio | +nan | nan | 10 | 正 |
| trend_persistence | +0.042 | 0.907 | 10 | 正 |
| sign_persistence | -0.289 | 0.418 | 10 | 负（特征越大越偏非线性） |
| skew | +0.067 | 0.855 | 10 | 正 |
| kurtosis | +0.345 | 0.328 | 10 | 正 |

> N 很小，单个 p 值不构成强证据；以下 LOO 才是稳健性关键。

## 4. C. Leave-one-asset-out 敏感性

| 特征 | ρ 范围 | 是否变号 | single-asset sensitive |
|---|---|---|---|
| acf1_abs | [+0.633, +0.817] | 否 | 否 |
| jump_ratio | [+nan, +nan] | 是 | 否 |
| kurtosis | [+0.100, +0.533] | 否 | 否 |
| realized_vol | [+0.533, +0.900] | 否 | 否 |
| relative_vol | [+0.117, +0.600] | 否 | 否 |
| sign_persistence | [-0.511, -0.119] | 否 | 否 |
| skew | [-0.067, +0.467] | 是 | ⚠️ 是 |
| trend_persistence | [-0.317, +0.283] | 是 | ⚠️ 是 |

## 5. D. 稳健性

### GSPC 分 seed（不作为独立资产并入主表）

| seed | Δ_interaction | 95%CI | 显著 |
|---|---|---|---|
| 2021 | -0.01205 | [-0.01545,-0.00876] | 是 |
| 2022 | -0.02248 | [-0.02719,-0.01792] | 是 |

### asset-level operator preference 与 volatility interaction（两者不混同）

| 资产 | 容量 128 维平均 ΔMSE（+ = 偏好线性） | Δ_interaction（波动率交互）|
|---|---|---|
| GSPC | +0.00606 | -0.01726 |
| BTCUSD | +0.29983 | +0.01502 |
| ETHUSD | -0.03205 | -0.00943 |
| USDJPY | +nan | -0.01218 |
| EURUSD | +nan | -0.13710 |
| SOX | +nan | +0.00439 |
| DJI | +nan | -0.00835 |
| BABA | +nan | +0.04045 |
| NVO | +nan | +0.09758 |
| TM | +nan | -0.00805 |

> 说明：前者是「该资产整体更偏好哪类算子」，后者是「优势是否随波动率变化」——是不同的量，不可互推。

## 6. 结论（协议 §六，只允许三种）

$$\boxed{\text{Candidate asset-level structural regularity}}$$

依据：符号跨零（6负/4正），且下列特征关联在 LOO 下方向稳定：realized_vol, acf1_abs。仍只是 **candidate**，不得称为 established law。

