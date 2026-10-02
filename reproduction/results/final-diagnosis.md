# Final Diagnosis — Asset-Level Operator Heterogeneity

## Executive Summary（≤1 页）

**1. 当前最可靠的发现**
- *Operator suitability is asset-dependent*：容量 128 维下，ETHUSD 全部状态/种子偏好非线性（平均 ΔMSE -0.03205），BTCUSD 全容量偏好线性（+0.29983），GSPC 高容量接近打平（+0.00606）。
- 主实验（23 维）观察到的"线性全面占优"**部分是容量假象**：the earlier rigidity was capacity-related。

**2. 当前被排除的假设**
- ❌ Sample-level routing：未发现稳定的 `StructuralState → OperatorPreference`（Case 1/2 均不成立；
  反转是资产级全状态而非状态级；状态间 Range(ΔMSE) 未随容量扩大；方向随种子波动）。
- ❌ "非线性算子普遍更差"：ETHUSD 在足够容量下全面偏好非线性。
- ❌ "波动率 → 门控"：gatev1 实测劣于无参数加法 8.6%。

**3. 尚未解决的 ambiguity**
- 结构性特征是否与 operator heterogeneity 相关：见 asset-dependence-exploration.md（当前判定：**Candidate asset-level structural regularity**）。
- 代理算子（展平窗口线性 / MLP）与框架内真实分支的保真度差异：结论不能外推到"非线性无用"。

**4. 本研究决策（§九 Path A/B/C/D）**
- 当前资产级异质性明确（operator evidence 强），但 sample-level routing 无证据（routing evidence 无）。
- 建议：**Path B（Nonlinear operator deserves targeted enhancement）**

**5. 是否允许重新考虑 NS**
- ❌ 本阶段**不允许**。协议规定：只有能明确回答 "there exists a reproducible context in which a stronger nonlinear operator is justified" 才可在下一阶段把 NS 作为**候选**之一（与 capacity-matched MLP 对照），且不得由 `ETH → nonlinear` 推出 `ETH → NS`。

---

## Q1–Q5 必答（协议 §九）

- **Q1 是否存在稳健 sample-level routing evidence？** → **No**（结构状态→算子偏好未发现稳定关系：Case 1/2 均不成立；反转是资产级全状态而非状态级；状态间 Range(ΔMSE) 未随容量扩大；方向随种子波动）
- **Q2 是否存在真实 asset-level operator heterogeneity？** → **Yes**（基于容量对照结果：ETHUSD 64/128 维下 36/36 观测偏好非线性（128 维平均 ΔMSE -0.03205）；BTCUSD 全容量偏好线性（+0.29983）；GSPC 高容量接近打平（+0.00606）。同时保留：23-d nonlinear bottleneck materially affected the earlier ETH result）
- **Q3 这种 heterogeneity 能否由现有 structural features 解释？** → 由 7 资产 Spearman + LOO 判定：**PATH 1 — Candidate asset-level structural regularity**（LOO 稳定特征：realized_vol, acf1_abs）
- **Q4 当前主要 uncertainty 在哪里？** → 主要在 **(d) asset dependence**（资产级差异已知但尚无稳定结构解释）与 **(b) operator form**（何种非线性算子形态在何种资产上值得加强）；**不**把 unknown 伪装成 representation bottleneck：目前只能说 current structural features do not yet establish a general sample-level operator-selection mechanism。routing（a）方向已基本排除。
- **Q5 下一阶段是否有资格重新考虑 NS？** → **可以（仅作为候选之一）**。判据：there exists a reproducible context in which a stronger nonlinear operator is justified；即使满足，也只能与 capacity-matched MLP 做受控对比，且不得由 ETH→nonlinear 推出 ETH→NS。

## 四类证据分列（协议 §七）

**A. Routing evidence**：$$\boxed{\text{No robust sample-level routing evidence}}$$

**B. Operator evidence**：$$\boxed{\text{Strong asset-level heterogeneity is observed}}$$
　ETH: nonlinear favored at adequate capacity（128 维 ΔMSE -0.03205）；BTC: linear favored（+0.29983）；GSPC: near parity（+0.00606）。

**C. Representation evidence**：Current structural features do not yet establish a general sample-level operator-selection mechanism.（**不**断言 representation 是唯一瓶颈）

**D. Asset-dependence evidence**：Candidate asset-level structural regularity（N=10 资产）

## 明确禁止的逻辑跳跃（协议 §十，已自查）
- `ETH → nonlinear` ⇏ `NS should work`；`BTC → linear` ⇏ `linear is generally superior`；
- `no routing evidence` ⇏ `representation definitely insufficient`；`correlation` ⇏ `causal mechanism`；
- `one asset` ⇏ `universal rule`。

## 下一步研究问题（不是答案）
- 是否存在**可重复的环境**（资产级、而非样本级）使更强的非线性算子获得稳定价值？
- 若存在：下一阶段做 **capacity-matched MLP vs structured nonlinear operator** 的受控对比（NS 仅是候选之一）。

