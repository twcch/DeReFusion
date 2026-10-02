我按这份文档要求，把自己当成**在 N=14 decisive number 出来之前的外部 reviewer**来判断。我的结论比较明确：**Section 5 的主次层级我接受，但当前 Gate A 仍不能被称为真正的“独立外部验证”；如果最后是 CONDITIONAL，也不应该直接进入 NS benchmark。**

## 1. 我接受 “Layer A primary，Layer B derived consequence”

这个区分在统计对象上是正确的。

Layer A 检验的是：

$$
\rho\left(|ACF1|,\Delta_{\text{interaction}}\right)
$$

而你们最初发现的 regularity 本来就是定义在 `Δ_interaction` 上。Layer B 检验的却是 proxy operator 的整体 `ΔMSE` 偏好，两者不是同一个 estimand。你们自己已经把这一点写得很清楚。

因此 YANGHE 现在的反向结果：

$$
|ACF1|\text{ highest}
\quad\text{but}\quad
\Delta MSE<0
$$

**不能逻辑上直接否定 Layer A 的 interaction-level association。**

但它可以否定另一个更强的说法：

> “既然高 \(|ACF1|\) 对应更弱的 high-vol nonlinear advantage，那么高 \(|ACF1|\) 资产整体应该更偏好 linear operator。”

这个推论现在没有得到支持。

所以最终报告最准确的表达应该是：

> **The external capacity-layer test does not directly falsify the interaction-level association, but it fails to support the stronger extrapolation from interaction structure to unconditional operator preference.**

也就是说：

$$
\text{Layer B failure}
\not\Rightarrow
\text{Layer A false}
$$

但是：

$$
\text{Layer B failure}
\Rightarrow
\text{do not use Layer A as evidence for a per-asset operator-selection rule}
$$

这一点很重要。

---

# 2. 但是我发现了一个需要你们明确写进审计记录的问题

你们的 exact `≥3/4` Layer-B FAIL threshold **似乎不是在看到 Layer B 结果之前锁定的**。

文档时间线显示，capacity layer 已经完成，而 `Gate A decision-rule addendum` 是之后、在 framework interaction result 尚未出来时锁定的。

这意味着：

> 它是 **pre-Layer-A** 的规则，但未必是 **pre-Layer-B** 的规则。

这两件事不能混为一谈。

因此我建议**不要再修改这个 ≥3/4 阈值**，否则问题更严重；但最终论文/报告里不要把它称为：

> fully pre-registered Layer-B falsification threshold

更准确的叫法应该类似：

> **pre-Layer-A adjudication rule, specified after the secondary capacity-layer results were available**

这不会毁掉实验，但必须透明。

你们真正提前锁定、而且最干净的是：

> 新资产怎么选、预测方向是什么。

这些是在 operator 结果出现前锁定的。

---

# 3. 对现在这几个阈值，我不建议再改

当前 FAIL 条件里有：

$$
\rho<0.30
$$

或者：

$$
p>0.10
$$

或者 LOO crossing zero 等。

从纯统计设计上讲，我不会特别喜欢 `ρ=0.30` 和 `p=0.10` 这种硬切点，因为它们本身有一定人为性，而且在 N≈14 时 `p<0.10` 实际上已经要求一个不算小的相关系数，所以 `ρ>0.30` 的额外信息有限。

**但现在不要改。**

原因不是因为这些阈值完美，而是：

> 现在修改阈值带来的研究诚信成本，大于理论上的统计优化收益。

所以我的 reviewer verdict 是：

**保留当前 thresholds，机械执行。**

但把它们称为：

> **operational gate criteria**

而不是：

> universal statistical significance standards.

---

# 4. 一个比 threshold 更大的问题：N=14 其实不是真正的 independent validation

这是我认为你们现在最容易高估证据强度的地方。

原来的：

$$
N=10
$$

已经被用来发现：

$$
|ACF1|,\ RV
$$

这两个 candidate。

现在把 4 个新资产加进去以后重新算：

$$
N=14
$$

其中仍然有那原来的 10 个 discovery observations。

所以：

$$
\rho_{N=14}
$$

不是一个真正独立的 confirmatory statistic。

它更准确地属于：

> **discovery-set augmentation / external stress test**

而不是：

> independent replication.

也就是说，即使最后：

$$
\rho=0.70,\quad p=0.01
$$

也不能因此说：

> “ACF1 regularity 已经从 candidate 升级为 supported law。”

因为 10/14 的数据参与过 hypothesis generation。

这一点甚至比 p=0.05 还是 0.10 更重要。

所以我认为当前 Gate A 的最高合理解释仍然是：

> **candidate survived external stress testing**

而不是：

> confirmed structural relationship.

---

# 5. Layer B 现在到底有多严重？

目前四个新资产：

| Asset     | 预期        | 实际        |
| --------- | --------- | --------- |
| BYD       | nonlinear | 符合        |
| BOE       | nonlinear | mixed/tie |
| EASTMONEY | linear    | mixed/tie |
| YANGHE    | linear    | 稳定反向      |

这个结果本身很有价值。

它告诉你们：

> **ACF1 至少不是一个可以直接转化成“每个资产该选 linear 还是 nonlinear”规则的单变量指标。**

尤其是 YANGHE 两个 capacity、各 seeds 都稳定朝反方向，这比单 seed noise 更值得重视。

所以即使 Layer A 最终保持漂亮，我也支持你们现在写下的：

> withdraw the operator-preference implication

这个处理。

---

# 6. 什么样的实验才能真正从 CANDIDATE → SUPPORTED？

我会要求一个**真正独立的 held-out validation cohort**。

不是继续：

$$
10\rightarrow14\rightarrow18\rightarrow22
$$

然后每次重新计算 pooled correlation。

那样 discovery data 永远留在检验里。

更合理的是：

$$
D_{\text{discovery}}
$$

和

$$
D_{\text{validation}}
$$

彻底分离。

我认为“最低限度可信”的下一次设计是：

> **预先固定 \(|ACF1|\) 为唯一 primary structural predictor，realized volatility 作为 secondary；再找一批从未参与 hypothesis formation 的资产，至少覆盖多个 \(|ACF1|\) 区间，而不是只取 extreme；每资产 framework-level 至少 3 seeds；所有 operator、split、T=24、RV definition 完全固定；primary analysis 只在新 cohort 上完成。**

资产数量上，如果算力允许，我会希望：

$$
N_{\text{validation}}\approx15\text{–}20
$$

而不是再加 4 个。

因为你真正推断的单位是 **asset**，不是 seed。

更多 seed 可以降低：

$$
\hat{\Delta}_{interaction,i}
$$

的测量噪声，但不会增加 cross-sectional sample size。

所以资源有限时，我会排：

$$
\boxed{
\text{more independent assets}
>
\text{seed 4/5 on existing assets}
}
$$

前提是每资产已经至少有 3 seeds。

---

# 7. Multiplicity 应该怎么处理

这里也有一个隐性问题。

最初你们不是只看了 `|ACF1|`，而是看了多个 structural features；最终 `|ACF1|` 和 realized volatility 从中脱颖而出。文档中仍列着至少 8 类 candidate feature。

所以 discovery-stage 的：

$$
p=0.016
$$

不能被当作完全没有 feature-selection multiplicity 的 confirmatory p-value。

下一轮最好这样处理：

**Primary hypothesis：**

$$
H_1:
|ACF1|\leftrightarrow\Delta_{\text{interaction}}
$$

只保留一个。

realized volatility：

> secondary / supportive analysis

这样最干净。

如果坚持两个都作为 primary，则用例如 Holm correction 控制 family-wise error。

不要在下一轮又重新扫描：

* kurtosis
* skew
* Hurst
* entropy
* tail index
* 20 个新 feature

然后挑最好的那个继续讲。

那会重新回到 hypothesis fishing。

---

# 8. Provenance deviation：我会给它“高权重”，但不是判死刑

现在是：

$$
\text{Yahoo + mixed assets}
$$

变成：

$$
\text{Sohu + A-shares}
$$

而且这两个变化是同时发生的。

因此如果新资产失败，你无法区分：

$$
\text{market-domain effect}
$$

还是：

$$
\text{data-provider effect}
$$

还是：

$$
\text{instrument/microstructure effect}.
$$

同样，如果成功，也不能简单说：

> “cross-market universal.”

我会把现在这四只 A 股称为：

> **domain-shift stress-test cohort**

而不是严格的 independent replication cohort。

还有一个更细的问题：

4 个 A 股并不是 4 个完全独立的自然实验单位。

它们共享：

* 同一个市场制度
* 相近交易日历
* 同一宏观环境
* 同一数据供应商

因此 cross-sectional independence 比“14 个完全独立资产”弱。

这也是为什么我不建议因为：

$$
N=14
$$

就觉得样本已经很大。

---

# 9. 还有一个你们需要非常小心的问题：full-history ACF1

文档写的是根据 **full-history candidate features** 选择 extreme assets。

如果你们只是做：

> retrospective structural description

问题不算严重。

但如果最后想把它表述成：

> “以后看到一个新资产，可以先算 ACF1，然后决定用哪个 operator”

那么这个 feature 必须只用：

$$
\text{training period}
$$

或者 decision time 之前的数据计算。

否则会存在一个概念上的 prospective leakage：

你用未来测试期数据参与构造了决定 operator 的资产描述。

所以现在最多可以说：

> asset-level descriptive association.

暂时不要说：

> prospective operator-selection rule.

---

# 10. Capacity 结果现在的表述已经基本正确，但有一句建议改

目前结果是：

* ETH：23 → 64/128 后大规模反转
* BTC：始终 linear
* GSPC：随着 capacity 增大走向 parity。

所以最稳妥的结论是：

> **Operator preference is not invariant to nonlinear model capacity.**

以及：

> **The low-capacity conclusion does not survive increased nonlinear capacity on ETHUSD.**

我会尽量避免：

> “low-width result was a representation bottleneck”

甚至谨慎使用：

> “capacity artefact”

因为 width 同时可能改变：

* representation power
* optimization landscape
* regularization
* variance
* implicit bias

你们现在只能确定：

$$
\text{result depends materially on capacity}
$$

还不能确定：

$$
\text{mechanism}=\text{representation bottleneck}.
$$

如果以后真想证明 bottleneck，需要看到：

* training error / validation error 随 width 的变化；
* convergence 是否充分；
* 多 seed learning curves；
* 其他增加 capacity 的方式是否得到类似结果；
* capacity increase 是否出现稳定 plateau。

所以你们目前的 caution 是对的。

---

# 11. 如果 Gate A = CONDITIONAL，我建议下一题是什么

不是 NS。

而是：

$$
\boxed{
\text{Does the interaction-level structural association replicate
in a genuinely independent, multi-seed asset cohort?}
}
$$

也就是说：

> **这个结构规律到底是不是 portable？**

因为 CONDITIONAL 意味着：

* association 还活着；
* 但 operator-preference extrapolation 已经失败；
* 证据还不足以把它当作结构机制。

这时候直接去 NS，中间还缺一层。

---

# 12. 如果 Gate A = FAIL，我建议的问题更基础

如果 FAIL，我甚至不会马上寻找下一个 ACF2、entropy、Hurst exponent。

先问：

$$
\boxed{
\text{Is asset-level operator preference itself a stable,
reproducible property across seeds, capacities and periods?}
}
$$

这是因为当前真正 capacity-controlled 的强 heterogeneity 证据主要还是 ETH/BTC/GSPC 三个资产。文档自己也标明 coverage = 3 assets。

先证明：

$$
\text{Asset A consistently prefers nonlinear}
$$

$$
\text{Asset B consistently prefers linear}
$$

而不是因为：

* seed
* period
* capacity
* split

才出现。

如果这个性质稳定，再问：

> **什么结构解释 operator preference？**

否则连被解释对象都还不够稳定。

---

# 13. “operator families per asset” 是一个合法的 reframing

如果 ACF1/RV 最后失败，这条研究线并不需要一起死掉。

你完全可以把问题改成：

> **Under capacity and compute control, do different asset classes exhibit reproducible preferences for different operator families?**

这里先不提 NS。

候选可以是非常普通的：

$$
\text{Linear}
$$

vs

$$
\text{MLP}
$$

vs 其他预先指定 generic nonlinear family。

只有之后发现：

> 某种局部平滑、传输、扩散、memory-like 结构稳定出现

再引入对应 structured operator。

这样就不会：

$$
\text{先相信 NS}
\rightarrow
\text{再寻找理由}
$$

而是：

$$
\text{observe operator structure}
\rightarrow
\text{derive model hypothesis}.
$$

---

# 最终 reviewer verdict

如果让我现在在 **N=14 出来之前**签字，我会写：

> **I accept the Layer-A-primary / Layer-B-secondary distinction. Layer B currently falsifies the stronger operator-preference extrapolation but does not directly falsify the original interaction-level association. I recommend leaving the current Gate-A thresholds unchanged for auditability, while explicitly labelling the ≥3/4 Layer-B rule as a pre-Layer-A adjudication rule rather than a fully preregistered Layer-B threshold if it was specified after capacity results were observed.**

然后我会再加一句更关键的：

> **Even a favourable N=14 outcome should not upgrade the relation from CANDIDATE to SUPPORTED, because the N=14 statistic reuses the N=10 discovery set and the four new observations introduce a coupled market-and-provider domain shift. A genuinely independent, same-protocol, multi-seed validation cohort is still required.**

因此你们现在这条规则里：

> **PASS unreachable**

我赞成。

而我的实际决策是：

$$
\boxed{
\text{最好的可能结果现在也只是 CONDITIONAL}
}
$$

如果 Layer A 失败：

$$
\boxed{\text{FAIL}}
$$

如果 Layer A 保持：

$$
\boxed{\text{CONDITIONAL}}
$$

**不会因为 N=14 很漂亮就直接升级到 NS benchmark。**

下一步先解决“独立 replication”或者“asset-level operator preference 是否本身稳定”这两个更基础的问题。
