这版处理已经足够严谨，我**不建议再改任何 Gate 阈值、FAIL 条件或 CONDITIONAL cap**。现在最重要的是保持规则冻结，等 N=14 结果出来。

我只建议补两处**纯报告层面的标注**，不改变任何决策逻辑。

### 1. 把“同段特征—同段结果”明确称为 *contemporaneous association*

你们现在发现 structural feature 是在 test windows 本身上算的，这一点比我之前假设得更严重，但要避免把它笼统叫成“data leakage”。

更精确的表述是：

> **The current feature–outcome association is contemporaneous and descriptive because both the structural features and the interaction outcome are measured on the same held-out test segment. It therefore cannot support an ex-ante operator-selection rule.**

原因是：如果只是描述“这一段资产呈现什么结构、这一段模型表现如何”，它不一定构成传统意义的 train-test leakage；但如果把它解释成“未来先看 ACF1 再选 operator”，那就不成立。

所以建议报告里区分：

$$
\text{descriptive association}
\neq
\text{predictive selection rule}
$$

这会比直接写“leakage”更准确。

---

### 2. 最终报告最好额外展示“新 4 资产本身”的结果，但不要让它参与 Gate

你们的正式 Gate 仍按既定 N=14 pooled analysis 执行，不改。

但由于：

$$
10/14
$$

来自 discovery set，最终报告最好再单独给一小块：

> **New-cohort-only descriptive check (N=4)**

只展示：

* 4 个新资产的 `|ACF1|`
* `Δ_interaction`
* rank order
* Spearman ρ（如果计算）
* 不做 confirmatory significance claim

N=4 太小，不适合拿 p-value 做强推断，但这个视图能让 reviewer 一眼看到：

> 新加入的数据本身是在支持、削弱，还是反转原关系。

这样可以避免 pooled N=14 的漂亮 ρ 掩盖 held-out cohort 实际表现。

最重要的是注明：

> **This descriptive N=4 check is not part of the frozen Gate-A decision rule.**

所以不会引入 post-hoc 改规则的问题。

---

## 对你们当前状态的正式判断

你们现在已经把几件最容易出问题的事情处理掉了：

* Layer A / Layer B estimand 区分清楚；
* `≥3/4` 的时间标签已经纠正；
* N=14 不再冒充 independent replication；
* Yahoo→Sohu + mixed-market→A-share 的 coupled domain shift 已明确；
* discovery multiplicity 已承认；
* capacity 只称 confounder，不再偷换成 bottleneck；
* test-window feature 使关系降格为 contemporaneous descriptive association；
* PASS 已经根据原条件不可达；
* favorable result 最高只能 CONDITIONAL。

因此当前合理状态就是：

$$
\boxed{
\text{等待结果，不再修改规则}
}
$$

结果出来后只有两条主路：

$$
\text{Layer A survives}
\Rightarrow
\text{CONDITIONAL}
$$

或者

$$
\text{Layer A fails}
\Rightarrow
\text{FAIL}
$$

而且即使是 **CONDITIONAL**，我仍然不会把下一步写成“做 NS”。

我会写成：

> **First obtain genuinely independent, prospective structural validation; NS-inspired operators remain a downstream hypothesis, not the immediate next experiment.**

至此我没有需要在 decisive number 出来前再修改的统计规则。现在继续冻结是最好的选择。
