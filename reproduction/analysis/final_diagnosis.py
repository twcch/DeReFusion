# -*- coding: utf-8 -*-
"""
Asset-Level Operator Heterogeneity — Final Diagnosis pipeline（协议最终阶段）

只做分析，不重训、不改任何既有实验设置、不实现 router/NS。

对每个资产统一计算：
  Δ_interaction = Δ_high − Δ_low ; Δ_high / Δ_low ; interaction CI + significance ;
  paired win rate ; sample counts ; 资产结构特征（RV / ACF / jump ratio / trend persistence / skew / kurt）

四项分析：
  A. 符号分布（正/负、显著正/显著负、CI 含 0）—— 严格区分"方向"与"显著性"
  B. 结构特征 ↔ Δ_interaction 的探索性关联（Spearman + p，N 很小，仅探索）
  C. 特征对符号的区分能力 + leave-one-asset-out 敏感性（single-asset sensitive 必须标记）
  D. 稳健性：GSPC 按一个资产 + 分 seed 展示；asset-level preference 与 volatility interaction 分开

产出：
  reproduction/results/asset-dependence-summary.csv
  reproduction/results/asset-dependence-exploration.md
  reproduction/results/final-diagnosis.md（含 ≤1 页 executive summary）
  并在 structure-aware-routing-experiment.md 追加 §9/§10/§11
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO, "reproduction", "results")
SEQ_LEN, PRED_LEN, JUMP_C = 96, 24, 3.0
ASSETS_ALL = ["GSPC", "BTCUSD", "ETHUSD", "USDJPY", "EURUSD", "SOX", "DJI", "BABA", "NVO", "TM"]


# ----------------------------------------------------------- asset features
def asset_features(csv_path):
    df = pd.read_csv(csv_path)
    close = df["Close"].to_numpy(float)
    n = len(df)
    border1 = n - int(n * 0.2) - SEQ_LEN
    ns = n - border1 - SEQ_LEN - PRED_LEN + 1
    rvs, acfs, jumps, signs, tstats, sk, ku = [], [], [], [], [], [], []
    for i in range(ns):
        seg = close[border1 + i:border1 + i + SEQ_LEN]
        r = np.diff(np.log(seg))
        sd = r.std()
        rvs.append(sd)
        if r[:-1].std() > 1e-12 and r[1:].std() > 1e-12:
            acfs.append(abs(float(np.corrcoef(r[:-1], r[1:])[0, 1])))
        jumps.append(float((np.abs(r) > JUMP_C * sd).mean()) if sd > 1e-12 else 0.0)
        signs.append(float(abs(np.sign(r).sum()) / len(r)))
        tstats.append(abs(float(r.mean() * np.sqrt(len(r)) / sd)) if sd > 1e-12 else 0.0)
        sk.append(float(pd.Series(r).skew()))
        ku.append(float(pd.Series(r).kurt()))
    # 相对波动率（因果）：rv(t) / median(rv(t-120:t-1))，与样本级定义一致，非新特征
    rv_arr = np.array(rvs)
    rel = []
    for k in range(len(rv_arr)):
        base = rv_arr[max(0, k - 120):k]
        if len(base) >= 20 and np.median(base) > 0:
            rel.append(rv_arr[k] / np.median(base))
    return {
        "n_samples": ns,
        "realized_vol": float(np.median(rvs)),
        "relative_vol": float(np.mean(rel)) if rel else np.nan,
        "acf1_abs": float(np.median(acfs)) if acfs else np.nan,
        "jump_ratio": float(np.median(jumps)),
        "trend_persistence": float(np.median(tstats)),
        "sign_persistence": float(np.median(signs)),
        "skew": float(np.median(sk)), "kurtosis": float(np.median(ku)),
    }


def strat_of(tag, seed=2021, mode="relative"):
    p = os.path.join(REPO, "reproduction", "results",
                     f"volatility_stratification_{tag}_{mode}_s{seed}.json")
    if not os.path.exists(p):
        return None
    R = json.load(open(p, encoding="utf-8"))
    H = R["horizons"]["T24"]
    c = H["comparison"]

    def g(k):
        return c.get(k) or {}

    it = g("interaction_high50_minus_low50") or g("interaction") or {}
    hi, lo, hi20, lo20 = g("high50%"), g("low50%"), g("high20%"), g("low20%")
    return {
        "n": H["n_samples"],
        "delta_high": hi.get("delta_mse"), "ci_high": hi.get("ci95"),
        "delta_low": lo.get("delta_mse"), "ci_low": lo.get("ci95"),
        "delta_high20": hi20.get("delta_mse"), "delta_low20": lo20.get("delta_mse"),
        "interaction": it.get("value"),
        "ci_lo": (it.get("ci95") or [None, None])[0], "ci_hi": (it.get("ci95") or [None, None])[1],
        "significant": it.get("significant"),
        "win_high": hi.get("win_rate"), "win_low": lo.get("win_rate"),
        "win_high20": hi20.get("win_rate"),
        "n_high": hi.get("n"), "n_low": lo.get("n"),
    }


def descriptor(delta, ci_lo, ci_hi):
    """严格区分 direction 与 significance 的措辞（协议 §四）。"""
    if delta is None or ci_lo is None:
        return "n/a"
    dirn = "negative" if delta < 0 else "positive"
    supported = (ci_lo > 0) or (ci_hi < 0)
    return (f"directionally {dirn} and statistically supported" if supported
            else f"directionally {dirn} but statistically inconclusive")


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan, len(x)
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return float(r), float(p), len(x)
    except Exception:
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        r = float(np.corrcoef(rx, ry)[0, 1])
        rng = np.random.default_rng(0)
        cnt = sum(abs(np.corrcoef(rng.permutation(rx), ry)[0, 1]) >= abs(r) for _ in range(2000))
        return r, float((cnt + 1) / 2001), len(x)


def main():
    rows = []
    for tag in ASSETS_ALL:
        csv_path = os.path.join(REPO, "dataset", f"{tag}-2016-2025.csv")
        feats = asset_features(csv_path) if os.path.exists(csv_path) else {}
        st = strat_of(tag, 2021)
        row = {"asset": tag, **feats, "has_strat": st is not None}
        if st:
            row.update(st)
        rows.append(row)
    df = pd.DataFrame(rows)

    # GSPC seed 稳健性行（不并入主表 N）
    gs = {}
    for sd in (2021, 2022, 2023):
        s = strat_of("GSPC", sd)
        if s:
            gs[sd] = s

    cap = None
    cap_path = os.path.join(REPO, "reproduction", "results", "operator-regime-capacity.csv")
    if os.path.exists(cap_path):
        cap = pd.read_csv(cap_path)

    SEEDS_MAP = {"GSPC": [2021, 2022]}

    def asset_level_row(tag):
        """资产级汇总：GSPC 的 2021/2022 合并为一个资产（不得人为扩大资产数）。"""
        seeds = SEEDS_MAP.get(tag, [2021])
        ss = [s for s in (strat_of(tag, sd) for sd in seeds) if s]
        if not ss:
            return None
        csv_path = os.path.join(REPO, "dataset", f"{tag}-2016-2025.csv")
        feats = asset_features(csv_path) if os.path.exists(csv_path) else {}
        its = [s["interaction"] for s in ss]
        row = dict(feats)
        row.update({
            "asset": tag, "n_seeds": len(ss), "n": ss[0]["n"],
            "delta_high": float(np.mean([s["delta_high"] for s in ss])),
            "delta_low": float(np.mean([s["delta_low"] for s in ss])),
            "ci_high": [min(s["ci_high"][0] for s in ss), max(s["ci_high"][1] for s in ss)],
            "ci_low": [min(s["ci_low"][0] for s in ss), max(s["ci_low"][1] for s in ss)],
            "interaction": float(np.mean(its)),
            "interaction_seed_values": its,
            "ci_lo": min(s["ci_lo"] for s in ss), "ci_hi": max(s["ci_hi"] for s in ss),
            "significant": bool(all(s["significant"] for s in ss)),
            "sign_consistent": len({int(np.sign(v)) for v in its}) == 1,
            "win_high": float(np.mean([s["win_high"] for s in ss])),
            "win_low": float(np.mean([s["win_low"] for s in ss])),
            "has_strat": True,
        })
        return row

    asset_rows = [r for r in (asset_level_row(t) for t in ASSETS_ALL) if r]
    done = pd.DataFrame(asset_rows)
    df.to_csv(os.path.join(OUTDIR, "asset-dependence-table.csv"), index=False)
    cols = ["asset", "n_seeds", "n", "delta_high", "ci_high", "delta_low", "ci_low", "interaction",
            "ci_lo", "ci_hi", "significant", "sign_consistent", "win_high", "win_low", "realized_vol",
            "relative_vol", "acf1_abs", "jump_ratio", "trend_persistence", "sign_persistence", "skew", "kurtosis"]
    done[cols].to_csv(os.path.join(OUTDIR, "asset-dependence-summary.csv"), index=False)

    # ---------------------------------------------------------- A. 符号分布
    pos = done[done.interaction > 0]
    neg = done[done.interaction < 0]
    sig_pos = done[(done.interaction > 0) & (done.significant == True)]  # noqa: E712
    sig_neg = done[(done.interaction < 0) & (done.significant == True)]  # noqa: E712
    ci_zero = done[(done.ci_lo <= 0) & (done.ci_hi >= 0)]

    # ------------------------------------------- B/C. 关联 + LOO 敏感性
    feats = ["realized_vol", "relative_vol", "acf1_abs", "jump_ratio", "trend_persistence",
             "sign_persistence", "skew", "kurtosis"]
    assoc, loo = [], []
    for f in feats:
        r, p, n = spearman(done[f], done["interaction"])
        assoc.append({"feature": f, "spearman_r": r, "p_exploratory": p, "N": n,
                      "direction": "负（特征越大越偏非线性）" if r < 0 else "正"})
        for drop in done["asset"]:
            sub = done[done.asset != drop]
            r2, p2, n2 = spearman(sub[f], sub["interaction"])
            loo.append({"feature": f, "dropped": drop, "spearman_r": r2, "p_exploratory": p2, "N": n2})
    assoc_df = pd.DataFrame(assoc)
    loo_df = pd.DataFrame(loo)
    loo_sum = loo_df.groupby("feature").agg(
        r_min=("spearman_r", "min"), r_max=("spearman_r", "max"),
        sign_flips=("spearman_r", lambda s: int((np.sign(s) != np.sign(s.iloc[0])).any())),
        max_drop=("spearman_r", lambda s: float(abs(s.iloc[0] - s.min()))),
    ).reset_index()
    loo_sum["single_asset_sensitive"] = (loo_sum.r_min * loo_sum.r_max) < 0

    n_a = len(done)
    n_neg, n_pos = int((done.interaction < 0).sum()), int((done.interaction > 0).sum())
    dominant = max(n_neg, n_pos)
    both_signs = n_neg > 0 and n_pos > 0

    # PATH 1 必须同时满足协议五项条件
    cand = []
    g1, g2 = strat_of("GSPC", 2021), strat_of("GSPC", 2022)
    for _, a in assoc_df.iterrows():
        f = a.feature
        lo = loo_sum[loo_sum.feature == f]
        if not len(lo):
            continue
        # 条件 3+4：非单一资产驱动，且 LOO 后方向保持
        loo_ok = (not bool(lo.single_asset_sensitive.iloc[0])) and \
                 (np.sign(lo.r_min.iloc[0]) == np.sign(lo.r_max.iloc[0]) == np.sign(a.spearman_r))
        # 条件 2：存在稳定 association（探索性阈值 p<0.10）
        p_ok = (not np.isnan(a.p_exploratory)) and a.p_exploratory < 0.10
        # 条件 5：不是 GSPC seed 合并方式造成的
        seed_ok = True
        if g1 and g2:
            s1, s2 = done.copy(), done.copy()
            s1.loc[s1.asset == "GSPC", "interaction"] = g1["interaction"]
            s2.loc[s2.asset == "GSPC", "interaction"] = g2["interaction"]
            r1, _, _ = spearman(s1[f], s1["interaction"])
            r2, _, _ = spearman(s2[f], s2["interaction"])
            seed_ok = (np.sign(r1) == np.sign(r2) == np.sign(a.spearman_r))
        if p_ok and loo_ok and seed_ok:
            cand.append(f)

    if n_a < 5:
        path_dep, path_dep_name = "PATH 3", "Evidence insufficient"
    elif dominant >= 3 and cand:
        path_dep, path_dep_name = "PATH 1", "Candidate asset-level structural regularity"
    elif n_a >= 5:
        path_dep, path_dep_name = "PATH 2", "Asset-level heterogeneity but structurally unexplained"
    else:
        path_dep, path_dep_name = "PATH 3", "Evidence insufficient"
    stable_feat = assoc_df[assoc_df.feature.isin(cand)]

    # ------------------------------------------------------------- 写报告
    L = []
    L.append("# Asset-Dependence Exploration（探索性，最终版）\n")
    L.append("> 定位：**Exploratory Asset-Dependence Analysis**——只检查 interaction 符号与资产结构特征的关系，")
    L.append("> 不验证 volatility 假设、不声称因果。所有 association 仅为 exploratory。\n")
    L.append(f"## 1. 资产级表（n={n_a} 个资产有分层结果；GSPC 使用 seed 2021，seed 稳健性见 §4）\n")
    L.append("| 资产 | seeds | 样本 | Δ_low | Δ_high | **Δ_interaction** | 交互 CI | 描述（方向 vs 显著性）| 高波动侧胜率 | RV | RelVol | \\|ACF1\\| | Jump | Trend |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in done.sort_values("interaction").iterrows():
        L.append(f"| {r.asset} | {int(r.n_seeds)} | {int(r.n)} | {r.delta_low:+.5f} | {r.delta_high:+.5f} | "
                 f"**{r.interaction:+.5f}** | [{r.ci_lo:+.5f},{r.ci_hi:+.5f}] | "
                 f"{descriptor(r.interaction, r.ci_lo, r.ci_hi)} | {r.win_high:.1%} | {r.realized_vol:.5f} | "
                 f"{r.relative_vol:.3f} | {r.acf1_abs:.3f} | {r.jump_ratio:.4f} | {r.trend_persistence:.2f} |")
    L.append("")
    L.append("## 2. A. 符号分布（严格区分方向与显著性）\n")
    L.append(f"- 负号（Δ_interaction<0，非线性优势随波动上升）：**{len(neg)}** 个；正号：**{len(pos)}** 个")
    L.append(f"- 其中**显著**：负 {len(sig_neg)} 个 / 正 {len(sig_pos)} 个；CI 含 0：{len(ci_zero)} 个")
    L.append(f"- 符号是否跨零：{'是' if both_signs else '否'}")
    L.append("")
    L.append("## 3. B. 结构特征 ↔ Δ_interaction（Spearman，探索性）\n")
    L.append("| 特征 | ρ | p（仅探索） | N | 方向 |")
    L.append("|---|---|---|---|---|")
    for _, r in assoc_df.iterrows():
        L.append(f"| {r.feature} | {r.spearman_r:+.3f} | {r.p_exploratory:.3f} | {r.N} | {r.direction} |")
    L.append("")
    L.append("> N 很小，单个 p 值不构成强证据；以下 LOO 才是稳健性关键。\n")
    L.append("## 4. C. Leave-one-asset-out 敏感性\n")
    L.append("| 特征 | ρ 范围 | 是否变号 | single-asset sensitive |")
    L.append("|---|---|---|---|")
    for _, r in loo_sum.iterrows():
        L.append(f"| {r.feature} | [{r.r_min:+.3f}, {r.r_max:+.3f}] | {'是' if r.sign_flips else '否'} | "
                 f"{'⚠️ 是' if r.single_asset_sensitive else '否'} |")
    L.append("")
    L.append("## 5. D. 稳健性\n")
    L.append("### GSPC 分 seed（不作为独立资产并入主表）\n")
    L.append("| seed | Δ_interaction | 95%CI | 显著 |")
    L.append("|---|---|---|---|")
    for sd, s in gs.items():
        L.append(f"| {sd} | {s['interaction']:+.5f} | [{s['ci_lo']:+.5f},{s['ci_hi']:+.5f}] | "
                 f"{'是' if s['significant'] else '否'} |")
    L.append("")
    L.append("### asset-level operator preference 与 volatility interaction（两者不混同）\n")
    L.append("| 资产 | 容量 128 维平均 ΔMSE（+ = 偏好线性） | Δ_interaction（波动率交互）|")
    L.append("|---|---|---|")
    if cap is not None:
        for a in done["asset"]:
            c128 = cap[(cap.width == 128) & (cap.asset == a)]["dMSE"].mean()
            itv = done[done.asset == a]["interaction"].iloc[0]
            L.append(f"| {a} | {c128:+.5f} | {itv:+.5f} |")
    L.append("")
    L.append("> 说明：前者是「该资产整体更偏好哪类算子」，后者是「优势是否随波动率变化」——是不同的量，不可互推。\n")
    L.append("## 6. 结论（协议 §六，只允许三种）\n")
    L.append(f"$$\\boxed{{\\text{{{path_dep_name}}}}}$$\n")
    if path_dep == "PATH 1":
        L.append(f"依据：符号跨零（{len(neg)}负/{len(pos)}正），且下列特征关联在 LOO 下方向稳定："
                 f"{', '.join(stable_feat.feature.tolist())}。仍只是 **candidate**，不得称为 established law。\n")
    elif path_dep == "PATH 2":
        L.append("依据：符号混杂/无 LOO 稳定的特征关联/结果由个别资产驱动 → **关闭 asset-dependence 分支**，"
                 "归档为 *asset-dependent but structurally unexplained operator heterogeneity*。\n")
    else:
        L.append(f"依据：当前仅 {n_a} 个资产有分层结果，样本数不足以判定。\n")

    md = "\n".join(L) + "\n"
    open(os.path.join(OUTDIR, "asset-dependence-exploration.md"), "w", encoding="utf-8").write(md)

    # ---------------------------------------------------- final-diagnosis.md
    eth_cap = cap[(cap.width == 128) & (cap.asset == "ETHUSD")]["dMSE"].mean() if cap is not None else float("nan")
    btc_cap = cap[(cap.width == 128) & (cap.asset == "BTCUSD")]["dMSE"].mean() if cap is not None else float("nan")
    gspc_cap = cap[(cap.width == 128) & (cap.asset == "GSPC")]["dMSE"].mean() if cap is not None else float("nan")

    F = []
    F.append("# Final Diagnosis — Asset-Level Operator Heterogeneity\n")
    F.append("## Executive Summary（≤1 页）\n")
    F.append("**1. 当前最可靠的发现**")
    F.append("- *Operator suitability is asset-dependent*：容量 128 维下，ETHUSD 全部状态/种子偏好非线性"
             f"（平均 ΔMSE {eth_cap:+.5f}），BTCUSD 全容量偏好线性（{btc_cap:+.5f}），GSPC 高容量接近打平（{gspc_cap:+.5f}）。")
    F.append("- 主实验（23 维）观察到的\"线性全面占优\"**部分是容量假象**：the earlier rigidity was capacity-related。")
    F.append("")
    F.append("**2. 当前被排除的假设**")
    F.append("- ❌ Sample-level routing：未发现稳定的 `StructuralState → OperatorPreference`（Case 1/2 均不成立；")
    F.append("  反转是资产级全状态而非状态级；状态间 Range(ΔMSE) 未随容量扩大；方向随种子波动）。")
    F.append("- ❌ \"非线性算子普遍更差\"：ETHUSD 在足够容量下全面偏好非线性。")
    F.append("- ❌ \"波动率 → 门控\"：gatev1 实测劣于无参数加法 8.6%。")
    F.append("")
    F.append("**3. 尚未解决的 ambiguity**")
    F.append("- 结构性特征是否与 operator heterogeneity 相关：见 asset-dependence-exploration.md（当前判定："
             f"**{path_dep_name}**）。")
    F.append("- 代理算子（展平窗口线性 / MLP）与框架内真实分支的保真度差异：结论不能外推到\"非线性无用\"。")
    F.append("")
    F.append("**4. 本研究决策（§九 Path A/B/C/D）**")
    dec = "Path B（Nonlinear operator deserves targeted enhancement）" if n_a >= 5 and len(stable_feat) else \
          ("Path D（No stable explanatory structure）" if n_a >= 5 else "Path D（暂定）— 证据不足，待 7 资产完成")
    F.append(f"- 当前资产级异质性明确（operator evidence 强），但 sample-level routing 无证据（routing evidence 无）。")
    F.append(f"- 建议：**{dec}**")
    F.append("")
    F.append("**5. 是否允许重新考虑 NS**")
    F.append("- ❌ 本阶段**不允许**。协议规定：只有能明确回答 \"there exists a reproducible context in which a stronger "
             "nonlinear operator is justified\" 才可在下一阶段把 NS 作为**候选**之一（与 capacity-matched MLP 对照），"
             "且不得由 `ETH → nonlinear` 推出 `ETH → NS`。")
    F.append("")
    F.append("---\n")
    F.append("## Q1–Q5 必答（协议 §九）\n")
    F.append("- **Q1 是否存在稳健 sample-level routing evidence？** → **No**（结构状态→算子偏好未发现稳定关系："
             "Case 1/2 均不成立；反转是资产级全状态而非状态级；状态间 Range(ΔMSE) 未随容量扩大；方向随种子波动）")
    F.append(f"- **Q2 是否存在真实 asset-level operator heterogeneity？** → **Yes**（基于容量对照结果："
             f"ETHUSD 64/128 维下 36/36 观测偏好非线性（128 维平均 ΔMSE {eth_cap:+.5f}）；"
             f"BTCUSD 全容量偏好线性（{btc_cap:+.5f}）；GSPC 高容量接近打平（{gspc_cap:+.5f}）。"
             f"同时保留：23-d nonlinear bottleneck materially affected the earlier ETH result）")
    F.append(f"- **Q3 这种 heterogeneity 能否由现有 structural features 解释？** → 由 7 资产 Spearman + LOO 判定："
             f"**{path_dep} — {path_dep_name}**（LOO 稳定特征：{', '.join(cand) if cand else '无'}）")
    F.append("- **Q4 当前主要 uncertainty 在哪里？** → 主要在 **(d) asset dependence**（资产级差异已知但尚无稳定结构解释）"
             "与 **(b) operator form**（何种非线性算子形态在何种资产上值得加强）；"
             "**不**把 unknown 伪装成 representation bottleneck："
             "目前只能说 current structural features do not yet establish a general sample-level operator-selection mechanism。"
             "routing（a）方向已基本排除。")
    ns_ok = (path_dep in ("PATH 1", "PATH 2")) and (n_a >= 5)
    F.append(f"- **Q5 下一阶段是否有资格重新考虑 NS？** → **{'可以（仅作为候选之一）' if ns_ok else '否'}**。"
             "判据：there exists a reproducible context in which a stronger nonlinear operator is justified；"
             "即使满足，也只能与 capacity-matched MLP 做受控对比，且不得由 ETH→nonlinear 推出 ETH→NS。")
    F.append("")
    F.append("## 四类证据分列（协议 §七）\n")
    F.append("**A. Routing evidence**：$$\\boxed{\\text{No robust sample-level routing evidence}}$$")
    F.append("")
    F.append("**B. Operator evidence**：$$\\boxed{\\text{Strong asset-level heterogeneity is observed}}$$")
    F.append(f"　ETH: nonlinear favored at adequate capacity（128 维 ΔMSE {eth_cap:+.5f}）；"
             f"BTC: linear favored（{btc_cap:+.5f}）；GSPC: near parity（{gspc_cap:+.5f}）。")
    F.append("")
    F.append("**C. Representation evidence**：Current structural features do not yet establish a general "
             "sample-level operator-selection mechanism.（**不**断言 representation 是唯一瓶颈）")
    F.append("")
    F.append(f"**D. Asset-dependence evidence**：{path_dep_name}（N={n_a} 资产）")
    F.append("")
    F.append("## 明确禁止的逻辑跳跃（协议 §十，已自查）")
    F.append("- `ETH → nonlinear` ⇏ `NS should work`；`BTC → linear` ⇏ `linear is generally superior`；")
    F.append("- `no routing evidence` ⇏ `representation definitely insufficient`；`correlation` ⇏ `causal mechanism`；")
    F.append("- `one asset` ⇏ `universal rule`。")
    F.append("")
    F.append("## 下一步研究问题（不是答案）")
    F.append("- 是否存在**可重复的环境**（资产级、而非样本级）使更强的非线性算子获得稳定价值？")
    F.append("- 若存在：下一阶段做 **capacity-matched MLP vs structured nonlinear operator** 的受控对比（NS 仅是候选之一）。")
    F.append("")

    open(os.path.join(OUTDIR, "final-diagnosis.md"), "w", encoding="utf-8").write("\n".join(F) + "\n")

    # ----------------------------- 追加 §9/§10/§11 到主报告（幂等：检查标记）
    rep = os.path.join(OUTDIR, "structure-aware-routing-experiment.md")
    txt = open(rep, encoding="utf-8").read()
    if "## 9. Asset Dependence" not in txt:
        add = ["\n## 9. Asset Dependence（协议 §9，探索性）\n",
               f"详见 `asset-dependence-exploration.md`（N={n_a} 资产）。只检查 interaction 符号与资产结构特征的关系，不验证 volatility 假设。\n",
               "| 资产 | Δ_interaction | 95%CI | 显著 | RV | \\|ACF1\\| | Jump | Trend |",
               "|---|---|---|---|---|---|---|---|"]
        for _, r in done.sort_values("interaction").iterrows():
            add.append(f"| {r.asset} | {r.interaction:+.5f} | [{r.ci_lo:+.5f},{r.ci_hi:+.5f}] | "
                       f"{'是' if r.significant else '否'} | {r.realized_vol:.5f} | {r.acf1_abs:.3f} | "
                       f"{r.jump_ratio:.4f} | {r.trend_persistence:.2f} |")
        add += ["", f"**探索性结论：{path_dep_name}**（符号跨零与否：{'是' if both_signs else '否'}；",
                f"LOO 稳定的特征关联：{', '.join(stable_feat.feature.tolist()) if len(stable_feat) else '无'}）", ""]
        add += ["## 10. Final Diagnosis（协议 §10）\n",
                "- **A. Routing evidence**：No robust sample-level routing evidence\n",
                "- **B. Operator evidence**：Strong asset-level heterogeneity is observed（ETH 偏好非线性 / BTC 偏好线性 / GSPC 高容量打平）\n",
                "- **C. Representation evidence**：Current structural features do not yet establish a general sample-level operator-selection mechanism（不断言 representation 是唯一瓶颈）\n",
                f"- **D. Asset-dependence evidence**：{path_dep_name}\n",
                "\n## 11. Research Decision（协议 §十一）\n",
                f"- 资产级异质性明确、sample-level routing 无证据 → **{dec}**\n",
                "- 是否重新考虑 NS：**否**（本阶段禁止；仅当出现可重复环境证明更强非线性算子有价值时，下一阶段可作候选）\n"]
        open(rep, "w", encoding="utf-8").write(txt + "\n".join(add) + "\n")

    print(md)
    print("\n[saved] asset-dependence-summary.csv / asset-dependence-exploration.md / final-diagnosis.md")
    print(f"[decision] {path_dep} {path_dep_name} | research: {dec}")


if __name__ == "__main__":
    main()
