# -*- coding: utf-8 -*-
"""
Exploratory Asset-Dependence Analysis

协议要求（降级为探索性）：把各资产的 interaction effect 符号与资产级结构特征做关系分析。
纪律：不为了得到显著相关性而修改特征、阈值或分组；发现稳定关系记为 candidate
asset-level structural regularity，符号随机则关闭该方向。

输入：
  - replication/results/volatility_stratification_<ASSET>_relative_s2021.json（各资产分层结果）
  - dataset/<ASSET>-2016-2025.csv（用于计算资产级结构特征）
输出：
  - reproduction/results/asset-dependence-exploration.md
  - reproduction/results/asset-dependence-table.csv
"""
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO, "reproduction", "results")
SEQ_LEN, PRED_LEN, JUMP_C = 96, 24, 3.0


def asset_features(csv_path):
    df = pd.read_csv(csv_path)
    close = df["Close"].to_numpy(float)
    n = len(df)
    border1 = n - int(n * 0.2) - SEQ_LEN
    ns = n - border1 - SEQ_LEN - PRED_LEN + 1
    rvs, acfs, jumps, signs = [], [], [], []
    for i in range(ns):
        seg = close[border1 + i:border1 + i + SEQ_LEN]
        r = np.diff(np.log(seg))
        sd = r.std()
        rvs.append(sd)
        if len(r) > 2 and r[:-1].std() > 1e-12 and r[1:].std() > 1e-12:
            acfs.append(abs(float(np.corrcoef(r[:-1], r[1:])[0, 1])))
        jumps.append(float((np.abs(r) > JUMP_C * sd).mean()) if sd > 1e-12 else 0.0)
        signs.append(float(abs(np.sign(r).sum()) / len(r)))
    return {
        "n_samples": ns,
        "rv_median": float(np.median(rvs)),
        "acf1_median_abs": float(np.median(acfs)) if acfs else np.nan,
        "jump_ratio_median": float(np.median(jumps)),
        "sign_persistence_median": float(np.median(signs)),
    }


def interaction_of(tag):
    p = os.path.join(REPO, "reproduction", "results", f"volatility_stratification_{tag}_relative_s2021.json")
    if not os.path.exists(p):
        return None
    R = json.load(open(p, encoding="utf-8"))
    H = R["horizons"]["T24"]
    c = H["comparison"]
    it = c.get("interaction_high50_minus_low50") or c.get("interaction")
    hi = c.get("high20%", {})
    lo = c.get("low20%", {})
    return {
        "interaction": None if not it else it["value"],
        "interaction_ci_lo": None if not it else it["ci95"][0],
        "interaction_ci_hi": None if not it else it["ci95"][1],
        "interaction_sig": None if not it else it["significant"],
        "high20_delta": hi.get("delta_mse"), "high20_win": hi.get("win_rate"),
        "high20_ci": hi.get("ci95"),
        "low20_delta": lo.get("delta_mse"), "low20_win": lo.get("win_rate"),
        "low20_ci": lo.get("ci95"),
        "n": H["n_samples"],
    }


def main():
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(REPO, "dataset", "*-2016-2025.csv"))):
        tag = os.path.basename(csv_path).split("-")[0]
        f = asset_features(csv_path)
        inter = interaction_of(tag)
        row = {"asset": tag, **f}
        if inter:
            row.update({k: v for k, v in inter.items()})
        else:
            row.update({"interaction": None, "interaction_sig": None, "high20_delta": None,
                        "high20_win": None, "low20_delta": None, "low20_win": None, "n": None})
        row["has_results"] = inter is not None
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "asset-dependence-table.csv"), index=False)
    # 协议 §5 要求的汇总表：只含已有分层结果的资产
    done_all = df[df.has_results & df.interaction.notna()].copy()
    summary_cols = ["asset", "interaction", "interaction_ci_lo", "interaction_ci_hi", "interaction_sig",
                    "high20_delta", "low20_delta", "high20_win", "n",
                    "rv_median", "acf1_median_abs", "jump_ratio_median", "sign_persistence_median"]
    done_all[summary_cols].to_csv(os.path.join(OUTDIR, "asset-dependence-summary.csv"), index=False)
    done = df[df.has_results & df.interaction.notna()].copy()

    lines = ["# Asset-Dependence Exploration（探索性）", ""]
    lines.append("> 协议定位：**Exploratory Asset-Dependence Analysis**，降级实验，不阻塞结构感知路由主线。")
    lines.append("> 目的：检查 interaction effect 的**符号**是否与资产级结构特征存在系统关系（而非验证 volatility 假设）。")
    lines.append(f"> 已完成分层结果的资产：{len(done)} 个")
    lines.append("")
    lines.append("## 1. 资产级表（结构特征 + 交互效应）")
    lines.append("")
    lines.append("| 资产 | 测试样本 | RV 中位 | \\|ACF1\\| 中位 | Jump 比例 | 符号持续性 | 交互效应 | 显著 | 最剧烈20% ΔMSE | 胜率 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.sort_values("asset").iterrows():
        if r.has_results and pd.notna(r.interaction):
            lines.append(f"| {r.asset} | {int(r.n)} | {r.rv_median:.5f} | {r.acf1_median_abs:.3f} | "
                         f"{r.jump_ratio_median:.4f} | {r.sign_persistence_median:.3f} | "
                         f"**{r.interaction:+.5f}** | {'是' if r.interaction_sig else '否'} | "
                         f"{r.high20_delta:+.5f} | {r.high20_win:.1%} |")
        else:
            lines.append(f"| {r.asset} | {int(r.n_samples)} | {r.rv_median:.5f} | {r.acf1_median_abs:.3f} | "
                         f"{r.jump_ratio_median:.4f} | {r.sign_persistence_median:.3f} | — | — | — | — |")
    lines.append("")

    if len(done) >= 3:
        lines.append("## 2. 交互效应 vs 资产结构特征（探索性相关）")
        lines.append("")
        lines.append("| 特征 | Pearson r | Spearman ρ | n | 说明 |")
        lines.append("|---|---|---|---|---|")
        feats = ["rv_median", "acf1_median_abs", "jump_ratio_median", "sign_persistence_median"]
        for f in feats:
            x, y = done[f].to_numpy(float), done["interaction"].to_numpy(float)
            if np.std(x) < 1e-12:
                lines.append(f"| {f} | — | — | {len(x)} | 资产间无变异 |")
                continue
            pr = float(np.corrcoef(x, y)[0, 1])
            sp = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
            lines.append(f"| {f} | {pr:+.3f} | {sp:+.3f} | {len(x)} | 样本极少，仅作探索 |")
        lines.append("")
        lines.append("**纪律声明**：特征、阈值、分组均按协议预先固定，未因结果调整；样本数 <5 时相关不具推断意义。")
        lines.append("")
        sign_pos = int((done["interaction"] > 0).sum())
        sign_neg = int((done["interaction"] < 0).sum())
        lines.append("## 3. 符号分布")
        lines.append("")
        lines.append(f"- 正号（偏好低波动侧/反号）：**{sign_pos}** 个；负号（非线性优势随波动上升）：**{sign_neg}** 个")
        if sign_pos == 0 or sign_neg == 0:
            lines.append("- 当前符号单一 → 与资产特征的关系需要更多资产才能判定。")
        else:
            lines.append("- 符号已跨零 → 可进入\"候选资产级规律\"的候选评估（需补种子）。")
        lines.append("")
        lines.append("## 4. 结论")
        lines.append("")
        lines.append("- 判定标准（协议）：若符号与资产特征存在**单调关系** → 记为 *candidate asset-level structural regularity*；"
                     "若符号**基本随机** → 关闭该方向。")
        lines.append("- 当前结论见上表；最终判定在 10 资产全部完成后给出。")
    else:
        lines.append("## 2. 结论")
        lines.append("")
        lines.append(f"**证据不足**：目前仅 {len(done)} 个资产有分层结果，不足以做符号—特征关系分析。")

    os.makedirs(OUTDIR, exist_ok=True)
    md = "\n".join(lines) + "\n"
    with open(os.path.join(OUTDIR, "asset-dependence-exploration.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print("[saved] asset-dependence-exploration.md / asset-dependence-table.csv")


if __name__ == "__main__":
    main()
