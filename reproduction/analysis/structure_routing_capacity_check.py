# -*- coding: utf-8 -*-
"""
容量敏感性实验（按协议 §3 要求）

唯一变量：nonlinear MLP 的 hidden dimension ∈ {23, 64, 128}
其余全部与主实验一致：数据、结构特征、train/val/test 切分、optimizer、learning rate、
batch size、训练步数、早停、随机种子、预处理、评估协议。

对每个 (d, asset, seed, structural state) 输出：
  MSE, MSPE, P95 error, paired win rate, ΔMSE = MSE_N − MSE_L, bootstrap 95% CI (4000)

三项重点检查：
  (1) 符号是否反转（ΔMSE_23 > 0 而 ΔMSE_128 < 0 等）
  (2) 状态间 Range(ΔMSE) 是否随容量扩大
  (3) 逐 seed 方向一致性（不只报均值）

注：linear 算子与宽度无关，同一 (asset, seed) 只训练一次并复用（同种子同数据 → 结果确定一致）。

输出：
  reproduction/results/operator-regime-capacity.csv       逐行完整指标
  reproduction/results/operator-regime-capacity-summary.csv  汇总
  reproduction/results/capacity-sign-reversal.md      符号反转与范围/一致性检查
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import structure_routing_experiment as S  # noqa: E402

OUTDIR = os.path.join(S.REPO, "reproduction", "results")
WIDTHS = [23, 64, 128]

# Optional asset override: --assets BYD,BOE,EASTMONEY,YANGHE (structural-validation extension).
# Only the ASSET SET changes; operators, states, protocol and metrics are untouched.
if "--assets" in sys.argv:
    S.ASSETS = sys.argv[sys.argv.index("--assets") + 1].split(",")
    SUFFIX = "_sv"          # structural-validation extension: never overwrite the E4 outputs
else:
    SUFFIX = ""
if "--suffix" in sys.argv:  # explicit output suffix (e.g. F1): OUTPUT NAMING ONLY, no definition change
    SUFFIX = sys.argv[sys.argv.index("--suffix") + 1]
if "--widths" in sys.argv:
    WIDTHS = [int(w) for w in sys.argv[sys.argv.index("--widths") + 1].split(",")]
print(f"[cfg] assets={S.ASSETS} widths={WIDTHS} suffix={SUFFIX!r}", flush=True)


def train_linear_once(tag, seed, data):
    x_tr, y_tr, x_va, y_va, x_te, y_te, mu, sd = data
    model, p, _ = S.train_operator("linear", x_tr, y_tr, x_va, y_va, seed)
    with torch.no_grad():
        pred = model(torch.tensor(x_te)).numpy()
    se = ((pred - y_te) ** 2).mean(1)
    price = pred * sd[0, -1] + mu[0, -1]
    ptrue = y_te * sd[0, -1] + mu[0, -1]
    return se, price, ptrue, p


def train_mlp_width(tag, seed, data, width):
    class MLPw(S.MLPOp):
        def __init__(self, d_in, d_out, hidden=width):
            super().__init__(d_in, d_out, hidden=hidden)

    S.MLPOp = MLPw
    x_tr, y_tr, x_va, y_va, x_te, y_te, mu, sd = data
    model, p, _ = S.train_operator("mlp", x_tr, y_tr, x_va, y_va, seed)
    with torch.no_grad():
        pred = model(torch.tensor(x_te)).numpy()
    se = ((pred - y_te) ** 2).mean(1)
    price = pred * sd[0, -1] + mu[0, -1]
    return se, price, p


def mspe(price_pred, price_true):
    d = np.where(np.abs(price_true) < 1e-9, np.nan, price_true)
    return float(np.nanmean(((price_pred - d) / d) ** 2))


def main():
    rows = []
    for tag in S.ASSETS:
        df, raw, close, num_train, border1_test = S.load_asset(tag)
        n_rows = len(df)
        n_test = n_rows - border1_test - S.SEQ_LEN - S.PRED_LEN + 1
        n_train_s = num_train - S.SEQ_LEN - S.PRED_LEN + 1
        border1_val = num_train - S.SEQ_LEN
        n_val = (n_rows - int(n_rows * S.TEST_R)) - border1_val - S.SEQ_LEN - S.PRED_LEN + 1
        scaled, mu, sd = S.standardize(raw, num_train)
        x_tr, y_tr = S.make_windows(scaled, 0, n_train_s)
        x_va, y_va = S.make_windows(scaled, border1_val, n_val)
        x_te, y_te = S.make_windows(scaled, border1_test, n_test)
        tr_f, te_f = S.all_features(close, border1_test, n_test, n_train_s)
        states, _ = S.build_regimes(tr_f, te_f)
        data = (x_tr, y_tr, x_va, y_va, x_te, y_te, mu, sd)

        for seed in S.SEEDS:
            se_l, px_l, px_t, p_l = train_linear_once(tag, seed, data)
            print(f"[linear {tag} s{seed}] params={p_l}", flush=True)
            for w in WIDTHS:
                se_n, px_n, p_n = train_mlp_width(tag, seed, data, w)
                print(f"[mlp w={w} {tag} s{seed}] params={p_n}", flush=True)
                for sname, mask in states.items():
                    if mask.sum() < 20:
                        continue
                    d = se_n[mask] - se_l[mask]
                    lo, hi = S.bootstrap_ci(d, seed=seed)
                    rows.append({
                        "width": w, "asset": tag, "seed": seed, "state": sname, "n": int(mask.sum()),
                        "params_L": p_l, "params_N": p_n,
                        "MSE_L": float(se_l[mask].mean()), "MSE_N": float(se_n[mask].mean()),
                        "MSPE_L": mspe(px_l[mask], px_t[mask]), "MSPE_N": mspe(px_n[mask], px_t[mask]),
                        "P95_L": float(np.percentile(se_l[mask], 95)), "P95_N": float(np.percentile(se_n[mask], 95)),
                        "dMSE": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                        "sig": "sig" if (lo > 0 or hi < 0) else "ns",
                        "winN": float((se_n[mask] < se_l[mask]).mean()),
                    })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(S.REPO, "reproduction", "results"), exist_ok=True)
    df.to_csv(os.path.join(S.REPO, "reproduction", "results", f"operator-regime-capacity{SUFFIX}.csv"), index=False)
    print(f"\n[saved] reproduction/results/operator-regime-capacity{SUFFIX}.csv", flush=True)

    # ---- 汇总 1：按容量
    g = df.groupby("width").agg(
        params=("params_N", "max"), mean_dMSE=("dMSE", "mean"),
        states_pref_nonlin=("dMSE", lambda s: int((s < 0).sum())), states_total=("dMSE", "size"),
        mean_winN=("winN", "mean"),
    ).reset_index()
    g["frac_pref_nonlin"] = (g.states_pref_nonlin / g.states_total).round(3)
    g.to_csv(os.path.join(OUTDIR, f"operator-regime-capacity-summary{SUFFIX}.csv"), index=False)

    # ---- 汇总 2：Range(ΔMSE) 随容量
    rng = []
    for w in WIDTHS:
        sub = df[df.width == w]
        per = sub.groupby(["asset", "seed", "state"])["dMSE"].mean()
        r = per.groupby(level=0).agg(lambda s: float(s.max() - s.min()))
        rng.append({"width": w, "mean_range": float(r.mean()), **{a: float(v) for a, v in r.items()}})
    rng_df = pd.DataFrame(rng)

    # ---- 汇总 3：符号反转 + 逐 seed 一致性
    piv = df.pivot_table(index=["asset", "state", "seed"], columns="width", values="dMSE")
    reversals = []
    for (a, st, sd_), row in piv.iterrows():
        signs = {w: (1 if row[w] > 0 else -1) for w in WIDTHS if not pd.isna(row.get(w))}
        if len(set(signs.values())) > 1:
            reversals.append({"asset": a, "state": st, "seed": int(sd_),
                              **{f"dMSE_{w}": float(row[w]) for w in WIDTHS if not pd.isna(row.get(w))}})

    cons = []
    for (w, a, st), grp in df.groupby(["width", "asset", "state"]):
        sg = grp["dMSE"].apply(lambda x: 1 if x > 0 else -1)
        cons.append({"width": w, "asset": a, "state": st, "n_seeds": len(sg),
                     "sign_consistent": bool(sg.nunique() == 1), "mean_dMSE": float(grp["dMSE"].mean())})
    cons_df = pd.DataFrame(cons)

    lines = ["# 容量敏感性检查（Capacity Sensitivity）", ""]
    lines.append(f"隐层宽度 ∈ {WIDTHS}；其余协议与主实验完全一致；linear 算子与宽度无关（同种子复用）。")
    lines.append("")
    lines.append("## 汇总：按容量")
    lines.append("")
    lines.append("| hidden | MLP 参数量 | 平均 ΔMSE | 偏好非线性的状态数 | 非线性平均胜率 |")
    lines.append("|---|---|---|---|---|")
    for _, r in g.iterrows():
        lines.append(f"| {int(r.width)} | {int(r.params)} | {r.mean_dMSE:+.5f} | "
                     f"{int(r.states_pref_nonlin)}/{int(r.states_total)} | {r.mean_winN:.1%} |")
    lines.append("")
    lines.append("## 检查 (2)：状态间 Range(ΔMSE) 随容量变化")
    lines.append("")
    lines.append("| hidden | 平均状态间极差 | " + " | ".join(S.ASSETS) + " |")
    lines.append("|---|---|" + "---|" * len(S.ASSETS))
    for _, r in rng_df.iterrows():
        lines.append(f"| {int(r.width)} | {r.mean_range:.5f} | " +
                     " | ".join(f"{r[a]:.5f}" for a in S.ASSETS) + " |")
    lines.append("")
    lines.append("## 检查 (1)：符号反转（同一 asset/state 在不同容量下 ΔMSE 变号）")
    lines.append("")
    if reversals:
        lines.append("| asset | state | seed | " + " | ".join(f"ΔMSE_{w}" for w in WIDTHS) + " |")
        lines.append("|---|---|---|" + "---|" * len(WIDTHS))
        for r in reversals:
            vals = " | ".join(f"{r.get(f'dMSE_{w}', float('nan')):+.5f}" for w in WIDTHS)
            lines.append(f"| {r['asset']} | {r['state']} | {r['seed']} | {vals} |")
    else:
        lines.append("**未发现任何符号反转**（所有 asset/state/seed 在 23/64/128 下 ΔMSE 同号）。")
    lines.append("")
    lines.append("## 检查 (3)：逐 seed 方向一致性")
    lines.append("")
    for w in WIDTHS:
        sub = cons_df[cons_df.width == w]
        bad = sub[~sub.sign_consistent]
        lines.append(f"- hidden={w}: {int(sub.sign_consistent.sum())}/{len(sub)} 个 asset×state 的 3 个种子方向一致"
                     + (f"；不一致：{', '.join((bad.asset + '/' + bad.state).tolist())}" if len(bad) else ""))
    lines.append("")
    lines.append("## 结论（按协议 §4 三路诊断）")
    lines.append("")
    frac128 = float(g[g.width == 128].frac_pref_nonlin.iloc[0])
    if frac128 == 0:
        lines.append("- **Case 1**：扩大容量后 GSPC/BTC 仍全面偏 linear，且没有出现跨状态的稳定反转 "
                     "→ 结论加强：*routing is not the bottleneck*；不再继续增加 MLP 宽度。")
    else:
        lines.append(f"- 128 维下有 {frac128:.1%} 的状态偏好非线性。是否构成 Case 2（跨 seed、CI 支持、非单资产）"
                     "需对照上表逐项核对；仅 ETH 成立则为 Case 3（asset-specific conditional specialization）。")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(OUTDIR, f"capacity-sign-reversal{SUFFIX}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md, flush=True)
    print(f"[saved] capacity-sign-reversal{SUFFIX}.md / operator-regime-capacity-summary{SUFFIX}.csv", flush=True)


if __name__ == "__main__":
    main()
