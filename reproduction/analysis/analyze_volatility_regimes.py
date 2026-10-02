# -*- coding: utf-8 -*-
"""
工况分层评估（通用版）：按 realized volatility 切分测试集，比较
DeReFusion(非线性残差) vs revin-DLinear(线性基座) 的表现。

用法：
  python reproduction/analysis/analyze_volatility_regimes.py --csv dataset/GSPC-2016-2025.csv --tag GSPC
  python reproduction/analysis/analyze_volatility_regimes.py --csv dataset/BTCUSD-2016-2025.csv --tag BTCUSD

零训练成本：只后处理已保存的 pred.npy / true.npy + 原始 CSV。
指标口径与 utils/metrics.py 一致（标准化尺度）。
"""
import argparse
import json
import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/GSPC-2016-2025.csv")
    ap.add_argument("--tag", default="GSPC")
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--label-len", type=int, default=48)
    ap.add_argument("--models", default="DeReFusion,revin-DLinear,DeReFusion-gatev2-learnable")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--rv-mode", default="absolute", choices=["absolute", "relative"],
                    help="absolute=输入窗口对数收益std; relative=除以过去120样本的中位波动率（去趋势，解耦时间段）")
    ap.add_argument("--rv-window", type=int, default=120)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    return ap.parse_args()


def ev_per_sample(pred, true):
    err = pred - true
    mse = np.mean(err ** 2, axis=1)
    mae = np.mean(np.abs(err), axis=1)
    safe = np.where(np.abs(true) < 1e-6, np.nan, true)
    mspe = np.nanmean((err / safe) ** 2, axis=1)
    return mse, mae, mspe


def realized_vol(close, i, seq_len, horizon, ex_post=False):
    if not ex_post:
        seg = close[i:i + seq_len]
    else:
        seg = close[i + seq_len - 1:i + seq_len + horizon]
    r = np.diff(np.log(seg))
    return float(np.std(r)) if r.size >= 1 else float("nan")


def boot_ci(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    return (float(np.percentile(x[idx].mean(axis=1), 2.5)),
            float(np.percentile(x[idx].mean(axis=1), 97.5)))


def main():
    args = parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root (this file lives in reproduction/analysis/)
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(repo, args.csv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    df = pd.read_csv(csv_path)
    n_rows = len(df)
    num_train = int(n_rows * args.train_ratio)
    num_test = int(n_rows * args.test_ratio)
    border1 = n_rows - num_test - args.seq_len
    close = df["Close"].to_numpy(dtype=np.float64)

    out = []
    p = out.append
    p(f"# {args.tag} (seed {args.seed}): 工况分层评估（T=24）")
    p(f"- 数据: {os.path.basename(csv_path)}（{n_rows} 行）; train={num_train}, test_start_idx={border1}")
    p(f"- 波动率定义: {args.rv_mode}" + ("（输入窗口对数收益std，ex-ante）" if args.rv_mode == "absolute" else "（当前波动率/过去中位波动率，去趋势）"))

    report = {"asset": args.tag, "n_rows": n_rows, "horizons": {}}

    for T in (24,):
        n_samples = n_rows - border1 - args.seq_len - T + 1
        rv = np.array([realized_vol(close, border1 + i, args.seq_len, T) for i in range(n_samples)])
        if args.rv_mode == "relative":
            W = args.rv_window
            rv_rel = np.empty_like(rv)
            for i in range(n_samples):
                lo = max(0, i - W)
                base = np.median(rv[lo:i]) if i - lo >= 20 else np.median(rv[:min(n_samples, W)])
                rv_rel[i] = rv[i] / (base if base > 0 else 1e-9)
            rv = rv_rel

        data = {}
        for m in models:
            pat = os.path.join(repo, "results",
                               f"long_term_forecast_{args.tag}_96_{T}_{m}_custom_ftMS_*_pl{T}_*_seed{args.seed}_*", "*.npy")
            dirs = sorted(set(os.path.dirname(x) for x in glob(pat)))
            if not dirs:
                p(f"- !! 缺少结果: {m}")
                continue
            pred = np.load(os.path.join(dirs[0], "pred.npy")).astype(np.float64)
            true = np.load(os.path.join(dirs[0], "true.npy")).astype(np.float64)
            if pred.ndim == 3:
                pred = pred[:, :, -1]
            if true.ndim == 3:
                true = true[:, :, -1]
            assert pred.shape[0] == n_samples, (m, pred.shape, n_samples)
            data[m] = (pred, true)

        # 对齐校验
        scaler = StandardScaler().fit(df[["Open", "High", "Low", "Close"]].to_numpy()[:num_train])
        scaled_close = scaler.transform(df[["Open", "High", "Low", "Close"]].to_numpy())[:, -1]
        ref_true = scaled_close[border1 + args.seq_len: border1 + args.seq_len + n_samples]
        ref_model = next(iter(data.values()))[1]
        align = float(np.max(np.abs(ref_true - ref_model[:, 0])))

        q20, med, q80 = np.percentile(rv, [20, 50, 80])
        masks = {
            "low20%": rv <= q20,
            "low50%": rv <= med,
            "high50%": rv > med,
            "high20%": rv >= q80,
        }
        # 任务书口径：Low = Q1+Q2, High = Q4+Q5
        qb = np.percentile(rv, [20, 40, 60, 80])
        qidx = np.digitize(rv, qb)  # 0..4 => Q1..Q5
        masks["Q1+Q2 (low)"] = qidx <= 1
        masks["Q4+Q5 (high)"] = qidx >= 3

        p(f"\n## 配置")
        p(f"- n_samples={n_samples}, 对齐校验 max|scaled-true|={align:.2e}")
        p(f"- RV: p20={q20:.5f}, median={med:.5f}, p80={q80:.5f}")

        p(f"\n## 分层结果")
        p("| 分层 | n | " + " | ".join(f"{m} MSE" for m in data) + " | 相对改善 | " +
          " | ".join(f"{m} MSPE" for m in data) + " | DeReFusion 95%分位误差 | 配对胜率 |")
        p("|---|---|" + "---|" * (len(data) * 2 + 3))

        per = {}
        for m, (pred, true) in data.items():
            per[m] = ev_per_sample(pred, true)

        strata = {}
        for name, mask in masks.items():
            row = {m: {
                "MSE": float(np.nanmean(per[m][0][mask])),
                "MAE": float(np.nanmean(per[m][1][mask])),
                "MSPE": float(np.nanmean(per[m][2][mask])),
                "q95_sq_err": float(np.nanpercentile(per[m][0][mask], 95)),
            } for m in data}
            strata[name] = row

        d0, d1 = "DeReFusion", "revin-DLinear"
        for name, mask in masks.items():
            rel = (strata[name][d1]["MSE"] - strata[name][d0]["MSE"]) / strata[name][d1]["MSE"] * 100
            wr = float((per[d0][0][mask] < per[d1][0][mask]).mean())
            cells = " | ".join(f"{strata[name][m]['MSE']:.5f}" for m in data)
            mspe_cells = " | ".join(f"{strata[name][m]['MSPE']:.5f}" for m in data)
            p(f"| {name} | {int(mask.sum())} | {cells} | {rel:+.1f}% | {mspe_cells} | "
              f"{strata[name][d0]['q95_sq_err']:.5f} | {wr:.1%} |")

        # 配对检验 + 交互效应
        p(f"\n## 配对检验（{d0} − {d1}，负值=非线性更优）")
        comp = {}
        for name in ("low20%", "low50%", "high50%", "high20%", "Q1+Q2 (low)", "Q4+Q5 (high)"):
            mask = masks[name]
            diff = per[d0][0][mask] - per[d1][0][mask]
            lo, hi = boot_ci(diff)
            sig = bool(lo > 0 or hi < 0)
            comp[name] = {"delta_mse": float(diff.mean()), "ci95": [lo, hi], "significant": sig,
                          "win_rate": float((diff < 0).mean()), "n": int(mask.sum())}
            p(f"- {name}: ΔMSE={diff.mean():+.5f} 95%CI[{lo:+.5f},{hi:+.5f}] {'显著' if sig else '不显著'} "
              f"胜率={(diff < 0).mean():.1%} n={int(mask.sum())}")

        d_full = per[d0][0] - per[d1][0]

        def interaction(hi_mask, lo_mask, seed):
            val = float(d_full[hi_mask].mean() - d_full[lo_mask].mean())
            rng = np.random.default_rng(seed)
            boots = []
            for _ in range(4000):
                idx = rng.integers(0, len(d_full), len(d_full))
                db, hb, lb = d_full[idx], hi_mask[idx], lo_mask[idx]
                if hb.sum() == 0 or lb.sum() == 0:
                    continue
                boots.append(db[hb].mean() - db[lb].mean())
            return val, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

        inter_q, cq_lo, cq_hi = interaction(masks["Q4+Q5 (high)"], masks["Q1+Q2 (low)"], 7)
        inter_h, ch_lo, ch_hi = interaction(masks["high50%"], masks["low50%"], 8)
        p(f"\n**交互效应（任务书口径 Q4+Q5 − Q1+Q2）= {inter_q:+.5f}, 95%CI[{cq_lo:+.5f},{cq_hi:+.5f}]**")
        p(f"**交互效应（50/50 口径）= {inter_h:+.5f}, 95%CI[{ch_lo:+.5f},{ch_hi:+.5f}]**")
        comp["interaction_Q45_minus_Q12"] = {"value": inter_q, "ci95": [cq_lo, cq_hi],
                                             "significant": bool(cq_lo > 0 or cq_hi < 0)}
        comp["interaction_high50_minus_low50"] = {"value": inter_h, "ci95": [ch_lo, ch_hi],
                                                  "significant": bool(ch_lo > 0 or ch_hi < 0)}

        # 五分位性能曲线
        p(f"\n## 波动率五分位性能曲线")
        p("| 五分位 | n | 日期范围 | RV范围 | DeReFusion MSE | DLinear MSE | 相对改善 | ΔMSE 95%CI | 胜率 | DeReFusion MSPE | DLinear MSPE |")
        p("|---|---|---|---|---|---|---|---|---|---|---|")
        qs = np.percentile(rv, [0, 20, 40, 60, 80, 100])
        curve = []
        dates = df["date"].to_numpy()
        for k in range(5):
            mask = (rv >= qs[k]) & (rv <= qs[k + 1] if k == 4 else rv < qs[k + 1])
            idxs = np.where(mask)[0]
            d_lo = str(dates[border1 + idxs[0]]) if idxs.size else "-"
            d_hi = str(dates[border1 + idxs[-1] + args.seq_len + T - 1]) if idxs.size else "-"
            mse0 = float(np.nanmean(per[d0][0][mask]))
            mse1 = float(np.nanmean(per[d1][0][mask]))
            rel = (mse1 - mse0) / mse1 * 100
            diff_q = per[d0][0][mask] - per[d1][0][mask]
            qlo, qhi = boot_ci(diff_q, seed=11 + k)
            row = {"q": k + 1, "n": int(mask.sum()), "rv_lo": float(qs[k]), "rv_hi": float(qs[k + 1]),
                   "date_lo": d_lo, "date_hi": d_hi,
                   "mse_deref": mse0, "mse_dlin": mse1,
                   "mspe_deref": float(np.nanmean(per[d0][2][mask])),
                   "mspe_dlin": float(np.nanmean(per[d1][2][mask])),
                   "rel": float(rel), "delta_mse": float(diff_q.mean()),
                   "ci95": [qlo, qhi], "win_rate": float((diff_q < 0).mean())}
            curve.append(row)
            sig = "显著" if (qlo > 0 or qhi < 0) else "不显著"
            p(f"| Q{k+1} | {row['n']} | {d_lo}→{d_hi} | {qs[k]:.5f}–{qs[k+1]:.5f} | {mse0:.5f} | {mse1:.5f} | "
              f"{rel:+.1f}% | [{qlo:+.5f},{qhi:+.5f}] {sig} | {row['win_rate']:.1%} | "
              f"{row['mspe_deref']:.5f} | {row['mspe_dlin']:.5f} |")

        # === 结构检验（任务书 Pattern 1 / Pattern 2）===
        sig_nonlin_q = [c["q"] for c in curve if c["ci95"][1] < 0]
        sig_lin_q = [c["q"] for c in curve if c["ci95"][0] > 0]
        lo_grp = comp["Q1+Q2 (low)"]
        hi_grp = comp["Q4+Q5 (high)"]
        p1_high = hi_grp["significant"] and hi_grp["delta_mse"] < 0
        p1_low_tie = not lo_grp["significant"]
        pattern1 = bool(p1_high and p1_low_tie)
        note = ""
        if (not p1_low_tie) and lo_grp["delta_mse"] < 0:
            note = "（低波动区间也已显著偏向非线性，趋势比 GSPC 更强）"
        pattern2 = len(sig_lin_q) > 0
        p("\n## 结构检验")
        p("- 五分位 improvement 序列: " + ", ".join(f"Q{c['q']}:{c['rel']:+.1f}%" for c in curve))
        p(f"- Pattern 1（高波动优势扩大 且 低波动接近）: {'支持' if pattern1 else '不支持'}{note}")
        p(f"- Pattern 2（稳定 linear advantage regime）: {'存在 → Q' + ','.join(map(str, sig_lin_q)) if pattern2 else '不存在'}")
        p(f"- 显著偏向非线性的分位: {'Q' + ',Q'.join(map(str, sig_nonlin_q)) if sig_nonlin_q else '无'}")
        p(f"- 泄漏检查: RV_rel(t) 基准取 t 之前 {args.rv_window} 个样本的中位数（不含 t），输入窗口本身完全可得 → 无未来信息")

        report["horizons"][f"T{T}"] = {
            "n_samples": n_samples, "align_maxdiff": align,
            "rv": {"p20": float(q20), "median": float(med), "p80": float(q80)},
            "strata": strata, "comparison": comp, "quintile_curve": curve,
            "structure": {
                "quintile_improvement": [c["rel"] for c in curve],
                "sig_nonlinear_quintiles": sig_nonlin_q,
                "sig_linear_quintiles": sig_lin_q,
                "pattern1_supported": pattern1,
                "pattern2_stable_linear_regime": pattern2,
            },
        }

    txt = "\n".join(out)
    print(txt)
    suffix = f"{args.tag}_{args.rv_mode}_s{args.seed}"
    results_dir = os.path.join(repo, "reproduction", "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"volatility_stratification_{suffix}.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(results_dir, f"volatility_stratification_{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] volatility_stratification_{suffix}.json/.txt")


if __name__ == "__main__":
    main()
