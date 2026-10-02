# -*- coding: utf-8 -*-
"""
结构感知路由：最小化验证实验（Structure-Aware Routing proxy experiment）

研究问题（唯一）：**输入的结构状态是否决定哪种算子更有效？**
    Structural State -> Operator Preference
而不是：volatility -> alpha(X) -> mixing

纪律（按任务书）：
  - 不改 DeReFusion 主模型、不加 NS、不做 adaptive gate
  - 特征只用输入窗口（严格因果，无未来信息）
  - 分组阈值只用训练集分位数（测试集无泄漏）
  - 两个算子容量尽可能接近、训练协议完全一致
  - 不因结果修改分组规则、不做 cherry-picking

输出：
  - reproduction/results/structure_routing_raw.json     原始逐状态结果
  - reproduction/results/structural-state-summary.csv
  - reproduction/results/operator-regime-results.csv
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO, "reproduction", "results")

SEQ_LEN, LABEL_LEN, PRED_LEN = 96, 48, 24
C_IN, C_OUT = 4, 1
TRAIN_R, TEST_R = 0.7, 0.2
EPOCHS, BATCH, LR, PATIENCE = 30, 32, 1e-4, 5
SEEDS = [2021, 2022, 2023]
ASSETS = ["GSPC", "BTCUSD", "ETHUSD"]
JUMP_C = 3.0           # 预先固定的跳变阈值（局部 sigma 倍数）
RV_WINDOW = 120        # rv_rel 的回看样本数
MLP_HIDDEN = 23        # 使 MLP 参数量与线性算子接近（见下方参数量报告）
KMEANS_K = 3


# ---------------------------------------------------------------- data
def load_asset(tag):
    df = pd.read_csv(os.path.join(REPO, "dataset", f"{tag}-2016-2025.csv"))
    n = len(df)
    num_train = int(n * TRAIN_R)
    border1_test = n - int(n * TEST_R) - SEQ_LEN
    cols = ["Open", "High", "Low", "Close"]
    raw = df[cols].to_numpy(float)
    close = df["Close"].to_numpy(float)
    return df, raw, close, num_train, border1_test


def standardize(raw, num_train):
    mu = raw[:num_train].mean(0, keepdims=True)
    sd = raw[:num_train].std(0, keepdims=True)
    return (raw - mu) / sd, mu, sd


# ------------------------------------------------- structural features
def _acf(r, lag):
    if len(r) <= lag + 1:
        return np.nan
    a, b = r[:-lag], r[lag:]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def window_features(close, i):
    """只用 [i, i+SEQ_LEN) 的价格（严格过去）计算结构特征。"""
    seg = close[i:i + SEQ_LEN]
    r = np.diff(np.log(seg))
    sd = r.std()
    f = {
        "acf1": _acf(r, 1),
        "acf5": _acf(r, 5),
        "acf10": _acf(r, 10),
        "trend_tstat": float(r.mean() * np.sqrt(len(r)) / sd) if sd > 1e-12 else 0.0,
        "sign_persistence": float(abs(np.sign(r).sum()) / len(r)),
        "jump_ratio": float((np.abs(r) > JUMP_C * sd).mean()) if sd > 1e-12 else 0.0,
        "rv": float(sd),
        "skew": float(pd.Series(r).skew()),
        "kurt": float(pd.Series(r).kurt()),
    }
    return f


def all_features(close, border1_test, n_test, n_train):
    """训练样本与测试样本的结构特征（样本索引对齐 repo 的窗口切分）。"""
    def one(base, count):
        rows = []
        for i in range(count):
            rows.append(window_features(close, base + i))
        return pd.DataFrame(rows)

    tr = one(0, n_train)
    te = one(border1_test, n_test)
    # rv_rel：当前 rv / 之前 120 个样本的 rv 中位数（严格因果，只用到过去样本）
    def rv_rel(series_rv):
        out = np.empty(len(series_rv))
        for k in range(len(series_rv)):
            lo = max(0, k - RV_WINDOW)
            base = series_rv[lo:k]
            out[k] = series_rv[k] / np.median(base) if len(base) >= 20 and np.median(base) > 0 else np.nan
        return out

    tr["rv_rel"] = rv_rel(tr["rv"].to_numpy())
    te["rv_rel"] = rv_rel(te["rv"].to_numpy())
    return tr, te


# ------------------------------------------------------- operators
class LinearOp(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x)


class MLPOp(nn.Module):
    def __init__(self, d_in, d_out, hidden=MLP_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def make_windows(scaled, border1, count):
    xs, ys = [], []
    for i in range(count):
        a = border1 + i
        xs.append(scaled[a:a + SEQ_LEN].reshape(-1))
        ys.append(scaled[a + SEQ_LEN:a + SEQ_LEN + PRED_LEN, -1])
    return np.array(xs, np.float32), np.array(ys, np.float32)


def train_operator(kind, x_tr, y_tr, x_va, y_va, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d_in, d_out = x_tr.shape[1], y_tr.shape[1]
    model = LinearOp(d_in, d_out) if kind == "linear" else MLPOp(d_in, d_out)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.MSELoss()
    xt = torch.tensor(x_tr); yt = torch.tensor(y_tr)
    xv = torch.tensor(x_va); yv = torch.tensor(y_va)
    best, best_state, bad = float("inf"), None, 0
    n = len(xt)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            loss = lossf(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = float(lossf(model(xv), yv))
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, n_params, best


# ------------------------------------------------------------ regimes
def build_regimes(tr_f, te_f):
    """阈值来自训练集；K-means 也只在训练集上拟合。"""
    states = {}
    FEATS = ["acf1", "sign_persistence", "jump_ratio", "rv_rel"]
    for f in FEATS:
        thr = float(np.nanmedian(tr_f[f]))
        states[f"{f}_hi"] = (te_f[f] > thr).fillna(False).to_numpy()
        states[f"{f}_lo"] = (te_f[f] <= thr).fillna(False).to_numpy()

    # 复合"结构强度"：|acf1| + sign_persistence 的 z 和 减 jump 的 z
    def z(series):
        m, s = np.nanmean(series), np.nanstd(series)
        return (series - m) / (s if s > 1e-12 else 1.0)
    comp_tr = z(np.abs(tr_f["acf1"])) + z(tr_f["sign_persistence"]) - z(tr_f["jump_ratio"])
    comp_te = z(np.abs(te_f["acf1"])) + z(te_f["sign_persistence"]) - z(te_f["jump_ratio"])
    thr = float(np.nanmedian(comp_tr))
    states["structure_strong"] = (comp_te > thr).fillna(False).to_numpy()
    states["structure_weak"] = (comp_te <= thr).fillna(False).to_numpy()

    # K-means（K=3）交叉检验
    labels = None
    try:
        from sklearn.cluster import KMeans
        cols = ["acf1", "acf5", "acf10", "trend_tstat", "sign_persistence", "jump_ratio", "rv_rel", "skew", "kurt"]
        Xtr = tr_f[cols].fillna(0).to_numpy()
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        km = KMeans(n_clusters=KMEANS_K, n_init=10, random_state=2021).fit((Xtr - mu) / sd)
        Xte = (te_f[cols].fillna(0).to_numpy() - mu) / sd
        labels = km.predict(Xte)
        prof = {}
        for k in range(KMEANS_K):
            m = labels == k
            prof[f"kmeans_{k}"] = {c: float(np.nanmean(te_f[c].to_numpy()[m])) for c in cols} | {"n": int(m.sum())}
            states[f"kmeans_{k}"] = m
        return states, prof
    except Exception as e:  # noqa
        print("kmeans unavailable:", e)
        return states, {}


# --------------------------------------------------------------- main
def bootstrap_ci(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    all_rows, state_rows, profiles = [], [], {}
    for tag in ASSETS:
        df, raw, close, num_train, border1_test = load_asset(tag)
        n_rows = len(df)
        n_test = n_rows - border1_test - SEQ_LEN - PRED_LEN + 1
        n_train_s = num_train - SEQ_LEN - PRED_LEN + 1
        border1_val = num_train - SEQ_LEN   # 验证段起点（repo 口径）
        n_val = (n_rows - int(n_rows * TEST_R)) - border1_val - SEQ_LEN - PRED_LEN + 1

        scaled, mu, sd = standardize(raw, num_train)
        x_tr, y_tr = make_windows(scaled, 0, n_train_s)
        x_va, y_va = make_windows(scaled, border1_val, n_val)
        x_te, y_te = make_windows(scaled, border1_test, n_test)

        tr_f, te_f = all_features(close, border1_test, n_test, n_train_s)
        states, prof = build_regimes(tr_f, te_f)
        profiles[tag] = prof
        te_f.to_csv(os.path.join(REPO, f"structure_features_{tag}_test.csv"), index=False)

        for seed in SEEDS:
            preds = {}
            for kind in ("linear", "mlp"):
                model, n_params, best_val = train_operator(kind, x_tr, y_tr, x_va, y_va, seed)
                with torch.no_grad():
                    p = model(torch.tensor(x_te)).numpy()
                preds[kind] = (p, n_params)
                print(f"[{tag} s{seed}] {kind}: params={n_params} val_mse={best_val:.5f}")

            lp, lp_n = preds["linear"]
            np_, np_n = preds["mlp"]
            se_l = ((lp - y_te) ** 2).mean(1)
            se_n = ((np_ - y_te) ** 2).mean(1)
            # 反归一化后的价格尺度误差（用于稳定的 MSPE）
            px_l = lp * sd[0, -1] + mu[0, -1]
            px_n = np_ * sd[0, -1] + mu[0, -1]
            px_t = y_te * sd[0, -1] + mu[0, -1]

            for sname, mask in states.items():
                if mask.sum() < 20:
                    continue
                d = se_n[mask] - se_l[mask]
                lo, hi = bootstrap_ci(d, seed=seed)
                sig = "sig" if (lo > 0 or hi < 0) else "ns"
                row = {
                    "asset": tag, "seed": seed, "state": sname, "n": int(mask.sum()),
                    "MSE_L": float(se_l[mask].mean()), "MSE_N": float(se_n[mask].mean()),
                    "dMSE": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "sig": sig,
                    "win_rate_N": float((se_n[mask] < se_l[mask]).mean()),
                    "MSPE_L_price": float(np.mean(((px_l[mask] - px_t[mask]) / np.where(np.abs(px_t[mask]) < 1e-9, np.nan, px_t[mask])) ** 2)),
                    "MSPE_N_price": float(np.mean(((px_n[mask] - px_t[mask]) / np.where(np.abs(px_t[mask]) < 1e-9, np.nan, px_t[mask])) ** 2)),
                    "P95_sqerr_L": float(np.percentile(se_l[mask], 95)),
                    "P95_sqerr_N": float(np.percentile(se_n[mask], 95)),
                    "params_L": lp_n, "params_N": np_n,
                }
                all_rows.append(row)
                print(f"    {sname:18s} n={row['n']:4d} MSE_L={row['MSE_L']:.5f} MSE_N={row['MSE_N']:.5f} "
                      f"dMSE={row['dMSE']:+.5f} [{lo:+.5f},{hi:+.5f}] {sig} winN={row['win_rate_N']:.1%}")

    os.makedirs(OUTDIR, exist_ok=True)
    dfres = pd.DataFrame(all_rows)
    dfres.to_csv(os.path.join(OUTDIR, "operator-regime-results.csv"), index=False)

    # 状态汇总（跨种子聚合）
    agg = dfres.groupby(["asset", "state"]).agg(
        n=("n", "median"),
        MSE_L=("MSE_L", "mean"), MSE_N=("MSE_N", "mean"),
        dMSE_mean=("dMSE", "mean"), dMSE_sd=("dMSE", "std"),
        dMSE_min=("dMSE", "min"), dMSE_max=("dMSE", "max"),
        sig_seeds=("sig", lambda s: int((s == "sig").sum())),
        winN=("win_rate_N", "mean"),
    ).reset_index()
    agg.to_csv(os.path.join(OUTDIR, "structural-state-summary.csv"), index=False)

    raw = {"rows": all_rows, "kmeans_profiles": profiles}
    with open(os.path.join(REPO, "reproduction", "results", "structure_routing_raw.json"), "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print("\n[saved] operator-regime-results.csv / structural-state-summary.csv / structure_routing_raw.json")
    print("\n=== 状态级 ΔMSE（跨种子均值，负=非线性更优）===")
    for _, r in agg.sort_values(["asset", "state"]).iterrows():
        print(f"{r['asset']:7s} {r['state']:18s} n={int(r['n']):4d} dMSE={r['dMSE_mean']:+.5f} "
              f"(sd={0 if pd.isna(r['dMSE_sd']) else r['dMSE_sd']:.5f}, 显著种子 {r['sig_seeds']}/3)")


if __name__ == "__main__":
    main()
