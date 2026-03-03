"""
Publication-quality visualization for time-series forecasting results.

Usage:
    python tools/draw_result.py                          # 互動式選擇
    python tools/draw_result.py --folder results/xxx/    # 指定資料夾
    python tools/draw_result.py --compare                # 多模型比較
    python tools/draw_result.py --all                    # 全部圖表
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import argparse
import os
import re

# ──────────────────────────────────────────────
# 全域期刊風格設定 (Nature / IEEE 風格)
# ──────────────────────────────────────────────
plt.rcParams.update({
    # Font
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    # Axes
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    # Legend
    "legend.frameon": True,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
})

# 配色方案 — 色盲友善 (Okabe-Ito 改良)
COLORS = {
    "gt":   "#2c3e50",   # 深灰藍 — Ground Truth
    "pred": "#e74c3c",   # 紅 — Prediction
    "fill": "#3498db",   # 藍 — 信賴區間 / 誤差帶
    "bar1": "#2980b9",
    "bar2": "#27ae60",
    "bar3": "#f39c12",
    "bar4": "#8e44ad",
    "bar5": "#e67e22",
    "bar6": "#1abc9c",
}

MODEL_PALETTE = [
    "#2980b9", "#e74c3c", "#27ae60", "#f39c12",
    "#8e44ad", "#e67e22", "#1abc9c", "#d35400",
    "#7f8c8d", "#c0392b", "#16a085", "#2c3e50",
]

METRIC_NAMES = ["MAE", "MSE", "RMSE", "MAPE", "MSPE", "R²"]


# ──────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────
def _load_results(folder_path: str):
    """載入 pred.npy, true.npy, metrics.npy"""
    preds = np.load(os.path.join(folder_path, "pred.npy"))
    trues = np.load(os.path.join(folder_path, "true.npy"))
    metrics = None
    mp = os.path.join(folder_path, "metrics.npy")
    if os.path.exists(mp):
        metrics = np.load(mp, allow_pickle=True)
    return preds, trues, metrics


def _ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _save_fig(fig, folder_path: str, filename: str):
    """將圖片存為 PNG 到 results/ 對應的 folder 下"""
    _ensure_dir(folder_path)
    fp = os.path.join(folder_path, filename)
    fig.savefig(fp, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"✅ Saved: {fp}")
    plt.close(fig)


def _parse_setting(folder_name: str) -> dict:
    """
    從 setting 字串解析模型名、資料集、pred_len 等資訊。
    典型格式: long_term_forecast_ETTh1_96_192_PatchTST_...
    """
    info = {"raw": folder_name}
    parts = folder_name.split("_")

    # 嘗試抓 model name
    known_models = _scan_known_models()
    for p in parts:
        if p in known_models:
            info["model"] = p
            break
    if "model" not in info:
        info["model"] = folder_name[:30]

    # 嘗試抓 pred_len (連續兩個數字常為 seq_len, pred_len)
    nums = re.findall(r"(?:^|_)(\d+)(?=_)", folder_name)
    if len(nums) >= 2:
        info["seq_len"] = int(nums[-2])
        info["pred_len"] = int(nums[-1])

    # 嘗試抓 dataset
    for ds in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL", "traffic",
                "weather", "illness", "Exchange", "TSMC", "yfinance"]:
        if ds.lower() in folder_name.lower():
            info["dataset"] = ds
            break

    return info


def _scan_known_models() -> set:
    """掃描 models/ 資料夾取得所有模型名稱"""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    names = set()
    if os.path.exists(models_dir):
        for root, dirs, files in os.walk(models_dir):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith(".py") and not f.startswith("__"):
                    names.add(f[:-3])
    return names


def list_result_folders() -> list:
    """列出 results/ 下所有有效資料夾"""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    if not os.path.exists(base):
        print("results/ 資料夾不存在")
        return []
    dirs = sorted([
        os.path.join(base, d) for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
        and os.path.exists(os.path.join(base, d, "pred.npy"))
    ])
    if not dirs:
        print("results/ 內未找到含有 pred.npy 的子資料夾")
    return dirs


# ──────────────────────────────────────────────
# Figure 1: 預測曲線 (單欄/雙欄寬)
# ──────────────────────────────────────────────
def fig_prediction_curves(
    folder_path: str,
    feature_idx: int = -1,
    n_samples: int = 3,
):
    """
    繪製 Ground Truth vs Prediction 對比曲線。
    每個 sample 一列，附帶誤差陰影帶。
    PNG 存入 folder_path。
    """
    preds, trues, metrics = _load_results(folder_path)
    info = _parse_setting(os.path.basename(folder_path.rstrip(os.sep)))
    n_total = preds.shape[0]
    pred_len = preds.shape[1]

    indices = np.linspace(0, n_total - 1, n_samples, dtype=int)

    fig_width = 7.16  # IEEE 雙欄寬 (inch)
    fig, axes = plt.subplots(
        n_samples, 1,
        figsize=(fig_width, 1.8 * n_samples + 0.6),
        sharex=True,
    )
    if n_samples == 1:
        axes = [axes]

    timesteps = np.arange(pred_len)

    for row, idx in enumerate(indices):
        ax = axes[row]
        gt = trues[idx, :, feature_idx]
        pd_ = preds[idx, :, feature_idx]
        err = np.abs(gt - pd_)

        ax.plot(timesteps, gt, color=COLORS["gt"], label="Ground Truth", zorder=3)
        ax.plot(timesteps, pd_, color=COLORS["pred"], linestyle="--", label="Prediction", zorder=3)
        ax.fill_between(
            timesteps,
            pd_ - err * 0.5,
            pd_ + err * 0.5,
            color=COLORS["fill"],
            alpha=0.12,
            label="Error band",
            zorder=1,
        )
        ax.set_ylabel("Value")
        if row == 0:
            ax.legend(loc="upper right", ncol=3)

        ax.text(
            0.98, 0.92, f"Sample #{idx}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, style="italic", color="0.4",
        )

    axes[-1].set_xlabel("Prediction Horizon (time steps)")

    title_parts = []
    if "model" in info:
        title_parts.append(info["model"])
    if "dataset" in info:
        title_parts.append(info["dataset"])
    if "pred_len" in info:
        title_parts.append(f"H={info['pred_len']}")
    fig.suptitle(" — ".join(title_parts) if title_parts else "Prediction Curves", fontweight="bold")

    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    _save_fig(fig, folder_path, "fig_prediction_curves.png")


# ──────────────────────────────────────────────
# Figure 2: 每步 MSE + 誤差直方圖 (雙面板)
# ──────────────────────────────────────────────
def fig_error_analysis(
    folder_path: str,
    feature_idx: int = -1,
):
    """(a) 每個 time-step 的 MSE；(b) 誤差值分布直方圖。"""
    preds, trues, metrics = _load_results(folder_path)
    info = _parse_setting(os.path.basename(folder_path.rstrip(os.sep)))
    pred_len = preds.shape[1]

    errors = (preds[:, :, feature_idx] - trues[:, :, feature_idx]).flatten()
    mse_per_step = np.mean((preds[:, :, feature_idx] - trues[:, :, feature_idx]) ** 2, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5))

    # (a) MSE per step — 漸層色 bar
    norm_vals = mse_per_step / (mse_per_step.max() + 1e-12)
    cmap = plt.cm.YlOrRd
    bar_colors = cmap(norm_vals * 0.7 + 0.15)
    ax1.bar(range(pred_len), mse_per_step, color=bar_colors, edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("Prediction Step")
    ax1.set_ylabel("MSE")
    ax1.set_title("(a) MSE per Horizon Step", fontsize=10)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

    # (b) Error distribution
    ax2.hist(errors, bins=80, density=True, color=COLORS["fill"], alpha=0.7, edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color=COLORS["pred"], linewidth=1.2, linestyle="--", alpha=0.8)
    mu, sigma = errors.mean(), errors.std()
    ax2.set_xlabel("Prediction Error")
    ax2.set_ylabel("Density")
    ax2.set_title(f"(b) Error Distribution ($\\mu$={mu:.4f}, $\\sigma$={sigma:.4f})", fontsize=10)

    fig.suptitle(
        f"Error Analysis — {info.get('model', '')}  {info.get('dataset', '')}",
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    _save_fig(fig, folder_path, "fig_error_analysis.png")


# ──────────────────────────────────────────────
# Figure 3: Metrics 雷達圖 (單一實驗)
# ──────────────────────────────────────────────
def fig_metrics_radar(
    folder_path: str,
):
    """六軸雷達圖顯示 MAE, MSE, RMSE, MAPE, MSPE, R²。"""
    _, _, metrics = _load_results(folder_path)
    if metrics is None or len(metrics) < 6:
        print("⚠️  metrics.npy 不存在或長度不足，跳過雷達圖")
        return
    info = _parse_setting(os.path.basename(folder_path.rstrip(os.sep)))

    values = metrics[:6].astype(float).tolist()
    # 將前 5 個 error 指標做 1/(1+x) 映射，使越小越好 → 越大越好
    radar_vals = [1.0 / (1.0 + v) for v in values[:5]] + [max(0, values[5])]

    N = len(METRIC_NAMES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    radar_vals += radar_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
    ax.plot(angles, radar_vals, "o-", color=COLORS["bar1"], linewidth=1.5, markersize=5)
    ax.fill(angles, radar_vals, color=COLORS["bar1"], alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), METRIC_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="0.5")
    ax.set_title(
        f"{info.get('model', 'Model')} — {info.get('dataset', '')}",
        fontweight="bold", pad=18,
    )

    # 在雷達圖下方列出原始數值
    text_lines = "  |  ".join(f"{n}: {v:.4f}" for n, v in zip(METRIC_NAMES, values))
    fig.text(0.5, 0.02, text_lines, ha="center", fontsize=7.5, color="0.35")

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    _save_fig(fig, folder_path, "fig_metrics_radar.png")


# ──────────────────────────────────────────────
# Figure 4: 多模型 / 多 pred_len 比較 (柱狀圖)
# ──────────────────────────────────────────────
def fig_multi_model_comparison(
    folders: list[str] | None = None,
    metric_indices: tuple = (0, 1),   # MAE, MSE
):
    """
    跨設定 / 跨模型比較指定 metric。
    自動從 results/ 搜集所有可用結果。
    PNG 存到 results/ 根目錄。
    """
    if folders is None:
        folders = list_result_folders()
    if not folders:
        return

    records = []
    for f in folders:
        mp = os.path.join(f, "metrics.npy")
        if not os.path.exists(mp):
            continue
        m = np.load(mp, allow_pickle=True)
        info = _parse_setting(os.path.basename(f.rstrip(os.sep)))
        label = info.get("model", "?")
        if "pred_len" in info:
            label += f"\nH={info['pred_len']}"
        records.append((label, m))

    if len(records) == 0:
        print("⚠️  未找到有 metrics.npy 的結果")
        return

    n_metrics = len(metric_indices)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics + 0.5, 3.2))
    if n_metrics == 1:
        axes = [axes]

    labels = [r[0] for r in records]
    x = np.arange(len(labels))
    bar_w = 0.6

    for col, mi in enumerate(metric_indices):
        ax = axes[col]
        vals = [r[1][mi] if mi < len(r[1]) else 0 for r in records]
        colors = [MODEL_PALETTE[i % len(MODEL_PALETTE)] for i in range(len(vals))]
        bars = ax.bar(x, vals, bar_w, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(METRIC_NAMES[mi] if mi < len(METRIC_NAMES) else f"Metric {mi}")
        ax.set_title(METRIC_NAMES[mi] if mi < len(METRIC_NAMES) else f"Metric {mi}", fontweight="bold")

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.4f}", ha="center", va="bottom", fontsize=7,
            )

    fig.suptitle("Model Comparison", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # 存到 results/ 根目錄
    results_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    _save_fig(fig, results_base, "fig_model_comparison.png")


# ──────────────────────────────────────────────
# Figure 5: 預測熱力圖 (所有 sample × 所有 step)
# ──────────────────────────────────────────────
def fig_error_heatmap(
    folder_path: str,
    feature_idx: int = -1,
    max_samples: int = 200,
):
    """
    橫軸: prediction horizon, 縱軸: sample index
    色彩: absolute error
    """
    preds, trues, _ = _load_results(folder_path)
    info = _parse_setting(os.path.basename(folder_path.rstrip(os.sep)))

    abs_err = np.abs(preds[:, :, feature_idx] - trues[:, :, feature_idx])
    if abs_err.shape[0] > max_samples:
        step = abs_err.shape[0] // max_samples
        abs_err = abs_err[::step][:max_samples]

    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    im = ax.imshow(
        abs_err, aspect="auto", cmap="YlOrRd", interpolation="nearest", origin="lower",
    )
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Sample Index")
    ax.set_title(
        f"Absolute Error Heatmap — {info.get('model', '')}  {info.get('dataset', '')}",
        fontweight="bold",
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02, aspect=30)
    cb.set_label("| Pred − True |", fontsize=9)
    plt.tight_layout()

    _save_fig(fig, folder_path, "fig_error_heatmap.png")


# ──────────────────────────────────────────────
# Figure 6: 綜合 Dashboard (一頁四圖)
# ──────────────────────────────────────────────
def fig_dashboard(
    folder_path: str,
    feature_idx: int = -1,
):
    """一頁四面板綜合報告：(a) 預測曲線 (b) MSE per step (c) 誤差分布 (d) 雷達圖。"""
    preds, trues, metrics = _load_results(folder_path)
    info = _parse_setting(os.path.basename(folder_path.rstrip(os.sep)))
    pred_len = preds.shape[1]
    n_total = preds.shape[0]

    fig = plt.figure(figsize=(7.16, 6.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)

    # ── (a) Prediction Curve ──
    ax_a = fig.add_subplot(gs[0, :])
    mid_idx = n_total // 2
    gt = trues[mid_idx, :, feature_idx]
    pd_ = preds[mid_idx, :, feature_idx]
    t = np.arange(pred_len)
    ax_a.plot(t, gt, color=COLORS["gt"], label="Ground Truth")
    ax_a.plot(t, pd_, color=COLORS["pred"], linestyle="--", label="Prediction")
    ax_a.fill_between(t, gt, pd_, color=COLORS["fill"], alpha=0.12)
    ax_a.set_xlabel("Time Step")
    ax_a.set_ylabel("Value")
    ax_a.legend(loc="upper right", ncol=2)
    ax_a.set_title("(a) Prediction vs Ground Truth", fontsize=10, fontweight="bold")

    # ── (b) MSE per step ──
    ax_b = fig.add_subplot(gs[1, 0])
    mse_step = np.mean((preds[:, :, feature_idx] - trues[:, :, feature_idx]) ** 2, axis=0)
    norm_v = mse_step / (mse_step.max() + 1e-12)
    cmap = plt.cm.YlOrRd
    ax_b.bar(range(pred_len), mse_step, color=cmap(norm_v * 0.7 + 0.15), edgecolor="white", linewidth=0.3)
    ax_b.set_xlabel("Prediction Step")
    ax_b.set_ylabel("MSE")
    ax_b.set_title("(b) MSE per Horizon", fontsize=10, fontweight="bold")

    # ── (c) Error distribution ──
    ax_c = fig.add_subplot(gs[1, 1])
    errors = (preds[:, :, feature_idx] - trues[:, :, feature_idx]).flatten()
    ax_c.hist(errors, bins=80, density=True, color=COLORS["fill"], alpha=0.7, edgecolor="white", linewidth=0.3)
    ax_c.axvline(0, color=COLORS["pred"], linewidth=1, linestyle="--")
    mu, sigma = errors.mean(), errors.std()
    ax_c.set_xlabel("Error")
    ax_c.set_ylabel("Density")
    ax_c.set_title(f"(c) Error Dist. ($\\mu$={mu:.3f}, $\\sigma$={sigma:.3f})", fontsize=10, fontweight="bold")

    title = f"{info.get('model', 'Model')} — {info.get('dataset', '')} — H={info.get('pred_len', '?')}"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # metrics 文字列
    if metrics is not None and len(metrics) >= 6:
        vals = metrics[:6].astype(float)
        txt = "  |  ".join(f"{n}: {v:.4f}" for n, v in zip(METRIC_NAMES, vals))
        fig.text(0.5, 0.005, txt, ha="center", fontsize=7.5, color="0.4")

    _save_fig(fig, folder_path, "fig_dashboard.png")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def _interactive_select() -> str:
    dirs = list_result_folders()
    if not dirs:
        raise SystemExit(1)
    print("\nAvailable results:")
    for i, d in enumerate(dirs):
        name = os.path.basename(d.rstrip(os.sep))
        print(f"  [{i}] {name}")
    try:
        choice = int(input("\nSelect folder index: "))
        return dirs[choice]
    except (ValueError, IndexError):
        print("Invalid selection.")
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Publication-quality visualization for FinTSLib results",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--folder", type=str, default=None, help="Path to a single result folder")
    parser.add_argument("--feature", type=int, default=-1, help="Feature index (-1 = last)")
    parser.add_argument("--n_samples", type=int, default=3, help="Number of sample curves to plot")
    parser.add_argument("--compare", action="store_true", help="Run multi-model comparison across all results/")
    parser.add_argument("--dashboard", action="store_true", help="Generate single-page dashboard")
    parser.add_argument("--all", action="store_true", help="Generate ALL figure types")
    parser.add_argument("--batch", action="store_true", help="Process ALL result folders under results/")
    args = parser.parse_args()

    if args.compare:
        fig_multi_model_comparison()
        return

    # --batch: 對 results/ 下所有資料夾批次產圖
    if args.batch:
        folders = list_result_folders()
        if not folders:
            return
        for i, folder in enumerate(folders):
            name = os.path.basename(folder.rstrip(os.sep))
            print(f"\n{'='*60}")
            print(f"📊 [{i+1}/{len(folders)}] {name}")
            print(f"{'='*60}")
            fig_prediction_curves(folder, feature_idx=args.feature, n_samples=args.n_samples)
            fig_error_analysis(folder, feature_idx=args.feature)
            fig_metrics_radar(folder)
            fig_error_heatmap(folder, feature_idx=args.feature)
            fig_dashboard(folder, feature_idx=args.feature)
        fig_multi_model_comparison()
        print(f"\n🎉 批次完成！共處理 {len(folders)} 個結果資料夾")
        return

    folder = args.folder or _interactive_select()

    if args.dashboard:
        fig_dashboard(folder, feature_idx=args.feature)
        return

    if args.all:
        fig_prediction_curves(folder, feature_idx=args.feature, n_samples=args.n_samples)
        fig_error_analysis(folder, feature_idx=args.feature)
        fig_metrics_radar(folder)
        fig_error_heatmap(folder, feature_idx=args.feature)
        fig_dashboard(folder, feature_idx=args.feature)
        fig_multi_model_comparison()
        return

    # 預設：逐一產出
    fig_prediction_curves(folder, feature_idx=args.feature, n_samples=args.n_samples)
    fig_error_analysis(folder, feature_idx=args.feature)
    fig_metrics_radar(folder)
    fig_error_heatmap(folder, feature_idx=args.feature)
    fig_dashboard(folder, feature_idx=args.feature)


if __name__ == "__main__":
    main()