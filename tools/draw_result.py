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
import traceback

# ──────────────────────────────────────────────
# 全域期刊風格設定 (Nature / IEEE 風格)
# ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "legend.frameon": True,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
})

COLORS = {
    "gt":   "#2c3e50",
    "pred": "#e74c3c",
    "fill": "#3498db",
    "bar1": "#2980b9",
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
    """載入 pred.npy, true.npy, metrics.npy，附帶 shape 檢查"""
    folder_path = os.path.abspath(folder_path)

    pred_path = os.path.join(folder_path, "pred.npy")
    true_path = os.path.join(folder_path, "true.npy")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"pred.npy 不存在: {pred_path}")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"true.npy 不存在: {true_path}")

    preds = np.load(pred_path)
    trues = np.load(true_path)

    print(f"  📐 pred.shape={preds.shape}, true.shape={trues.shape}")

    # shape 不一致時自動修正 feature 維度
    if preds.shape[-1] != trues.shape[-1]:
        print(f"  ⚠️  pred 與 true 的 feature 維度不同 "
              f"(pred={preds.shape[-1]}, true={trues.shape[-1]})，"
              f"將以較小者為準")

    metrics = None
    mp = os.path.join(folder_path, "metrics.npy")
    if os.path.exists(mp):
        metrics = np.load(mp, allow_pickle=True)
        print(f"  📊 metrics={metrics}")
    else:
        print(f"  ⚠️  metrics.npy 不存在")

    return preds, trues, metrics


def _safe_feature_idx(preds, trues, feature_idx: int) -> int:
    """安全取得 feature index，避免越界"""
    min_features = min(preds.shape[-1], trues.shape[-1])
    if feature_idx < 0:
        feature_idx = min_features + feature_idx
    if feature_idx < 0 or feature_idx >= min_features:
        print(f"  ⚠️  feature_idx={feature_idx} 越界 (max={min_features-1})，改用 0")
        feature_idx = 0
    return feature_idx


def _ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _save_fig(fig, folder_path: str, filename: str):
    """將圖片存為 PNG 到 results/ 對應的 folder 下"""
    folder_path = os.path.abspath(folder_path)
    _ensure_dir(folder_path)
    fp = os.path.join(folder_path, filename)
    fig.savefig(fp, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    if os.path.exists(fp):
        print(f"  ✅ Saved: {fp}  ({os.path.getsize(fp)} bytes)")
    else:
        print(f"  ❌ Failed to save: {fp}")
    plt.close(fig)


def _parse_setting(folder_name: str) -> dict:
    info = {"raw": folder_name}
    parts = folder_name.split("_")

    known_models = _scan_known_models()
    for p in parts:
        if p in known_models:
            info["model"] = p
            break
    if "model" not in info:
        info["model"] = folder_name[:30]

    nums = re.findall(r"(?:^|_)(\d+)(?=_)", folder_name)
    if len(nums) >= 2:
        info["seq_len"] = int(nums[-2])
        info["pred_len"] = int(nums[-1])

    for ds in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL", "traffic",
                "weather", "illness", "Exchange", "TSMC", "yfinance"]:
        if ds.lower() in folder_name.lower():
            info["dataset"] = ds
            break

    return info


def _scan_known_models() -> set:
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    names = set()
    if os.path.exists(models_dir):
        for root, dirs, files in os.walk(models_dir):
            dirs[:] = sorted([d for d in dirs if d != "__pycache__"])
            for f in sorted(files):
                if f.endswith(".py") and not f.startswith("__"):
                    names.add(f[:-3])
    return names


def list_result_folders() -> list:
    base_candidates = [
        os.path.join(os.getcwd(), "results"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"),
    ]
    base = None
    for candidate in base_candidates:
        if os.path.exists(candidate):
            base = candidate
            break
    if base is None:
        print("❌ results/ 資料夾不存在，搜尋過的路徑:")
        for c in base_candidates:
            print(f"   - {c}")
        return []

    print(f"📂 掃描目錄: {base}")
    dirs = sorted([
        os.path.join(base, d) for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
        and os.path.exists(os.path.join(base, d, "pred.npy"))
    ])
    if not dirs:
        print("⚠️  results/ 內未找到含有 pred.npy 的子資料夾")
    return dirs


# ──────────────────────────────────────────────
# Figure 1: 預測曲線
# ──────────────────────────────────────────────
def fig_prediction_curves(folder_path: str, feature_idx: int = -1, n_samples: int = 3):
    preds, trues, _ = _load_results(folder_path)
    fi = _safe_feature_idx(preds, trues, feature_idx)
    info = _parse_setting(os.path.basename(folder_path.rstrip(os.sep)))
    n_total = preds.shape[0]
    pred_len = preds.shape[1]

    indices = np.linspace(0, n_total - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 1, figsize=(7.16, 1.8 * n_samples + 0.6), sharex=True)
    if n_samples == 1:
        axes = [axes]

    timesteps = np.arange(pred_len)

    for row, idx in enumerate(indices):
        ax = axes[row]
        gt = trues[idx, :, fi]
        pd_ = preds[idx, :, fi]
        err = np.abs(gt - pd_)

        ax.plot(timesteps, gt, color=COLORS["gt"], label="Ground Truth", zorder=3)
        ax.plot(timesteps, pd_, color=COLORS["pred"], linestyle="--", label="Prediction", zorder=3)
        ax.fill_between(timesteps, pd_ - err * 0.5, pd_ + err * 0.5,
                         color=COLORS["fill"], alpha=0.12, label="Error band", zorder=1)
        ax.set_ylabel("Value")
        if row == 0:
            ax.legend(loc="upper right", ncol=3)
        ax.text(0.98, 0.92, f"Sample #{idx}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, style="italic", color="0.4")

    axes[-1].set_xlabel("Prediction Horizon (time steps)")

    title_parts = []
    if "model" in info: title_parts.append(info["model"])
    if "dataset" in info: title_parts.append(info["dataset"])
    if "pred_len" in info: title_parts.append(f"H={info['pred_len']}")
    fig.suptitle(" — ".join(title_parts) if title_parts else "Prediction Curves", fontweight="bold")
    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    _save_fig(fig, folder_path, "fig_prediction_curves.png")


# ──────────────────────────────────────────────
# Figure 2: 誤差分析
# ──────────────────────────────────────────────
def fig_error_analysis(folder_path: str, feature_idx: int = -1):
    preds, trues, _ = _load_results(folder_path)
    fi = _safe_feature_idx(preds, trues, feature_idx)
    info = _parse_setting(os.path.basename(folder_path.rstrip(os.sep)))
    pred_len = preds.shape[1]

    errors = (preds[:, :, fi] - trues[:, :, fi]).flatten()
    mse_per_step = np.mean((preds[:, :, fi] - trues[:, :, fi]) ** 2, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5))

    norm_vals = mse_per_step / (mse_per_step.max() + 1e-12)
    cmap = plt.cm.YlOrRd
    bar_colors = cmap(norm_vals * 0.7 + 0.15)
    ax1.bar(range(pred_len), mse_per_step, color=bar_colors, edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("Prediction Step")
    ax1.set_ylabel("MSE")
    ax1.set_title("(a) MSE per Horizon Step", fontsize=10)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

    ax2.hist(errors, bins=80, density=True, color=COLORS["fill"], alpha=0.7, edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color=COLORS["pred"], linewidth=1.2, linestyle="--", alpha=0.8)
    mu, sigma = errors.mean(), errors.std()
    ax2.set_xlabel("Prediction Error")
    ax2.set_ylabel("Density")
    ax2.set_title(f"(b) Error Distribution ($\\mu$={mu:.4f}, $\\sigma$={sigma:.4f})", fontsize=10)

    fig.suptitle(f"Error Analysis — {info.get('model', '')}  {info.get('dataset', '')}", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    _save_fig(fig, folder_path, "fig_error_analysis.png")


# ──────────────────────────────────────────────
# Figure 3: 雷達圖
# ──────────────────────────────────────────────
def fig_metrics_radar(folder_path: str):
    _, _, metrics = _load_results(folder_path)
    if metrics is None or len(metrics) < 6:
        print("  ⚠️  metrics.npy 不存在或長度不足，跳過雷達圖")
        return
    info = _parse_setting(os.path.basename(folder_path.rstrip(os.sep)))

    values = metrics[:6].astype(float).tolist()
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
    ax.set_title(f"{info.get('model', 'Model')} — {info.get('dataset', '')}", fontweight="bold", pad=18)

    text_lines = "  |  ".join(f"{n}: {v:.4f}" for n, v in zip(METRIC_NAMES, values))
    fig.text(0.5, 0.02, text_lines, ha="center", fontsize=7.5, color="0.35")
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    _save_fig(fig, folder_path, "fig_metrics_radar.png")


# ──────────────────────────────────────────────
# Figure 4: 多模型比較
# ──────────────────────────────────────────────
def fig_multi_model_comparison(folders=None, metric_indices=(0, 1)):
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
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Model Comparison", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    results_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    _save_fig(fig, results_base, "fig_model_comparison.png")


# ──────────────────────────────────────────────
# Figure 5: 誤差熱力圖
# ──────────────────────────────────────────────
def fig_error_heatmap(folder_path: str, feature_idx: int = -1, max_samples: int = 200):
    preds, trues, _ = _load_results(folder_path)
    fi = _safe_feature_idx(preds, trues, feature_idx)
    info = _parse_setting(os.path.basename(folder_path.rstrip(os.sep)))

    abs_err = np.abs(preds[:, :, fi] - trues[:, :, fi])
    if abs_err.shape[0] > max_samples:
        step = abs_err.shape[0] // max_samples
        abs_err = abs_err[::step][:max_samples]

    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    im = ax.imshow(abs_err, aspect="auto", cmap="YlOrRd", interpolation="nearest", origin="lower")
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Sample Index")
    ax.set_title(f"Absolute Error Heatmap — {info.get('model', '')}  {info.get('dataset', '')}", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, pad=0.02, aspect=30)
    cb.set_label("| Pred − True |", fontsize=9)
    plt.tight_layout()

    _save_fig(fig, folder_path, "fig_error_heatmap.png")


# ──────────────────────────────────────────────
# Figure 6: Dashboard
# ──────────────────────────────────────────────
def fig_dashboard(folder_path: str, feature_idx: int = -1):
    preds, trues, metrics = _load_results(folder_path)
    fi = _safe_feature_idx(preds, trues, feature_idx)
    info = _parse_setting(os.path.basename(folder_path.rstrip(os.sep)))
    pred_len = preds.shape[1]
    n_total = preds.shape[0]

    fig = plt.figure(figsize=(7.16, 6.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, :])
    mid_idx = n_total // 2
    gt = trues[mid_idx, :, fi]
    pd_ = preds[mid_idx, :, fi]
    t = np.arange(pred_len)
    ax_a.plot(t, gt, color=COLORS["gt"], label="Ground Truth")
    ax_a.plot(t, pd_, color=COLORS["pred"], linestyle="--", label="Prediction")
    ax_a.fill_between(t, gt, pd_, color=COLORS["fill"], alpha=0.12)
    ax_a.set_xlabel("Time Step")
    ax_a.set_ylabel("Value")
    ax_a.legend(loc="upper right", ncol=2)
    ax_a.set_title("(a) Prediction vs Ground Truth", fontsize=10, fontweight="bold")

    ax_b = fig.add_subplot(gs[1, 0])
    mse_step = np.mean((preds[:, :, fi] - trues[:, :, fi]) ** 2, axis=0)
    norm_v = mse_step / (mse_step.max() + 1e-12)
    cmap = plt.cm.YlOrRd
    ax_b.bar(range(pred_len), mse_step, color=cmap(norm_v * 0.7 + 0.15), edgecolor="white", linewidth=0.3)
    ax_b.set_xlabel("Prediction Step")
    ax_b.set_ylabel("MSE")
    ax_b.set_title("(b) MSE per Horizon", fontsize=10, fontweight="bold")

    ax_c = fig.add_subplot(gs[1, 1])
    errors = (preds[:, :, fi] - trues[:, :, fi]).flatten()
    ax_c.hist(errors, bins=80, density=True, color=COLORS["fill"], alpha=0.7, edgecolor="white", linewidth=0.3)
    ax_c.axvline(0, color=COLORS["pred"], linewidth=1, linestyle="--")
    mu, sigma = errors.mean(), errors.std()
    ax_c.set_xlabel("Error")
    ax_c.set_ylabel("Density")
    ax_c.set_title(f"(c) Error Dist. ($\\mu$={mu:.3f}, $\\sigma$={sigma:.3f})", fontsize=10, fontweight="bold")

    title = f"{info.get('model', 'Model')} — {info.get('dataset', '')} — H={info.get('pred_len', '?')}"
    fig.suptitle(title, fontsize=12, fontweight="bold")

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
        if choice < 0 or choice >= len(dirs):
            print(f"❌ 索引超出範圍 (0~{len(dirs)-1})")
            raise SystemExit(1)
        selected = dirs[choice]
        print(f"📁 選擇: {selected}")
        return selected
    except ValueError:
        print("❌ 請輸入數字")
        raise SystemExit(1)


def _run_all_figures(folder: str, feature_idx: int, n_samples: int):
    """對單一資料夾產出所有圖表，每張圖獨立 try/except"""
    figures = [
        ("prediction_curves", lambda: fig_prediction_curves(folder, feature_idx=feature_idx, n_samples=n_samples)),
        ("error_analysis",    lambda: fig_error_analysis(folder, feature_idx=feature_idx)),
        ("metrics_radar",     lambda: fig_metrics_radar(folder)),
        ("error_heatmap",     lambda: fig_error_heatmap(folder, feature_idx=feature_idx)),
        ("dashboard",         lambda: fig_dashboard(folder, feature_idx=feature_idx)),
    ]
    success = 0
    for fig_name, fig_func in figures:
        try:
            print(f"\n🎨 繪製 {fig_name}...")
            fig_func()
            success += 1
        except Exception as e:
            print(f"  ❌ {fig_name} 失敗: {e}")
            traceback.print_exc()
    print(f"\n📊 完成 {success}/{len(figures)} 張圖 → {os.path.abspath(folder)}")


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

    if args.batch:
        folders = list_result_folders()
        if not folders:
            return
        for i, folder in enumerate(folders):
            name = os.path.basename(folder.rstrip(os.sep))
            print(f"\n{'='*60}")
            print(f"📊 [{i+1}/{len(folders)}] {name}")
            print(f"{'='*60}")
            _run_all_figures(folder, args.feature, args.n_samples)
        try:
            fig_multi_model_comparison()
        except Exception as e:
            print(f"❌ 多模型比較失敗: {e}")
        print(f"\n🎉 批次完成！共處理 {len(folders)} 個結果資料夾")
        return

    folder = args.folder or _interactive_select()

    if args.dashboard:
        try:
            fig_dashboard(folder, feature_idx=args.feature)
        except Exception as e:
            print(f"❌ Dashboard 失敗: {e}")
            traceback.print_exc()
        return

    _run_all_figures(folder, args.feature, args.n_samples)

    if args.all:
        try:
            fig_multi_model_comparison()
        except Exception as e:
            print(f"❌ 多模型比較失敗: {e}")


if __name__ == "__main__":
    main()