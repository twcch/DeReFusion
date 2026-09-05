"""
Publication-quality visualization for time-series forecasting results.

Integrated into the FinTSLib pipeline — called automatically after test()
saves .npy files, or usable standalone:

    python -m utils.visualization --input results/xxx/ --output figures/
    python -m utils.visualization --input results/xxx/ --output figures/ --dashboard
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import os
import re
import traceback

# Use non-interactive backend (same as utils/tools.py)
matplotlib.use('Agg')

_JOURNAL_STYLE = {
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
}

COLORS = {
    "gt":   "#2c3e50",
    "pred": "#e74c3c",
    "fill": "#3498db",
    "bar1": "#2980b9",
}

METRIC_NAMES = ["MAE", "MSE", "RMSE", "MAPE", "MSPE", "R²"]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _load_results(folder_path: str):
    """Load pred.npy, true.npy, metrics.npy with shape validation."""
    folder_path = os.path.abspath(folder_path)

    pred_path = os.path.join(folder_path, "pred.npy")
    true_path = os.path.join(folder_path, "true.npy")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"pred.npy not found: {pred_path}")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"true.npy not found: {true_path}")

    preds = np.load(pred_path)
    trues = np.load(true_path)

    print(f"  pred.shape={preds.shape}, true.shape={trues.shape}")

    if preds.shape[-1] != trues.shape[-1]:
        print(f"  Warning: feature dim mismatch "
              f"(pred={preds.shape[-1]}, true={trues.shape[-1]}), "
              f"using min")

    metrics = None
    mp = os.path.join(folder_path, "metrics.npy")
    if os.path.exists(mp):
        metrics = np.load(mp, allow_pickle=True)
        print(f"  metrics={metrics}")
    else:
        print(f"  Warning: metrics.npy not found")

    return preds, trues, metrics


def _safe_feature_idx(preds, trues, feature_idx: int) -> int:
    min_features = min(preds.shape[-1], trues.shape[-1])
    if feature_idx < 0:
        feature_idx = min_features + feature_idx
    if feature_idx < 0 or feature_idx >= min_features:
        print(f"  Warning: feature_idx={feature_idx} out of range (max={min_features-1}), using 0")
        feature_idx = 0
    return feature_idx


def _ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _save_fig(fig, folder_path: str, filename: str):
    folder_path = os.path.abspath(folder_path)
    _ensure_dir(folder_path)
    fp = os.path.join(folder_path, filename)
    fig.savefig(fp, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    if os.path.exists(fp):
        print(f"  Saved: {fp}  ({os.path.getsize(fp)} bytes)")
    else:
        print(f"  Failed to save: {fp}")
    plt.close(fig)


def _parse_setting(folder_name: str) -> dict:
    info = {"raw": folder_name}
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


# ──────────────────────────────────────────────
# Figure 1: Prediction Curves
# ──────────────────────────────────────────────
def fig_prediction_curves(input_path: str, output_path: str, feature_idx: int = -1, n_samples: int = 3):
    preds, trues, _ = _load_results(input_path)
    fi = _safe_feature_idx(preds, trues, feature_idx)
    info = _parse_setting(os.path.basename(input_path.rstrip(os.sep)))
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

    _save_fig(fig, output_path, "fig_prediction_curves.png")


# ──────────────────────────────────────────────
# Figure 2: Error Analysis
# ──────────────────────────────────────────────
def fig_error_analysis(input_path: str, output_path: str, feature_idx: int = -1):
    preds, trues, _ = _load_results(input_path)
    fi = _safe_feature_idx(preds, trues, feature_idx)
    info = _parse_setting(os.path.basename(input_path.rstrip(os.sep)))
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

    _save_fig(fig, output_path, "fig_error_analysis.png")


# ──────────────────────────────────────────────
# Figure 3: Metrics Radar
# ──────────────────────────────────────────────
def fig_metrics_radar(input_path: str, output_path: str):
    _, _, metrics = _load_results(input_path)
    if metrics is None or len(metrics) < 6:
        print("  Warning: metrics.npy missing or insufficient, skipping radar chart")
        return
    info = _parse_setting(os.path.basename(input_path.rstrip(os.sep)))

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

    _save_fig(fig, output_path, "fig_metrics_radar.png")


# ──────────────────────────────────────────────
# Figure 4: Error Heatmap
# ──────────────────────────────────────────────
def fig_error_heatmap(input_path: str, output_path: str, feature_idx: int = -1, max_samples: int = 200):
    preds, trues, _ = _load_results(input_path)
    fi = _safe_feature_idx(preds, trues, feature_idx)
    info = _parse_setting(os.path.basename(input_path.rstrip(os.sep)))

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
    cb.set_label("| Pred - True |", fontsize=9)
    plt.tight_layout()

    _save_fig(fig, output_path, "fig_error_heatmap.png")


# ──────────────────────────────────────────────
# Figure 5: Full Pred vs True
# ──────────────────────────────────────────────
def fig_pred_true(input_path: str, output_path: str, feature_idx: int = -1):
    """Concatenate first time step of each prediction window into a continuous curve."""
    preds, trues, metrics = _load_results(input_path)
    fi = _safe_feature_idx(preds, trues, feature_idx)
    info = _parse_setting(os.path.basename(input_path.rstrip(os.sep)))

    pred_vals = preds[:, 0, fi]
    true_vals = trues[:, 0, fi]
    timesteps = np.arange(len(true_vals))

    fig, ax = plt.subplots(figsize=(7.16, 2.8))

    ax.plot(timesteps, true_vals, color=COLORS["gt"], label="Ground Truth", linewidth=1.5, zorder=3)
    ax.plot(timesteps, pred_vals, color=COLORS["pred"], linestyle="--", label="Prediction", linewidth=1.5, zorder=3)
    ax.fill_between(timesteps, true_vals, pred_vals, color=COLORS["fill"], alpha=0.10, zorder=1)

    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left", ncol=2)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

    title_parts = []
    if "model" in info:
        title_parts.append(info["model"])
    if "dataset" in info:
        title_parts.append(info["dataset"])
    if "pred_len" in info:
        title_parts.append(f"H={info['pred_len']}")
    ax.set_title(" — ".join(title_parts) if title_parts else "Prediction vs Ground Truth", fontweight="bold")

    if metrics is not None and len(metrics) >= 6:
        vals = metrics[:6].astype(float)
        txt = "  |  ".join(f"{n}: {v:.4f}" for n, v in zip(METRIC_NAMES, vals))
        fig.text(0.5, 0.005, txt, ha="center", fontsize=7.5, color="0.4")
        plt.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        plt.tight_layout()

    _save_fig(fig, output_path, "fig_pred_true.png")


# ──────────────────────────────────────────────
# Figure 6: Dashboard
# ──────────────────────────────────────────────
def fig_dashboard(input_path: str, output_path: str, feature_idx: int = -1):
    preds, trues, metrics = _load_results(input_path)
    fi = _safe_feature_idx(preds, trues, feature_idx)
    info = _parse_setting(os.path.basename(input_path.rstrip(os.sep)))
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

    _save_fig(fig, output_path, "fig_dashboard.png")


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def generate_figures(input_path: str, output_path: str, feature_idx: int = -1, n_samples: int = 3):
    """Generate all publication-quality figures from .npy results.

    Called automatically by exp classes after test(), or manually:
        from utils.visualization import generate_figures
        generate_figures('./results/setting/', './test_results/setting/')

    Args:
        input_path: Folder containing pred.npy, true.npy, metrics.npy
        output_path: Folder where PNG figures will be saved
        feature_idx: Feature index to plot (-1 = last)
        n_samples: Number of sample curves for prediction_curves plot
    """
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    if not os.path.isdir(input_path):
        print(f"Warning: generate_figures input path does not exist: {input_path}, skipping")
        return

    _ensure_dir(output_path)

    # Apply journal style only during figure generation
    original_rcparams = plt.rcParams.copy()
    plt.rcParams.update(_JOURNAL_STYLE)
    try:
        _run_all_figures(input_path, output_path, feature_idx, n_samples)
    finally:
        plt.rcParams.update(original_rcparams)


def _run_all_figures(input_path: str, output_path: str, feature_idx: int, n_samples: int):
    """Generate all figure types, each with independent error handling."""
    figures = [
        ("prediction_curves", lambda: fig_prediction_curves(input_path, output_path, feature_idx=feature_idx, n_samples=n_samples)),
        ("error_analysis",    lambda: fig_error_analysis(input_path, output_path, feature_idx=feature_idx)),
        ("metrics_radar",     lambda: fig_metrics_radar(input_path, output_path)),
        ("error_heatmap",     lambda: fig_error_heatmap(input_path, output_path, feature_idx=feature_idx)),
        ("pred_true",         lambda: fig_pred_true(input_path, output_path, feature_idx=feature_idx)),
        ("dashboard",         lambda: fig_dashboard(input_path, output_path, feature_idx=feature_idx)),
    ]
    success = 0
    for fig_name, fig_func in figures:
        try:
            print(f"\nGenerating {fig_name}...")
            fig_func()
            success += 1
        except Exception as e:
            print(f"  {fig_name} failed: {e}")
            traceback.print_exc()
    print(f"\nFigures complete: {success}/{len(figures)} -> {os.path.abspath(output_path)}")


# ──────────────────────────────────────────────
# CLI: python -m utils.visualization --input ... --output ...
# ──────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Publication-quality visualization for FinTSLib results",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Path to result folder (contains pred.npy, true.npy)")
    parser.add_argument("--output", type=str, default=None, help="Output folder for figures (default: same as input)")
    parser.add_argument("--feature", type=int, default=-1, help="Feature index (-1 = last)")
    parser.add_argument("--n_samples", type=int, default=3, help="Number of sample curves to plot")
    parser.add_argument("--dashboard", action="store_true", help="Generate dashboard only")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output) if args.output else input_path

    if not os.path.isdir(input_path):
        print(f"Error: input path does not exist: {input_path}")
        raise SystemExit(1)

    plt.rcParams.update(_JOURNAL_STYLE)

    if args.dashboard:
        try:
            fig_dashboard(input_path, output_path, feature_idx=args.feature)
        except Exception as e:
            print(f"Dashboard failed: {e}")
            traceback.print_exc()
        return

    _ensure_dir(output_path)
    _run_all_figures(input_path, output_path, args.feature, args.n_samples)


if __name__ == "__main__":
    main()
