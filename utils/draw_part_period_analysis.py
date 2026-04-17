import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from datetime import datetime, timedelta
import sys
import os

def plot_regime(true_path, index_name="Index", save_path=None, 
                start_date="2024-01-01", rolling_window=20):
    """
    Plot test period price series with bullish/bearish regime shading.
    
    Args:
        true_path: path to true.npy from TSLib
        index_name: name for the title
        save_path: output file path
        start_date: approximate start date of test period (for x-axis)
        rolling_window: window for regime classification
    """
    true = np.load(true_path).squeeze()
    N = len(true)
    
    # Generate approximate trading dates (skip weekends)
    dates = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    while len(dates) < N:
        if current.weekday() < 5:  # Mon-Fri
            dates.append(current)
        current += timedelta(days=1)
    
    # Classify regime
    regime = np.zeros(N, dtype=int)  # 0=bull, 1=bear
    for i in range(N):
        start = max(0, i - rolling_window)
        if true[i] < true[start]:
            regime[i] = 1
    
    n_bull = (regime == 0).sum()
    n_bear = (regime == 1).sum()
    
    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    # Shade regimes
    i = 0
    while i < N:
        j = i
        while j < N and regime[j] == regime[i]:
            j += 1
        color = '#E8F5E9' if regime[i] == 0 else '#FFEBEE'  # light green / light red
        ax.axvspan(dates[i], dates[min(j, N-1)], 
                   alpha=0.7, color=color, linewidth=0)
        i = j
    
    # Plot price line
    ax.plot(dates, true, color='#1565C0', linewidth=1.2, zorder=3)
    
    # Formatting
    ax.set_title(f'{index_name} — Test Period with Market Regime Classification', 
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Normalized Price', fontsize=10)
    
    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    # Legend
    legend_elements = [
        Patch(facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=0.8,
              label=f'Bullish (n={n_bull})'),
        Patch(facecolor='#FFEBEE', edgecolor='#F44336', linewidth=0.8,
              label=f'Bearish (n={n_bear})'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
             framealpha=0.9, edgecolor='#CCCCCC')
    
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim(dates[0], dates[-1])
    
    # Add regime detection annotation
    ax.text(0.98, 0.02, f'Regime: 20-day rolling change', 
            transform=ax.transAxes, fontsize=8, color='#666666',
            ha='right', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Saved to {save_path}")
    
    plt.close()
    return fig


# ---- Run for the uploaded data ----
if __name__ == "__main__":
    true_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/twcch/Cloud Drive/Dev/Research/DeReFusion/results/seed2/long_term_forecast_DJI-2016-2025_96_48_1_DyVolFusion_custom_ftMS_sl96_ll48_pl1_dm32_nh2_el2_dl1_df64_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/true.npy"
    index_name = sys.argv[2] if len(sys.argv) > 2 else "DJI"
    save_path = sys.argv[3] if len(sys.argv) > 3 else "regime_plot.png"
    start_date = sys.argv[4] if len(sys.argv) > 4 else "2024-01-02"
    
    plot_regime(true_path, index_name, save_path, start_date)
    print("Done!")