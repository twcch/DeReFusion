import numpy as np
import sys

def sub_period_analysis(pred_path, true_path, index_name="Index", rolling_window=20):
    pred = np.load(pred_path).squeeze()  # (N,)
    true = np.load(true_path).squeeze()  # (N,)
    N = len(true)
    
    # ---- Metrics ----
    def calc_metrics(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        # MAPE (avoid division by zero)
        mask = np.abs(y_true) > 1e-8
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    # ---- Define Bull / Bear regime ----
    # rolling_window-day change in true values
    regime = np.zeros(N, dtype=int)  # 0=bull, 1=bear
    for i in range(N):
        start = max(0, i - rolling_window)
        if true[i] < true[start]:
            regime[i] = 1  # bear
    
    bull_mask = regime == 0
    bear_mask = regime == 1
    
    # ---- Calculate ----
    overall = calc_metrics(true, pred)
    bull = calc_metrics(true[bull_mask], pred[bull_mask])
    bear = calc_metrics(true[bear_mask], pred[bear_mask])
    
    # ---- Print results ----
    print(f"\n{'='*70}")
    print(f"  Sub-Period Analysis: {index_name}")
    print(f"  Rolling window for regime detection: {rolling_window} trading days")
    print(f"{'='*70}")
    print(f"\n  {'Regime':<12} {'#Samples':>10} {'MSE':>12} {'MAE':>12} {'RMSE':>12} {'MAPE(%)':>12}")
    print(f"  {'-'*58}")
    
    for name, mask, metrics in [
        ('Overall', np.ones(N, dtype=bool), overall),
        ('Bullish', bull_mask, bull),
        ('Bearish', bear_mask, bear),
    ]:
        n = mask.sum()
        print(f"  {name:<12} {n:>10} {metrics['MSE']:>12.6f} {metrics['MAE']:>12.6f} {metrics['RMSE']:>12.6f} {metrics['MAPE']:>12.4f}")
    
    print()
    
    # ---- Direction accuracy (bonus) ----
    if N > 1:
        true_dir = np.diff(true) > 0
        pred_dir = np.diff(pred) > 0
        dir_acc_overall = np.mean(true_dir == pred_dir) * 100
        
        # Direction accuracy by regime (use regime[1:] since diff reduces length by 1)
        bull_dir_mask = regime[1:] == 0
        bear_dir_mask = regime[1:] == 1
        
        if bull_dir_mask.sum() > 0:
            dir_acc_bull = np.mean(true_dir[bull_dir_mask] == pred_dir[bull_dir_mask]) * 100
        else:
            dir_acc_bull = float('nan')
        if bear_dir_mask.sum() > 0:
            dir_acc_bear = np.mean(true_dir[bear_dir_mask] == pred_dir[bear_dir_mask]) * 100
        else:
            dir_acc_bear = float('nan')
        
        print(f"  Direction Accuracy:")
        print(f"    Overall:  {dir_acc_overall:.2f}%")
        print(f"    Bullish:  {dir_acc_bull:.2f}%")
        print(f"    Bearish:  {dir_acc_bear:.2f}%")
        print()
    
    return {
        'index': index_name,
        'overall': overall, 'bull': bull, 'bear': bear,
        'n_bull': int(bull_mask.sum()), 'n_bear': int(bear_mask.sum()),
    }


if __name__ == "__main__":
    # Default: run with uploaded files
    pred_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/twcch/Cloud Drive/Dev/Research/DeReFusion/results/seed2/long_term_forecast_GSPC-2016-2025_96_48_1_TimesNet_custom_ftMS_sl96_ll48_pl1_dm32_nh2_el2_dl1_df64_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/pred.npy"
    true_path = sys.argv[2] if len(sys.argv) > 2 else "/Users/twcch/Cloud Drive/Dev/Research/DeReFusion/results/seed2/long_term_forecast_GSPC-2016-2025_96_48_1_TimesNet_custom_ftMS_sl96_ll48_pl1_dm32_nh2_el2_dl1_df64_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/true.npy"
    index_name = sys.argv[3] if len(sys.argv) > 3 else "Unknown"
    
    sub_period_analysis(pred_path, true_path, index_name)
    