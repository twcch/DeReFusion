import subprocess
import sys
import time
import itertools
import os
import hashlib

# 進度記錄檔: 每行一個已成功完成的實驗識別碼，程式中斷後重跑會自動跳過已完成的
PROGRESS_FILE = ".run_batch_progress.log"


def _cmd_key(cmd):
    """用指令內容產生穩定的識別碼，用來判斷是否已跑過"""
    return hashlib.md5(cmd.encode("utf-8")).hexdigest()


def _load_done_keys():
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        return {line.split()[0] for line in f if line.strip()}


def _mark_done(key, cmd):
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{key}  # {cmd}\n")


def run_experiments():
    # 可傳入多個 data_name 和 pred_len，會自動跑所有組合
    # data: ["TSMC-2013-2023", "AAPL-2013-2023", "GSPC-2013-2023", "NDX-2013-2023", "SOX-2013-2023", "FTSE-2013-2023", "JPM-2013-2023", "N225-2013-2023"]
    # data: ["GSPC-2016-2025", "DJI-2016-2025", "SOX-2016-2025", "TSMC-2016-2025", "AAPL-2016-2025", "NVDA-2016-2025", "TSLA-2016-2025", "XOM-2016-2025"]
    data_names = ["GSPC-2016-2025", "DJI-2016-2025", "SOX-2016-2025", "TSMC-2016-2025", "AAPL-2016-2025", "NVDA-2016-2025", "TSLA-2016-2025", "XOM-2016-2025"]
    
    # [7, 12, 24, 36, 48, 60, 72, 84, 96, 30]
    seq_lens = [48, 96, 192]
    
    label_lens = [48]
    
    # short-term: [1, 7, 12, 24, 36, 48, 60, 72, 84]
    # long-term: [96, 192]
    pred_lens = [1, 7, 12, 24, 36]

    # 共用參數
    task_name = "long_term_forecast"
    root_path = "./dataset/2016_2025/"
    data = "custom"
    features = "MS"
    target = "Close"
    freq = "b"
    enc_in = 4
    dec_in = 4
    c_out = 1
    embed = "timeF"
    loss = "MSE"
    patience = 5
    dropout = 0.3
    des = "Exp"
    itr = 1
    use_norm = 1
    train_epochs = 30
    batch_size = 32
    learning_rate = 0.0001
    lradj = "cosine"
    rand_seeds = [2020, 2021, 2022, 2023, 2024]

    # 模型定義
    model_configs = [
        #{"model": "RevTransLSTM-AR", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        
        {"model": "DeReFusion", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        #{"model": "DeReFusion_woDy", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        #{"model": "DeReFusion_woLSTM", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        #{"model": "DeReFusion_woTransformer", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        #{"model": "DeReFusion_gatev1_volatilityaware", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        #{"model": "DeReFusion_gatev2_learnable", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        #{"model": "DeReFusion_gatev3_inputconditioned", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        
        # 已內建 RevIN (或等效的可逆正規化)
        # {"model": "PatchTST", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        # {"model": "iTransformer", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        # {"model": "TSMixer", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        # {"model": "PAttn", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        
        # 未內建 RevIN (或等效的可逆正規化)
        # {"model": "Autoformer", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        # {"model": "DLinear", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        # {"model": "TimesNet", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        # {"model": "FEDformer", "e_layers": 2, "d_layers": 2, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        # {"model": "ETSformer", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        # {"model": "Informer", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        # {"model": "RevINInformer", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        # {"model": "Reformer", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        # {"model": "LightTS", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        
        # {"model": "RevINTransformerEncoder", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        # {"model": "RevINTransformer", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},

        # Foundation models
        # {"model": "TimesFM", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        # {"model": "Chronos", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        # {"model": "Moirai", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
    ]

    # 建立所有 (data_name, pred_len, model_config) 的組合
    commands = []
    for data_name, seq_len, label_len, pred_len, rand_seed in itertools.product(data_names, seq_lens, label_lens, pred_lens, rand_seeds):
        data_path = data_name + ".csv"
        base_args = (
            f"{sys.executable} -u run.py"
            f" --is_training 1"
            f" --use_dtw"
            f" --task_name {task_name}"
            f" --root_path {root_path}"
            f" --data_path {data_path}"
            f" --data {data}"
            f" --features {features}"
            f" --target {target}"
            f" --freq {freq}"
            f" --enc_in {enc_in}"
            f" --dec_in {dec_in}"
            f" --c_out {c_out}"
            f" --seq_len {seq_len}"
            f" --label_len {label_len}"
            f" --pred_len {pred_len}"
            f" --embed {embed}"
            f" --loss {loss}"
            f" --patience {patience}"
            f" --dropout {dropout}"
            f" --des {des}"
            f" --itr {itr}"
            f" --use_norm {use_norm}"
            f" --train_epochs {train_epochs}"
            f" --batch_size {batch_size}"
            f" --learning_rate {learning_rate}"
            f" --lradj {lradj}"
            f" --rand_seed {rand_seed}"
        )
        model_id = f"{data_name}_{seq_len}_{label_len}_{pred_len}"
        for cfg in model_configs:
            cmd = (
                f"{base_args}"
                f" --model_id {model_id}"
                f" --model {cfg['model']}"
                f" --e_layers {cfg['e_layers']}"
                f" --d_layers {cfg['d_layers']}"
                f" --factor {cfg['factor']}"
                f" --d_model {cfg['d_model']}"
                f" --d_ff {cfg['d_ff']}"
                f" --n_heads {cfg['n_heads']}"
            )
            commands.append(cmd)

    total_experiments = len(commands)
    done_keys = _load_done_keys()
    print(f"任務啟動: 共計 {total_experiments} 個實驗準備執行...")
    if done_keys:
        print(f"偵測到進度記錄: 已完成 {len(done_keys)} 個，將自動跳過。")
    print()

    for i, cmd in enumerate(commands):
        key = _cmd_key(cmd)
        if key in done_keys:
            print(f"⏭️  跳過實驗 [{i+1}/{total_experiments}] (已完成)")
            continue

        print("=" * 80)
        print(f"🚀 開始執行實驗 [{i+1}/{total_experiments}]")
        print(f"指令內容:\n{cmd}")
        print("=" * 80)

        start_time = time.time()

        try:
            # 呼叫系統終端機執行指令，這會將執行過程的 log 實時印在你的畫面上
            subprocess.run(cmd, shell=True, check=True)

            elapsed_time = time.time() - start_time
            _mark_done(key, cmd)
            print(f"\n✅ 實驗 [{i+1}/{total_experiments}] 成功執行完畢! 耗時: {elapsed_time/60:.2f} 分鐘\n")

            # 每個實驗跑完後休息 3 秒，讓 GPU 釋放記憶體
            time.sleep(3)

        except subprocess.CalledProcessError as e:
            # 如果某個實驗發生 Bug 崩潰，會捕捉錯誤並印出，然後直接停止後續實驗
            print(f"\n❌ 實驗 [{i+1}/{total_experiments}] 發生錯誤而中斷!")
            print(f"系統回傳錯誤碼: {e.returncode}")
            print("批次任務已停止。")
            break
        except KeyboardInterrupt:
            # 讓你可以用 Ctrl+C 隨時中斷整個批次腳本
            print("\n🛑 接收到使用者中斷指令，批次任務結束。")
            break

if __name__ == "__main__":
    run_experiments()