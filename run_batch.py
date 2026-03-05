import subprocess
import time
import itertools

def run_experiments():
    # 可傳入多個 data_name 和 pred_len，會自動跑所有組合
    data_names = ["sp500"]
    pred_lens = [7, 12, 24, 48, 96]

    # 共用參數
    task_name = "long_term_forecast"
    root_path = "./dataset/yfinance/"
    data = "custom"
    features = "MS"
    target = "Close"
    freq = "b"
    enc_in = 5
    dec_in = 5
    c_out = 1
    seq_len = 96
    label_len = 48
    embed = "timeF"
    loss = "MSE"
    patience = 5
    dropout = 0.2
    des = "Exp"
    itr = 1
    use_norm = 1
    train_epochs = 30
    batch_size = 32
    learning_rate = 0.0001
    lradj = "cosine"

    # 模型定義（不含隨組合變動的 model_id）
    model_configs = [
        {"model": "DyVolFusion", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        {"model": "RevINTransformer", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        {"model": "RevINTransLSTM", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        {"model": "Autoformer", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        {"model": "DLinear", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        {"model": "iTransformer", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        {"model": "PatchTST", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        {"model": "PAttn", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        {"model": "TimesNet", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
        {"model": "TSMixer", "e_layers": 2, "d_layers": 1, "factor": 3, "d_model": 32, "d_ff": 64, "n_heads": 2},
    ]

    # 建立所有 (data_name, pred_len, model_config) 的組合
    commands = []
    for data_name, pred_len in itertools.product(data_names, pred_lens):
        data_path = data_name + ".csv"
        base_args = (
            f"python -u run.py"
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
    print(f"任務啟動: 共計 {total_experiments} 個實驗準備執行...\n")

    for i, cmd in enumerate(commands):
        print("=" * 80)
        print(f"🚀 開始執行實驗 [{i+1}/{total_experiments}]")
        print(f"指令內容:\n{cmd}")
        print("=" * 80)

        start_time = time.time()

        try:
            # 呼叫系統終端機執行指令，這會將執行過程的 log 實時印在你的畫面上
            subprocess.run(cmd, shell=True, check=True)

            elapsed_time = time.time() - start_time
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