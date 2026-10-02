# BTCUSD T=24 跨市场复核：DeReFusion + revin-DLinear
# 协议与 GSPC 实验完全一致（仅 data_path / model_id 不同）
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$py = ".venv\Scripts\python.exe"

$jobs = @(
  @{mid="BTCUSD_96_24"; model="DeReFusion"},
  @{mid="BTCUSD_96_24"; model="revin-DLinear"}
)

foreach ($j in $jobs) {
  Write-Output "==== START $($j.model) BTC T=24 $(Get-Date -Format HH:mm:ss) ===="
  & $py run.py --task_name long_term_forecast --is_training 1 --model_id $j.mid --model $j.model `
    --data custom --root_path ./dataset/ --data_path BTCUSD-2016-2025.csv `
    --features MS --target Close --freq b --seq_len 96 --label_len 48 --pred_len 24 `
    --enc_in 4 --dec_in 4 --c_out 1 --d_model 32 --moving_avg 25 `
    --train_epochs 30 --batch_size 32 --learning_rate 0.0001 --patience 5 --lradj cosine `
    --rand_seed 2021 --no_use_gpu 2>&1 | Select-Object -Last 4
  Write-Output "==== END $($j.model) BTC T=24 $(Get-Date -Format HH:mm:ss) ===="
}
Write-Output "BTC ALL DONE"