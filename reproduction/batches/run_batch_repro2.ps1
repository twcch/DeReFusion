# 批量对比实验 v2：exp_basic emoji 已修复
# 需跑: DeReFusion T=1; revin-DLinear T={1,24}; gatev2-learnable T={1,24}
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$py = ".venv\Scripts\python.exe"

$jobs = @(
  @{mid="GSPC_96_1";  model="DeReFusion";                 pl="1"},
  @{mid="GSPC_96_24"; model="revin-DLinear";              pl="24"},
  @{mid="GSPC_96_1";  model="revin-DLinear";              pl="1"},
  @{mid="GSPC_96_24"; model="DeReFusion-gatev2-learnable"; pl="24"},
  @{mid="GSPC_96_1";  model="DeReFusion-gatev2-learnable"; pl="1"}
)

foreach ($j in $jobs) {
  Write-Output "==== START $($j.model) T=$($j.pl) $(Get-Date -Format HH:mm:ss) ===="
  & $py run.py --task_name long_term_forecast --is_training 1 --model_id $j.mid --model $j.model `
    --data custom --root_path ./dataset/ --data_path GSPC-2016-2025.csv `
    --features MS --target Close --freq b --seq_len 96 --label_len 48 --pred_len $j.pl `
    --enc_in 4 --dec_in 4 --c_out 1 --d_model 32 --moving_avg 25 `
    --train_epochs 30 --batch_size 32 --learning_rate 0.0001 --patience 5 --lradj cosine `
    --rand_seed 2021 --no_use_gpu 2>&1 | Select-Object -Last 4
  Write-Output "==== END $($j.model) T=$($j.pl) ===="
}
Write-Output "ALL DONE"
# 结束后把结果日志拷贝一份快照
Copy-Item result_long_term_forecast.txt repro_results_snapshot.txt -Force
Write-Output "SNAPSHOT SAVED"