# Afternoon pipeline (14:00-18:00, CPU only, zero LLM cost)
# 1) wait for existing training  2) BTC stratification  3) experiment queue (deadline guarded)
# 4) stratification analyses  5) generate digest for the 18:00 wake
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$py = ".venv\Scripts\python.exe"
$deadline = [datetime]"2026-09-11T17:40:00"
$log = "afternoon_batch_log.txt"

function Say($m) { "$m" | Tee-Object -FilePath $log -Append }

Say "=== afternoon batch start $(Get-Date -Format HH:mm:ss) ==="

# 1) wait until no run.py python process is alive (avoid CPU contention)
while ($true) {
  $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
       Where-Object { $_.CommandLine -like "*run.py*" }
  if (-not $p) { break }
  Start-Sleep -Seconds 60
}
Say "existing training finished $(Get-Date -Format HH:mm:ss)"

# 2) BTCUSD stratification (needs both models done)
& $py -W ignore reproduction/analysis/analyze_volatility_regimes.py --csv dataset/BTCUSD-2016-2025.csv --tag BTCUSD --seed 2021 --rv-mode relative 2>&1 | Out-Null
& $py -W ignore reproduction/analysis/analyze_volatility_regimes.py --csv dataset/BTCUSD-2016-2025.csv --tag BTCUSD --seed 2021 --rv-mode absolute 2>&1 | Out-Null
Say "BTCUSD stratification done $(Get-Date -Format HH:mm:ss)"

# 3) experiment queue, priority ordered
$jobs = @(
  @{mid="GSPC_96_24";   model="DeReFusion-gatev1-volatilityaware"; data="GSPC-2016-2025.csv";   seed=2021},
  @{mid="GSPC_96_24";   model="DeReFusion";                       data="GSPC-2016-2025.csv";   seed=2022},
  @{mid="GSPC_96_24";   model="revin-DLinear";                    data="GSPC-2016-2025.csv";   seed=2022},
  @{mid="ETHUSD_96_24"; model="DeReFusion";                       data="ETHUSD-2016-2025.csv"; seed=2021},
  @{mid="ETHUSD_96_24"; model="revin-DLinear";                    data="ETHUSD-2016-2025.csv"; seed=2021},
  @{mid="GSPC_96_24";   model="DeReFusion";                       data="GSPC-2016-2025.csv";   seed=2023},
  @{mid="GSPC_96_24";   model="revin-DLinear";                    data="GSPC-2016-2025.csv";   seed=2023}
)

foreach ($j in $jobs) {
  if ((Get-Date) -gt $deadline) { Say "deadline reached, skipping remaining jobs"; break }
  Say "==== START $($j.model) seed$($j.seed) $(Get-Date -Format HH:mm:ss) ===="
  & $py run.py --task_name long_term_forecast --is_training 1 --model_id $($j.mid) --model $($j.model) `
    --data custom --root_path ./dataset/ --data_path $($j.data) `
    --features MS --target Close --freq b --seq_len 96 --label_len 48 --pred_len 24 `
    --enc_in 4 --dec_in 4 --c_out 1 --d_model 32 --moving_avg 25 `
    --train_epochs 30 --batch_size 32 --learning_rate 0.0001 --patience 5 --lradj cosine `
    --rand_seed $($j.seed) --no_use_gpu 2>&1 | Select-Object -Last 3 | ForEach-Object { Say $_ }
  Say "==== END $($j.model) seed$($j.seed) $(Get-Date -Format HH:mm:ss) ===="
}

# 4) stratification analyses for all settings
$analyses = @(
  @{csv="dataset/GSPC-2016-2025.csv";   tag="GSPC";   seed=2021; mode="relative"},
  @{csv="dataset/GSPC-2016-2025.csv";   tag="GSPC";   seed=2021; mode="absolute"},
  @{csv="dataset/GSPC-2016-2025.csv";   tag="GSPC";   seed=2022; mode="relative"},
  @{csv="dataset/GSPC-2016-2025.csv";   tag="GSPC";   seed=2023; mode="relative"},
  @{csv="dataset/ETHUSD-2016-2025.csv"; tag="ETHUSD"; seed=2021; mode="relative"},
  @{csv="dataset/BTCUSD-2016-2025.csv"; tag="BTCUSD"; seed=2021; mode="relative"}
)
foreach ($a in $analyses) {
  Say "---- analysis $($a.tag) s$($a.seed) $($a.mode) $(Get-Date -Format HH:mm:ss)"
  & $py -W ignore reproduction/analysis/analyze_volatility_regimes.py --csv $a.csv --tag $a.tag --seed $a.seed --rv-mode $a.mode 2>&1 | Select-Object -Last 1 | ForEach-Object { Say $_ }
}

# 5) digest for the 18:00 wake
& $py reproduction/analysis/make_summary.py 2>&1 | ForEach-Object { Say $_ }
Say "ALL DONE $(Get-Date -Format HH:mm:ss)"
