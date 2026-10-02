# Structural validation: framework-level runs for the four pre-registered new assets.
# 4 assets x (DeReFusion, revin-DLinear), T=24, seed 2021 -- protocol identical to the ten
# existing assets (same flags as run_asset_sweep.ps1). Two runs in parallel; per-run logs.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$py = ".venv\Scripts\python.exe"
$log = "sv_batch_log.txt"
$runDir = "sweep_runs"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

function Say($m) { "$m" | Tee-Object -FilePath $log -Append }
Say "=== structural-validation batch start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

$assets = @("BYD", "BOE", "EASTMONEY", "YANGHE")
$models = @("DeReFusion", "revin-DLinear")
$jobs = @()
foreach ($a in $assets) {
  foreach ($m in $models) {
    $needle = "long_term_forecast_${a}_96_24_${m}_custom_ftMS_"
    $done = Get-ChildItem results -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "$needle*" -and $_.Name -like "*seed2021*" }
    if ($done) { Say "SKIP $a $m (done)"; continue }
    $jobs += [pscustomobject]@{ tag = $a; model = $m; out = "$runDir\sv_${a}_$($m).log" }
  }
}
Say "queued $($jobs.Count) runs"

$running = @()
foreach ($j in $jobs) {
  $args = @("run.py", "--task_name", "long_term_forecast", "--is_training", "1",
            "--model_id", "$($j.tag)_96_24", "--model", $j.model,
            "--data", "custom", "--root_path", "./dataset/", "--data_path", "$($j.tag)-2016-2025.csv",
            "--features", "MS", "--target", "Close", "--freq", "b",
            "--seq_len", "96", "--label_len", "48", "--pred_len", "24",
            "--enc_in", "4", "--dec_in", "4", "--c_out", "1",
            "--d_model", "32", "--moving_avg", "25",
            "--train_epochs", "30", "--batch_size", "32", "--learning_rate", "0.0001",
            "--patience", "5", "--lradj", "cosine", "--rand_seed", "2021", "--no_use_gpu")
  Say "==== START $($j.tag) $($j.model) $(Get-Date -Format HH:mm:ss) ===="
  $p = Start-Process -FilePath $py -ArgumentList $args -NoNewWindow -PassThru `
        -RedirectStandardOutput $j.out -RedirectStandardError "$($j.out).err"
  $running += [pscustomobject]@{ proc = $p; tag = $j.tag; model = $j.model }
  while ((@($running | Where-Object { -not $_.proc.HasExited })).Count -ge 2) { Start-Sleep -Seconds 20 }
  $done = @($running | Where-Object { $_.proc.HasExited })
  foreach ($d in $done) { Say "==== END $($d.tag) $($d.model) $(Get-Date -Format HH:mm:ss) ====" }
  $running = @($running | Where-Object { -not $_.proc.HasExited })
}
while ((@($running | Where-Object { -not $_.proc.HasExited })).Count -gt 0) { Start-Sleep -Seconds 20 }
foreach ($d in $running) { Say "==== END $($d.tag) $($d.model) $(Get-Date -Format HH:mm:ss) ====" }

Say "---- stratifications $(Get-Date -Format HH:mm:ss)"
foreach ($a in $assets) {
  & $py -W ignore reproduction/analysis/analyze_volatility_regimes.py --csv "dataset/$a-2016-2025.csv" --tag $a --seed 2021 --rv-mode relative 2>&1 |
    Select-Object -Last 1 | ForEach-Object { Say $_ }
}
Say "ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
