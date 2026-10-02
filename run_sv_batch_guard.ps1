# Self-healing launcher for the structural-validation batch.
# Safe to run repeatedly (e.g. from a 5-minute scheduled task): it does nothing while a
# training run is already in flight, and the underlying batch script skips completed runs.
$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $RepoRoot

$runs = (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
         Where-Object { $_.CommandLine -like '*run.py*' } | Measure-Object).Count
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

if ($runs -gt 0) {
    " [guard] $stamp  skip: $runs run(s) already in flight" | Out-File -Append -Encoding utf8 sv_batch_log.txt
    exit 0
}

" [guard] $stamp  no run in flight -> resuming batch" | Out-File -Append -Encoding utf8 sv_batch_log.txt
& (Join-Path $RepoRoot "reproduction\batches\run_structural_validation.ps1")
