@echo off
REM Entry point referenced by the scheduled task "sv_batch_resume".
REM Delegates to the guard, which (a) exits if a training run is already in flight and
REM (b) otherwise resumes the idempotent structural-validation batch.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_sv_batch_guard.ps1"
