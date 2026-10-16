# Start scripts/run_queue.py detached, so the queue keeps running after this shell (or the
# Claude session that called it) is closed. Safe to run again: finished jobs are skipped and
# interrupted ones resume. Stop cleanly with:  New-Item runs\STOP
param(
    [string]$DataRoot = "C:\Users\bapti\Documents\Unreal Projects\SyMTRS\Saved\DatasetCaptures",
    [string]$Queue = "runs\queue.txt",
    [string]$OutRoot = "runs\validation"
)
$repo = Split-Path -Parent $PSScriptRoot
Remove-Item (Join-Path $repo "runs\STOP") -ErrorAction SilentlyContinue
$py = Join-Path $repo ".venv\Scripts\python.exe"
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$out = Join-Path $repo "logs\run_queue_$stamp.out"
$p = Start-Process -FilePath $py -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError "$out.err" `
    -ArgumentList @("scripts\run_queue.py", "--queue", $Queue, "--out-root", $OutRoot, "--data-root", "`"$DataRoot`"")
"queue started, pid $($p.Id), output $out"
$p.Id | Out-File -Encoding ascii (Join-Path $repo "logs\run_queue.pid")
