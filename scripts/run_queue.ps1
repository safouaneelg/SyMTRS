# Start scripts/run_queue.py detached, so the queue keeps running after this shell (or the
# Claude session that called it) is closed. Safe to run again: finished jobs are skipped and
# interrupted ones resume. Stop cleanly with:  New-Item runs\STOP
#
# The dataset root is never hard-coded: pass -DataRoot <DATA_ROOT> or set SYMTRS_DATA_ROOT.
#   $env:SYMTRS_DATA_ROOT = "<DATA_ROOT>"; powershell -File scripts\run_queue.ps1
param(
    [string]$DataRoot = $env:SYMTRS_DATA_ROOT,
    [string]$Queue = "runs\queue.txt",
    [string]$OutRoot = "runs\validation"
)
if (-not $DataRoot) {
    Write-Error ("No dataset root. Pass -DataRoot <DATA_ROOT> or set the SYMTRS_DATA_ROOT " +
                 "environment variable to the folder that contains train/, val/ and test/.")
    exit 1
}
if (-not (Test-Path (Join-Path $DataRoot "train"))) {
    Write-Error "DataRoot '$DataRoot' has no train/ folder. Check -DataRoot / SYMTRS_DATA_ROOT."
    exit 1
}
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
