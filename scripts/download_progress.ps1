# Live progress of the pre-downloads started by scripts\prefetch_all.ps1. Ctrl+C to quit
# (closing this window does NOT stop the downloads; they run in their own processes).
# The weights percentage compares the size on disk (.cache + checkpoints) with an ESTIMATED
# total of ~14 GB (MiDaS x5, ZoeDepth x3, DA3 x2, Depth Pro, SegFormer, DeepLab, Inception,
# shadow repos), so it is approximate and may end a little above or below 100 %.
param([double]$WeightsGB = 14, [int]$Every = 3)
$repo = Split-Path -Parent $PSScriptRoot

function SizeGB($paths) {
    $s = 0
    foreach ($p in $paths) {
        if (Test-Path $p) {
            $s += (Get-ChildItem $p -Recurse -File -Force -ErrorAction SilentlyContinue |
                   Measure-Object Length -Sum).Sum
        }
    }
    return [math]::Round($s / 1GB, 2)
}
function Bar($label, $have, $want) {
    $pct = [math]::Min(100, [math]::Round(100 * $have / $want))
    $n = [int]($pct / 2.5)
    "{0,-14} [{1}{2}] {3,3}%  {4:N2} / ~{5} GB" -f $label, ('#' * $n), ('.' * (40 - $n)), $pct, $have, $want
}
function LastLog($pattern) {
    Get-ChildItem (Join-Path $repo "logs\$pattern") -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
}
function Has($log, $marker) { $log -and (Select-String -Path $log.FullName -Pattern $marker -Quiet) }
function Tail1($log) {
    if (-not $log) { return "-" }
    $l = Get-Content $log.FullName -Tail 1 -ErrorAction SilentlyContinue
    if ($l.Length -gt 90) { $l = $l.Substring(0, 90) + "..." }
    return $l
}

function NetBytes { (Get-NetAdapterStatistics -ErrorAction SilentlyContinue | Measure-Object ReceivedBytes -Sum).Sum }

$weights = @((Join-Path $repo ".cache"), (Join-Path $repo "checkpoints"))
$prevNet = NetBytes; $prevT = Get-Date; $t0 = Get-Date
while ($true) {
    $w = SizeGB $weights
    # Rate from the network adapter, not from folder sizes: uv downloads torch into its own
    # global cache, so the folders do not grow while the biggest download is in flight.
    $net = NetBytes; $now = Get-Date
    $rate = [math]::Round(($net - $prevNet) / 1MB / [math]::Max(0.1, ($now - $prevT).TotalSeconds), 1)
    $prevNet = $net; $prevT = $now
    $mainLog = LastLog "prefetch_main_*.log"
    $zoeLog = LastLog "install_zoe_*.log"
    $zoeVenv = SizeGB @((Join-Path $repo ".venv-zoe"))

    if (Has $mainLog "PREFETCH_MAIN_DONE") { $mainState = "done" } else { $mainState = "downloading" }
    if (Has $zoeLog "PREFETCH_ZOE_DONE") { $zoeState = "done (venv + weights)" }
    elseif (Has $zoeLog "ZOE_OK") { $zoeState = "venv OK, downloading ZoeDepth weights" }
    elseif ($zoeVenv -gt 0.2) { $zoeState = "installing packages into .venv-zoe ($zoeVenv GB)" }
    else { $zoeState = "downloading packages (torch cu128 ~3 GB)" }

    Clear-Host
    "SyMTRS pre-downloads   {0:HH:mm:ss}   (elapsed {1:hh\:mm\:ss})" -f (Get-Date), ((Get-Date) - $t0)
    ""
    Bar "Weights" $w $WeightsGB
    "                 network: $rate MB/s (whole machine)"
    ""
    "Main weights   : $mainState"
    "  log> " + (Tail1 $mainLog)
    "ZoeDepth       : $zoeState"
    "  log> " + (Tail1 $zoeLog)
    if ((Has $mainLog "PREFETCH_MAIN_DONE") -and (Has $zoeLog "PREFETCH_ZOE_DONE")) {
        ""; "ALL DOWNLOADS FINISHED - you can close this window."
    }
    Start-Sleep -Seconds $Every
}
