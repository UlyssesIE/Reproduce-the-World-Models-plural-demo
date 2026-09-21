param(
    [string]$Repo  = "D:\Code demos\World Models reproduce\Reproduce-the-World-Models-plural-demo",
    [string]$Run   = "controller_867",
    [int]   $Total = 100,
    [int]   $Every = 30
)
$log = Join-Path $Repo "runs\$Run\log.jsonl"
Write-Host "watching: $log" -ForegroundColor Cyan
Write-Host "Ctrl+C to stop watching (this does NOT stop training).`n" -ForegroundColor DarkGray

while ($true) {
    $stamp = Get-Date -Format "HH:mm:ss"
    $n = (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
          Where-Object { $_.CommandLine -match 'train_controller' } | Measure-Object).Count
    Write-Host "[$stamp] training processes: $n" -ForegroundColor Yellow

    if (Test-Path $log) {
        $lines = @(Get-Content $log)
        Write-Host "  log lines: $($lines.Count)"
        $lines | Select-Object -Last 5 | ForEach-Object { Write-Host "   $_" }
        if ($lines.Count -gt 0) {
            $l = $lines[-1] | ConvertFrom-Json
            $perGen = $l.secs / ($l.gen + 1)
            $eta    = $perGen * ($Total - $l.gen - 1) / 3600
            Write-Host ("  -> gen {0}/{1} | {2:N0} s/gen | elapsed {3:N1} h | ETA {4:N1} h" `
                -f $l.gen, $Total, $perGen, ($l.secs/3600), $eta) -ForegroundColor Green
        }
    } else {
        Write-Host "  LOG NOT FOUND: $log" -ForegroundColor Red
        Write-Host "  runs\ contains:" -ForegroundColor DarkGray
        Get-ChildItem (Join-Path $Repo "runs") -Directory |
            Select-Object -ExpandProperty Name | ForEach-Object { Write-Host "    $_" }
    }
    Start-Sleep $Every
}
