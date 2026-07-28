param(
    [int]$Port = 0
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$envPath = Join-Path $projectRoot ".env"
$runDir = Join-Path $projectRoot ".run"
$pidFile = Join-Path $runDir "tc4-api.pid"

function Get-EnvValue {
    param(
        [string]$Path,
        [string]$Name,
        [string]$DefaultValue
    )

    if (-not (Test-Path $Path)) {
        return $DefaultValue
    }

    foreach ($line in Get-Content $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }

        if ($trimmed.StartsWith("$Name=")) {
            return $trimmed.Substring($Name.Length + 1).Trim()
        }
    }

    return $DefaultValue
}

function Stop-ExistingProcess {
    param(
        [int]$ProcessId,
        [string]$Reason
    )

    if ($ProcessId -eq $PID) {
        return $false
    }

    try {
        $process = Get-Process -Id $ProcessId -ErrorAction Stop
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
        Write-Host "Encerrado PID $($process.Id) ($($process.ProcessName)) por $Reason."
        return $true
    }
    catch {
        Write-Host "PID $ProcessId ja nao estava em execucao."
        return $false
    }
}

$stoppedSomething = $false

if (Test-Path $pidFile) {
    $rawPid = (Get-Content $pidFile -Raw).Trim()
    $trackedPid = 0
    if ([int]::TryParse($rawPid, [ref]$trackedPid)) {
        if (Stop-ExistingProcess -ProcessId $trackedPid -Reason "pid salvo em $pidFile") {
            $stoppedSomething = $true
        }
    }

    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

if ($Port -le 0) {
    $Port = [int](Get-EnvValue -Path $envPath -Name "TC4_API_PORT" -DefaultValue "8010")
}

if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique

    foreach ($listenerPid in $listeners) {
        if (Stop-ExistingProcess -ProcessId $listenerPid -Reason "uso da porta $Port") {
            $stoppedSomething = $true
        }
    }
}

if ($stoppedSomething) {
    Write-Host "TC4 finalizado."
}
else {
    Write-Host "Nenhum processo do TC4 foi encontrado em execucao."
}
