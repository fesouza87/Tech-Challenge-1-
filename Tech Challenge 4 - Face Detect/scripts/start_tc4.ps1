param(
    [switch]$NoBrowser,
    [switch]$Foreground,
    [int]$StartupTimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$envPath = Join-Path $projectRoot ".env"
$pythonExe = Join-Path $projectRoot ".venv_tc4\Scripts\python.exe"
$runDir = Join-Path $projectRoot ".run"
$pidFile = Join-Path $runDir "tc4-api.pid"
$stdoutLog = Join-Path $runDir "tc4-api.stdout.log"
$stderrLog = Join-Path $runDir "tc4-api.stderr.log"

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
        return
    }

    try {
        $process = Get-Process -Id $ProcessId -ErrorAction Stop
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
        Write-Host "Encerrado PID $($process.Id) ($($process.ProcessName)) por $Reason."
    }
    catch {
        Write-Host "PID $ProcessId ja nao estava em execucao."
    }
}

function Stop-TrackedProcess {
    param([string]$PidFilePath)

    if (-not (Test-Path $PidFilePath)) {
        return
    }

    $rawPid = (Get-Content $PidFilePath -Raw).Trim()
    if (-not $rawPid) {
        Remove-Item $PidFilePath -Force -ErrorAction SilentlyContinue
        return
    }

    $trackedPid = 0
    if ([int]::TryParse($rawPid, [ref]$trackedPid)) {
        Stop-ExistingProcess -ProcessId $trackedPid -Reason "pid salvo em $PidFilePath"
    }

    Remove-Item $PidFilePath -Force -ErrorAction SilentlyContinue
}

function Stop-PortListeners {
    param([int]$Port)

    $listeners = @()
    if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
        $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    }

    foreach ($listenerPid in $listeners) {
        Stop-ExistingProcess -ProcessId $listenerPid -Reason "uso da porta $Port"
    }
}

function Stop-ProjectUvicorn {
    param(
        [string]$ProjectRoot,
        [string]$PythonPath
    )

    $uvicornProcesses = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and
            $_.ProcessId -ne $PID -and
            $_.CommandLine -like "*$PythonPath*" -and
            $_.CommandLine -like "*uvicorn*" -and
            $_.CommandLine -like "*src.main:app*"
        }

    foreach ($proc in $uvicornProcesses) {
        Stop-ExistingProcess -ProcessId $proc.ProcessId -Reason "uvicorn do TC4 ja em execucao"
    }
}

function Wait-ForHealth {
    param(
        [string]$HealthUrl,
        [int]$TimeoutSeconds
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        Start-Sleep -Milliseconds 750
        try {
            $response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -eq 200) {
                return $true
            }
        }
        catch {
        }
    } while ((Get-Date) -lt $deadline)

    return $false
}

if (-not (Test-Path $pythonExe)) {
    throw "Nao encontrei a venv esperada em $pythonExe"
}

New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$hostValue = Get-EnvValue -Path $envPath -Name "TC4_API_HOST" -DefaultValue "127.0.0.1"
$portValue = Get-EnvValue -Path $envPath -Name "TC4_API_PORT" -DefaultValue "8010"
$baseUrl = "http://{0}:{1}" -f $hostValue, $portValue
$healthUrl = "$baseUrl/health"

Write-Host "Reiniciando TC4 em $baseUrl ..."
Stop-TrackedProcess -PidFilePath $pidFile
Stop-PortListeners -Port ([int]$portValue)
Stop-ProjectUvicorn -ProjectRoot $projectRoot -PythonPath $pythonExe

if ($Foreground) {
    Write-Host "Subindo API em primeiro plano."
    & $pythonExe -m uvicorn main:app --app-dir src --host $hostValue --port $portValue
    exit $LASTEXITCODE
}

if (Test-Path $stdoutLog) {
    Remove-Item $stdoutLog -Force -ErrorAction SilentlyContinue
}
if (Test-Path $stderrLog) {
    Remove-Item $stderrLog -Force -ErrorAction SilentlyContinue
}

$process = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList @("-m", "uvicorn", "main:app", "--app-dir", "src", "--host", $hostValue, "--port", $portValue) `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Set-Content -Path $pidFile -Value $process.Id

if (-not (Wait-ForHealth -HealthUrl $healthUrl -TimeoutSeconds $StartupTimeoutSeconds)) {
    $tail = ""
    if (Test-Path $stderrLog) {
        $tail = (Get-Content $stderrLog -Tail 20) -join [Environment]::NewLine
    }

    Stop-ExistingProcess -ProcessId $process.Id -Reason "timeout de inicializacao"
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue

    throw "A API nao respondeu em $HealthUrl dentro de $StartupTimeoutSeconds s.`n$tail"
}

Write-Host "TC4 no ar em $baseUrl"
Write-Host "Swagger: $baseUrl/docs"
Write-Host "Health:  $healthUrl"
Write-Host "Logs:    $stdoutLog"

if (-not $NoBrowser) {
    Start-Process $baseUrl | Out-Null
}
