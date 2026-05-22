param(
    [string]$ProjectDir = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

Set-Location -Path $ProjectDir

try {
    & docker info *> $null
} catch {
    Write-Host "Docker não está disponível (daemon não está rodando ou sem permissão)." -ForegroundColor Red
    throw
}

$running = $false
try {
    $ids = & docker compose ps -q
    if ($ids -and ($ids | Out-String).Trim().Length -gt 0) {
        $running = $true
    }
} catch {
    $running = $false
}

if ($running) {
    Write-Host "Compose está UP. Executando down..." -ForegroundColor Yellow
} else {
    Write-Host "Compose não parece estar UP. Executando down mesmo assim..." -ForegroundColor Yellow
}

& docker compose down

Write-Host "Subindo novamente..." -ForegroundColor Yellow
if ($NoBuild) {
    & docker compose up -d
} else {
    & docker compose up -d --build
}

Write-Host "OK. Acesse http://localhost:8000/" -ForegroundColor Green
