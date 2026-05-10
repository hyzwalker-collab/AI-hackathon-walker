$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements.txt"
$Proxy = "http://127.0.0.1:7897"

if (!(Test-Path $Python)) {
    throw "Python not found: $Python"
}

if (!(Test-Path $Requirements)) {
    throw "requirements.txt not found: $Requirements"
}

Set-Location $ProjectRoot

$items = Get-Content $Requirements |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -and !($_.StartsWith("#")) }

Write-Host "Project: $ProjectRoot"
Write-Host "Python:  $Python"
Write-Host "Proxy:   $Proxy"
Write-Host "Packages: $($items.Count)"
Write-Host ""

foreach ($item in $items) {
    $start = Get-Date -Format "HH:mm:ss"
    Write-Host "[$start] Installing $item" -ForegroundColor Cyan

    & $Python -m pip install $item `
        --proxy $Proxy `
        --progress-bar on `
        --timeout 60 `
        --retries 2 `
        --disable-pip-version-check

    if ($LASTEXITCODE -ne 0) {
        throw "pip failed while installing: $item"
    }

    $end = Get-Date -Format "HH:mm:ss"
    Write-Host "[$end] Done $item" -ForegroundColor Green
    Write-Host ""
}

Write-Host "All requirements installed." -ForegroundColor Green
