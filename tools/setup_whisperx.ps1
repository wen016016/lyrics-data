$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot '.venv-whisperx'

function Ensure-WingetPackage($Id, $CommandName) {
    if (Get-Command $CommandName -ErrorAction SilentlyContinue) {
        Write-Host "[OK] $CommandName is available."
        return
    }

    Write-Host "[Install] $Id"
    winget install --id $Id -e --accept-source-agreements --accept-package-agreements
}

try {
    & py -3.12 --version | Out-Null
}
catch {
    Write-Host '[Install] Python 3.12'
    winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
}

Ensure-WingetPackage 'Gyan.FFmpeg.Shared' 'ffmpeg'
Ensure-WingetPackage 'OpenJS.NodeJS.LTS' 'node'

if (-not (Test-Path $VenvPath)) {
    Write-Host '[Create] Python 3.12 virtual environment'
    & py -3.12 -m venv $VenvPath
}

$Python = Join-Path $VenvPath 'Scripts\python.exe'
$Requirements = Join-Path $PSScriptRoot 'requirements.txt'
& $Python -m pip install --upgrade pip
# 套件清單集中在 requirements.txt，避免這裡和那邊各寫一份、久了就對不上
& $Python -m pip install -r $Requirements

Write-Host ''
Write-Host '[Done] Environment is ready.'
Write-Host 'Optional: set ANTHROPIC_API_KEY (or OPENAI_API_KEY) to enable automatic Chinese translation.'
Write-Host 'If FFmpeg or Node.js was installed just now, close and reopen PowerShell once before running alignment.'
