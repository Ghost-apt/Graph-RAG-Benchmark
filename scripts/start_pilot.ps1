<#
.SYNOPSIS
    Real-User Pilot Launcher for Graph RAG Benchmark (Phase 10).
.DESCRIPTION
    Validates Python runtime, auth configuration, user registry, and Ollama engine
    before launching the Streamlit pilot application.
#>

[CmdletBinding()]
param (
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8501,
    [ValidateSet("dev", "oidc", "none")]
    [string]$AuthMode = "dev",
    [switch]$NoBrowser = $false
)

$ErrorActionPreference = "Stop"
$WorkspaceRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location -Path $WorkspaceRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  GRAPH RAG BENCHMARK - PILOT LAUNCHER (v0.10.0)            " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Workspace Root: $WorkspaceRoot"
Write-Host "Configured Auth Mode: $AuthMode"

# 1. Environment Variable Setup
$env:PYTHONPATH = "src"
$env:GRAPHRAG_AUTH_MODE = $AuthMode

# 2. Verify Virtual Environment
$VenvPython = Join-Path $WorkspaceRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment python not found at: $VenvPython"
    exit 1
}

# 3. Verify Required Directories
$RequiredDirs = @(
    "data",
    "data\processed",
    "data\chroma",
    "data\knowledge_graph",
    "data\backups",
    "data\users",
    "logs"
)
foreach ($dir in $RequiredDirs) {
    $fullDir = Join-Path $WorkspaceRoot $dir
    if (-not (Test-Path $fullDir)) {
        New-Item -ItemType Directory -Path $fullDir -Force | Out-Null
        Write-Host "[CREATED] Directory: $dir" -ForegroundColor Yellow
    }
}
Write-Host "[OK] Pilot storage directories verified." -ForegroundColor Green

# 4. Check Auth Configuration
Write-Host "Verifying authentication configuration..."
& $VenvPython scripts/check_auth_config.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Authentication configuration verification failed. Aborting pilot startup."
    exit 1
}

# 5. Check System Health
Write-Host "Executing pre-flight health probe..."
& $VenvPython scripts/pilot_health_check.py
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Health check reported warnings/issues. Review status before proceeding."
}

# 6. Launch Streamlit Application
Write-Host "`nStarting Streamlit Pilot on http://${HostAddress}:${Port}..." -ForegroundColor Cyan
$StreamlitArgs = @(
    "-m", "streamlit", "run", "app.py",
    "--server.address=$HostAddress",
    "--server.port=$Port",
    "--browser.gatherUsageStats=false"
)
if ($NoBrowser) {
    $StreamlitArgs += "--server.headless=true"
}

# Record PID for stop script
$Proc = Start-Process -FilePath $VenvPython -ArgumentList $StreamlitArgs -PassThru
$Proc.Id | Out-File -FilePath (Join-Path $WorkspaceRoot "logs\pilot.pid") -Encoding utf8
Write-Host "[RUNNING] Pilot server started with Process ID $($Proc.Id)" -ForegroundColor Green
Write-Host "To stop the pilot server, run: .\scripts\stop_pilot.ps1" -ForegroundColor Gray
