<#
.SYNOPSIS
    Production Launcher for Graph RAG Benchmark (Local Windows Profile A).
.DESCRIPTION
    Validates Python environment, configuration, directory permissions, and Ollama connectivity
    before starting the Streamlit operational web dashboard.
#>

[CmdletBinding()]
param (
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8501,
    [switch]$NoBrowser = $false
)

$ErrorActionPreference = "Stop"
$WorkspaceRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location -Path $WorkspaceRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  GRAPH RAG BENCHMARK - PRODUCTION LAUNCHER (PROFILE A)     " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Workspace Root: $WorkspaceRoot"

# 1. Verify Virtual Environment
$VenvPython = Join-Path $WorkspaceRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment python binary not found at: $VenvPython`nPlease initialize .venv before launching."
    exit 1
}

# 2. Check Python Runtime
$PyVersion = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
Write-Host "[OK] Python Runtime: $PyVersion" -ForegroundColor Green

# 3. Verify Required Directories
$RequiredDirs = @("data", "data\processed", "data\chroma", "data\knowledge_graph", "data\backups", "logs")
foreach ($dir in $RequiredDirs) {
    $fullDir = Join-Path $WorkspaceRoot $dir
    if (-not (Test-Path $fullDir)) {
        New-Item -ItemType Directory -Path $fullDir -Force | Out-Null
        Write-Host "[CREATED] Directory: $dir" -ForegroundColor Yellow
    }
}
Write-Host "[OK] Storage directories verified." -ForegroundColor Green

# 4. Check Ollama Connectivity
Write-Host "Checking Ollama connectivity..."
try {
    $OllamaCheck = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction SilentlyContinue
    if ($OllamaCheck -and $OllamaCheck.models) {
        $ModelCount = $OllamaCheck.models.Count
        Write-Host "[OK] Ollama server reachable ($ModelCount models installed)." -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Ollama server reached but no models found." -ForegroundColor Yellow
    }
} catch {
    Write-Host "[WARNING] Ollama server unreachable at http://localhost:11434. Generation features will be degraded." -ForegroundColor Yellow
}

# 5. Execute Pre-Flight Health Check
Write-Host "Executing pre-flight health evaluation..."
& $VenvPython scripts/health_check.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pre-flight health check returned failure exit code $LASTEXITCODE. Aborting startup."
    exit 1
}

# 6. Launch Streamlit Application
Write-Host "`nStarting Streamlit Application on http://${HostAddress}:${Port}..." -ForegroundColor Cyan
$StreamlitArgs = @(
    "-m", "streamlit", "run", "app.py",
    "--server.address=$HostAddress",
    "--server.port=$Port",
    "--server.headless=true"
)
if ($NoBrowser) {
    $StreamlitArgs += "--server.headless=true"
}

& $VenvPython @StreamlitArgs
