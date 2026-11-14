<#
.SYNOPSIS
    Stops the Graph RAG Benchmark Pilot Server.
.DESCRIPTION
    Reads logs/pilot.pid or terminates processes bound to port 8501.
#>

[CmdletBinding()]
param (
    [int]$Port = 8501
)

$ErrorActionPreference = "SilentlyContinue"
$WorkspaceRoot = (Resolve-Path "$PSScriptRoot\..").Path
$PidFile = Join-Path $WorkspaceRoot "logs\pilot.pid"

Write-Host "Stopping Graph RAG Benchmark Pilot Server..." -ForegroundColor Cyan

$Stopped = $false

# 1. Stop by PID file if exists
if (Test-Path $PidFile) {
    $TargetPid = Get-Content -Path $PidFile -ErrorAction SilentlyContinue | Out-String
    $TargetPid = $TargetPid.Trim()
    if ($TargetPid -match '^\d+$') {
        $Proc = Get-Process -Id [int]$TargetPid -ErrorAction SilentlyContinue
        if ($Proc) {
            Stop-Process -Id [int]$TargetPid -Force -ErrorAction SilentlyContinue
            Write-Host "[STOPPED] Terminated pilot process ID $TargetPid." -ForegroundColor Green
            $Stopped = $true
        }
    }
    Remove-Item -Path $PidFile -Force -ErrorAction SilentlyContinue
}

# 2. Check for lingering processes listening on target port
try {
    $Netstat = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if ($Netstat) {
        foreach ($conn in $Netstat) {
            $OwningPid = $conn.OwningProcess
            if ($OwningPid -and $OwningPid -ne 0) {
                Stop-Process -Id $OwningPid -Force -ErrorAction SilentlyContinue
                Write-Host "[STOPPED] Terminated process ID $OwningPid listening on port $Port." -ForegroundColor Green
                $Stopped = $true
            }
        }
    }
} catch {}

if (-not $Stopped) {
    Write-Host "No active pilot server found running on port $Port." -ForegroundColor Yellow
} else {
    Write-Host "Pilot server stopped cleanly." -ForegroundColor Green
}
