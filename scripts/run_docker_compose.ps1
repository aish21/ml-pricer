<#
Helper PowerShell script to build and run the docker compose setup and tail logs.

Usage (PowerShell, from repo root):
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\scripts\run_docker_compose.ps1

This script only runs Docker commands; it does not install Docker.
#>

function Get-CommandExists {
    param([string]$cmd)
    $null -ne (Get-Command $cmd -ErrorAction SilentlyContinue)
}

if (-not (Get-CommandExists -cmd "docker")) {
    Write-Host "Docker CLI not found in PATH. Please install Docker Desktop and ensure 'docker' is available." -ForegroundColor Yellow
    exit 2
}

Write-Host "Building and starting containers (detached)..." -ForegroundColor Cyan
try {
    & docker compose up --build -d
} catch {
    # try legacy binary
    & docker-compose up --build -d
}

Write-Host "Listing containers..." -ForegroundColor Cyan
docker compose ps

Write-Host "Tailing backend logs (last 200 lines):" -ForegroundColor Cyan
docker compose logs --tail 200 backend

Write-Host "Tailing frontend logs (last 200 lines):" -ForegroundColor Cyan
docker compose logs --tail 200 frontend

Write-Host "If containers failed to start, examine the logs above and ensure requirements.txt includes FastAPI, uvicorn and streamlit and that final/results contains model artifacts." -ForegroundColor Green
