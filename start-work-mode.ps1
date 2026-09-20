# Start chatgpt-use WORK mode stack (MCP server + tunnel)
# Usage: powershell -ExecutionPolicy Bypass -File .\start-work-mode.ps1 -Project "C:\path\to\your\app"

param(
    [string]$Project = (Get-Location).Path,
    [int]$Port = 8788
)

$ErrorActionPreference = "Stop"
$Bin = Join-Path $env:USERPROFILE ".local\bin\chatgpt-use.exe"
$AuthFile = Join-Path $env:USERPROFILE ".chatgpt-use\auth.json"

if (-not (Test-Path $Bin)) {
    Write-Error "chatgpt-use not found. Run setup-windows.ps1 first."
}

$token = (Get-Content $AuthFile | ConvertFrom-Json).token
if (-not $token) {
    Write-Error "No auth token. Run: chatgpt-use init"
}

Write-Host "=== chatgpt-use WORK mode ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project folder: $Project"
Write-Host "MCP port:       $Port"
Write-Host "Auth token:     $token"
Write-Host ""
Write-Host "Starting MCP server (profile=full, allows write/bash)..." -ForegroundColor Yellow
Write-Host "Keep this window OPEN while using work mode."
Write-Host ""

# Start MCP server in background job
$mcpJob = Start-Job -ScriptBlock {
    param($bin, $port, $project, $token)
    & $bin mcp --port $port --token $token --cwd $project --profile full --permission-mode trusted
} -ArgumentList $Bin, $Port, $Project, $token

Start-Sleep -Seconds 2

# Check cloudflared
$cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
if (-not $cloudflared) {
    Write-Host "cloudflared not found. Install it:" -ForegroundColor Red
    Write-Host "  winget install Cloudflare.cloudflared"
    Write-Host ""
    Write-Host "MCP server is running locally at http://127.0.0.1:$Port"
    Write-Host "You still need a tunnel for ChatGPT to reach it."
    Write-Host ""
    Write-Host "Press Ctrl+C to stop."
    Wait-Job $mcpJob
    exit 0
}

Write-Host "Starting Cloudflare tunnel..." -ForegroundColor Yellow
Write-Host ""
Write-Host "=== COPY THE HTTPS URL BELOW ===" -ForegroundColor Green
Write-Host "Register it in ChatGPT -> Settings -> Apps -> Add custom connector"
Write-Host "  URL:   https://<tunnel-url>/"
Write-Host "  Auth:  Authorization: Bearer $token"
Write-Host "         (or append ?token=$token to the URL)"
Write-Host ""
Write-Host "IMPORTANT: In connector settings, set permissions to"
Write-Host "  'Always allow (without confirmation)'"
Write-Host ""
Write-Host "Then run work commands in a NEW terminal:"
Write-Host '  chatgpt-use work "Build a todo app with HTML/CSS/JS" --loop'
Write-Host ""

# Run tunnel in foreground so user sees the URL
cloudflared tunnel --url "http://127.0.0.1:$Port"

Stop-Job $mcpJob -ErrorAction SilentlyContinue
Remove-Job $mcpJob -ErrorAction SilentlyContinue
