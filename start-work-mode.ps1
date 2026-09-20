# Start the chatgpt-use WORK mode stack (MCP server + optional tunnel).
#
#   powershell -ExecutionPolicy Bypass -File .\start-work-mode.ps1 -Project "C:\path\to\your\app"
#
# The server runs as a tracked child process, not a PowerShell job, so it is
# always stopped on the way out - a job left behind keeps the port bound and the
# next run fails with "only one usage of each socket address".

[CmdletBinding()]
param(
    [string]$Project = (Get-Location).Path,
    [int]$Port = 8788,
    # `mcp` defaults to the read-only profile, and this script keeps that default.
    # write/bash over a public tunnel is a real remote shell on this machine, so
    # it takes an explicit flag rather than being the out-of-the-box behaviour.
    [switch]$FullProfile,
    [switch]$ShowToken,
    [string]$LogPath = (Join-Path $env:TEMP 'chatgpt-use-mcp.log'),
    # Overridable so the guards can be exercised without a full install, and so a
    # non-default -BinDir from setup-windows.ps1 keeps working here.
    [string]$Bin = (Join-Path $env:USERPROFILE '.local\bin\chatgpt-use.exe'),
    [string]$AuthFile = (Join-Path $env:USERPROFILE '.chatgpt-use\auth.json')
)

$ErrorActionPreference = 'Stop'

function Die([string]$m, [string]$hint) {
    Write-Host "error: $m" -ForegroundColor Red
    if ($hint) { Write-Host "       $hint" -ForegroundColor Yellow }
    exit 1
}

if (-not (Test-Path $Bin)) {
    Die "chatgpt-use not found at $Bin" 'run .\setup-windows.ps1 first'
}
# Existence check before reading: on a machine that has never run `init`, the old
# script died inside Get-Content with a .NET path error instead of saying
# "run chatgpt-use init".
if (-not (Test-Path $AuthFile)) {
    Die "no auth token at $AuthFile" 'run: chatgpt-use init'
}
if (-not (Test-Path $Project)) { Die "project path does not exist: $Project" }

$token = $null
try { $token = (Get-Content -Raw $AuthFile | ConvertFrom-Json).token } catch { }
if (-not $token) { Die "auth.json has no token field" 're-run: chatgpt-use init --force' }

# Refusing to start on a busy port is better than starting a server that is
# unreachable and blaming the tunnel.
try {
    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        $owner = ($listener | Select-Object -First 1).OwningProcess
        Die "port $Port is already in use by pid $owner" `
            'stop that server, or pass another -Port'
    }
} catch {
    # Get-NetTCPConnection is absent on some SKUs; the bind attempt below still
    # catches the real collision, so this is only a lost early warning.
}

$args = @('mcp', '--port', "$Port", '--token', $token, '--cwd', $Project)
if ($FullProfile) {
    Write-Host "`nFULL PROFILE: ChatGPT may write files and run commands in $Project." -ForegroundColor Red
    Write-Host "Anyone holding the tunnel URL and this token gets that too."
    $args += @('--profile', 'full', '--permission-mode', 'trusted')
} else {
    $args += @('--profile', 'read-only')
}

Write-Host "=== chatgpt-use MCP server ===" -ForegroundColor Cyan
Write-Host "  project : $Project"
Write-Host "  port    : $Port"
Write-Host "  profile : $(if ($FullProfile) { 'full (write + bash)' } else { 'read-only' })"
Write-Host "  log     : $LogPath"
if ($ShowToken) { Write-Host "  token   : $token" -ForegroundColor Yellow }

Start-Process -FilePath $Bin -ArgumentList $args -WindowStyle Hidden `
    -RedirectStandardOutput $LogPath -RedirectStandardError "$LogPath.err"
Start-Sleep -Seconds 2

# The child may have died immediately (bad port, bad token); say so here rather
# than sending the user to configure a connector against nothing.
$server = Get-Process -Name chatgpt-use -ErrorAction SilentlyContinue |
    Where-Object { $_.StartTime -gt (Get-Date).AddSeconds(-20) } |
    Select-Object -First 1
if (-not $server) {
    Write-Host "`nthe server exited within 2s. Last lines of $LogPath.err:" -ForegroundColor Red
    Get-Content "$LogPath.err" -Tail 5 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" }
    Die 'MCP server did not stay up' "see $LogPath and $LogPath.err"
}
$serverPid = $server.Id

try {
    $cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
    if (-not $cloudflared) {
        Write-Host "`ncloudflared not found - the server is local-only at http://127.0.0.1:$Port" -ForegroundColor Yellow
        Write-Host 'ChatGPT cannot reach it until you tunnel it. Install with:'
        Write-Host '  winget install Cloudflare.cloudflared'
        Write-Host "`nPress Ctrl+C to stop the server."
        Wait-Process -Id $serverPid
        return
    }

    Write-Host "`n=== starting Cloudflare tunnel ===" -ForegroundColor Yellow
    Write-Host 'Register this URL in ChatGPT -> Settings -> Apps -> Add custom connector:'
    Write-Host "  URL    https://<tunnel-url>/"
    Write-Host "  Auth   Authorization: Bearer $(if ($ShowToken) { $token } else { '<run with -ShowToken to display>' })"
    Write-Host 'In the connector settings set permissions to "Always allow (without confirmation)".'
    Write-Host "`nThen run work commands in a NEW terminal:" -ForegroundColor Green
    Write-Host '  chatgpt-use work "Build a todo app with HTML/CSS/JS" --loop'
    Write-Host "`nCtrl+C stops the tunnel; the server is stopped with it.`n"
    & cloudflared tunnel --url "http://127.0.0.1:$Port"
} finally {
    # Always, including on Ctrl+C and on a throw - this is the leak the job
    # version had.
    Stop-Process -Id $serverPid -Force -ErrorAction SilentlyContinue
    Write-Host "`nstopped MCP server (pid $serverPid)" -ForegroundColor Cyan
}
