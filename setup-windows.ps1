# chatgpt-use Windows setup helper
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File .\setup-windows.ps1

$ErrorActionPreference = "Stop"
$BinDir = Join-Path $env:USERPROFILE ".local\bin"
$ChromeUseVersion = "v1.5.123"

function Ensure-BinDir {
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    $pathParts = $env:Path -split ';'
    if ($pathParts -notcontains $BinDir) {
        Write-Host "Adding $BinDir to your user PATH..."
        [Environment]::SetEnvironmentVariable("Path", "$BinDir;" + $env:Path, "User")
        $env:Path = "$BinDir;" + $env:Path
    }
}

function Install-ChromeUse {
    $asset = "chrome-use-win32-x64.tar.gz"
    $url = "https://github.com/leeguooooo/chrome-use/releases/download/$ChromeUseVersion/$asset"
    $tmp = Join-Path $env:TEMP "chrome-use-install"
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    Write-Host "Downloading chrome-use $ChromeUseVersion..."
    Invoke-WebRequest -Uri $url -OutFile (Join-Path $tmp $asset)
    tar -xzf (Join-Path $tmp $asset) -C $tmp
    Copy-Item -Force (Join-Path $tmp "chrome-use.exe") (Join-Path $BinDir "chrome-use.exe")
}

function Build-ChatGptUse {
    Write-Host "Building chatgpt-use (requires Rust 1.98+)..."
    cargo build --release
    Copy-Item -Force ".\target\release\chatgpt-use.exe" (Join-Path $BinDir "chatgpt-use.exe")
}

Ensure-BinDir

if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) {
    Write-Error "Rust is required. Install from https://rustup.rs then re-run this script."
}

if (-not (Test-Path (Join-Path $BinDir "chrome-use.exe"))) {
    Install-ChromeUse
} else {
    Write-Host "chrome-use already installed at $BinDir\chrome-use.exe"
}

Build-ChatGptUse

Write-Host ""
Write-Host "Registering chrome-use native messaging host..."
& (Join-Path $BinDir "chrome-use.exe") extension install

Write-Host ""
Write-Host "Generating MCP auth token..."
& (Join-Path $BinDir "chatgpt-use.exe") init

Write-Host ""
Write-Host "=== Setup complete ==="
Write-Host ""
Write-Host "1. Install the chrome-use extension in Chrome:"
Write-Host "   https://chromewebstore.google.com/detail/knfcmbamhjmaonkfnjhldjedeobeafmk"
Write-Host ""
Write-Host "2. Log into https://chatgpt.com in that Chrome profile."
Write-Host ""
Write-Host "3. Quick test (sidekick mode):"
Write-Host '   chatgpt-use ask "Say hello in one sentence"'
Write-Host ""
Write-Host "4. For MCP / closed-loop mode, start the server:"
Write-Host "   chatgpt-use mcp --port 8788 --cwd ."
Write-Host "   Then expose it with cloudflared and add it as a ChatGPT connector."
Write-Host ""
Write-Host "Binaries: $BinDir"
