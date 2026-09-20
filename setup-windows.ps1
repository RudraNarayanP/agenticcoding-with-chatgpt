# chatgpt-use Windows setup helper.
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File .\setup-windows.ps1
#
# Installs chrome-use, builds chatgpt-use, registers the browser extension host,
# and creates the MCP auth token. Re-runnable: every step checks for an existing
# install before touching anything.

[CmdletBinding()]
param(
    # Empty means "whatever is the latest chrome-use release", resolved from the
    # GitHub API. Pin only to reproduce a problem: -ChromeUseVersion v1.5.125
    [string]$ChromeUseVersion = '',
    [string]$BinDir = (Join-Path $env:USERPROFILE '.local\bin'),
    # Skip the cargo build (chrome-use + extension host only).
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 still defaults to TLS 1.0/1.2 negotiation that GitHub's
# API sometimes refuses; ask for a modern protocol before the first request.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

function Write-Step([string]$m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Write-Ok([string]$m)   { Write-Host "    $m" -ForegroundColor DarkGray }

function Add-BinDirToPath {
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    $parts = ($env:Path -split ';')
    if ($parts -notcontains $BinDir) {
        Write-Step "adding $BinDir to your user PATH"
        # Written to the registry as well as this process, so a new terminal
        # finds it; this process needs it too because we invoke by name below.
        [Environment]::SetEnvironmentVariable('Path', "$BinDir;$env:Path", 'User')
        $env:Path = "$BinDir;$env:Path"
    } else {
        Write-Ok "$BinDir is already on PATH"
    }
}

# A toolchain counts as present only if it actually runs. This machine proved the
# difference: `Get-Command rustc` succeeded on a half-installed rustup whose
# `rustc --version` errored, so the old check reported "Rust is required" while
# Rust was half there, or the reverse, depending on the failure.
function Test-Rust {
    $probe = & rustc --version 2>&1
    if ($LASTEXITCODE -ne 0 -or -not $probe) { return $null }
    # "cargo 1.98.1" - cargo can be missing while rustc works.
    $cargoProbe = & cargo --version 2>&1
    if ($LASTEXITCODE -ne 0 -or -not $cargoProbe) { return $null }
    return "$probe / $cargoProbe"
}

function Get-LatestChromeUseTag {
    if ($ChromeUseVersion) { return $ChromeUseVersion }
    try {
        $json = (Invoke-WebRequest -UseBasicParsing `
            -Uri 'https://api.github.com/repos/leeguooooo/chrome-use/releases/latest').Content
        if ($json -match '"tag_name":\s*"([^"]+)"') { return $Matches[1] }
    } catch {
        Write-Host "    could not reach the GitHub API ($($_.Exception.Message)); using the pinned version" -ForegroundColor Yellow
    }
    return 'v1.5.125'
}

function Install-ChromeUse([string]$Tag) {
    $asset = 'chrome-use-win32-x64.tar.gz'
    $base  = "https://github.com/leeguooooo/chrome-use/releases/download/$Tag"
    $tmp   = Join-Path $env:TEMP "chrome-use-install-$Tag"
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null

    Write-Step "downloading chrome-use $Tag"
    Invoke-WebRequest -UseBasicParsing -Uri "$base/$asset" -OutFile (Join-Path $tmp $asset)

    # The sidecar is `sha256sum` output: "<64 hex><spaces><filename>". Verifying
    # matters more here than it looks: the artifact is an executable that gets a
    # native-messaging host registered under it.
    Write-Step 'verifying checksum'
    Invoke-WebRequest -UseBasicParsing -Uri "$base/$asset.sha256" -OutFile (Join-Path $tmp "$asset.sha256")
    # `-Raw` + split, not `(Get-Content f)[0]`: a one-line file makes Get-Content
    # return a scalar string, and `[0]` on a string is its first *character* -
    # which made the check compare "c" against the real hash and always fail.
    $sumText = (Get-Content -Raw (Join-Path $tmp "$asset.sha256")).Trim()
    $expected = ((-split $sumText)[0]).ToLower()
    $actual   = (Get-FileHash -Algorithm SHA256 (Join-Path $tmp $asset)).Hash.ToLower()
    if ($expected -ne $actual) {
        throw "checksum mismatch for $asset`n  expected $expected`n  actual   $actual"
    }
    Write-Ok "sha256 $actual"

    Write-Step 'extracting'
    # Use the built-in bsdtar by absolute path. A session whose PATH puts a Git
    # for Windows / MSYS tar first gets a tar that is handed Windows paths and
    # fails with "Error is not recoverable".
    $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
    if (-not (Test-Path $tar)) { $tar = 'tar' }
    & $tar -xzf (Join-Path $tmp $asset) -C $tmp
    if ($LASTEXITCODE -ne 0) { throw "tar failed extracting $asset (exit $LASTEXITCODE)" }
    # Locate the binary rather than assuming it sits at the archive root.
    $found = Get-ChildItem -Path $tmp -Filter 'chrome-use.exe' -Recurse | Select-Object -First 1
    if (-not $found) { throw "chrome-use.exe not found inside $asset" }
    Copy-Item -Force $found.FullName (Join-Path $BinDir 'chrome-use.exe')
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    Write-Ok "installed $(Join-Path $BinDir 'chrome-use.exe')"
}

function Build-ChatGptUse {
    Write-Step 'building chatgpt-use (release)'
    # Note: `cargo build` writes target\ under the repo. On machines with Smart
    # App Control or a WDAC policy, executables produced under some user-writable
    # locations (Desktop included) are blocked from running; CARGO_TARGET_DIR can
    # point somewhere the policy allows.
    & cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit $LASTEXITCODE)" }
    Copy-Item -Force '.\target\release\chatgpt-use.exe' (Join-Path $BinDir 'chatgpt-use.exe')
    Write-Ok "installed $(Join-Path $BinDir 'chatgpt-use.exe')"
}

# --- preflight ---------------------------------------------------------------

$rust = Test-Rust
if (-not $rust -and -not $SkipBuild) {
    Write-Host "`nRust is required to build, and no working toolchain was found." -ForegroundColor Red
    Write-Host "Install it with:  winget install Rustlang.Rustup"
    Write-Host "then 'rustup default stable', and re-run this script."
    Write-Host "A C linker is also needed for the default windows-msvc target: any"
    Write-Host "Visual Studio 2022 edition with the 'Desktop development with C++'"
    Write-Host "workload satisfies it (Build Tools alone is enough)."
    if (Test-Path (Join-Path $BinDir 'chrome-use.exe')) {
        Write-Host "`nchrome-use is already installed, so you can skip -SkipBuild once Rust is in place."
    }
    exit 1
}
if ($rust) { Write-Ok "toolchain: $rust" }

if (-not (Test-Path (Join-Path $env:SystemRoot 'System32\tar.exe')) -and -not (Get-Command tar -ErrorAction SilentlyContinue)) {
    # tar ships with Windows 10 1803+; only an old build lacks it.
    Write-Host "`n'tar' is required to unpack the chrome-use release (built in since Windows 10 1803)." -ForegroundColor Red
    exit 1
}

Add-BinDirToPath

$installed = Join-Path $BinDir 'chrome-use.exe'
if (Test-Path $installed) {
    Write-Ok "chrome-use already at $installed (delete it to reinstall)"
} else {
    Install-ChromeUse (Get-LatestChromeUseTag)
}

if (-not $SkipBuild) { Build-ChatGptUse }

$chromeUse = Join-Path $BinDir 'chrome-use.exe'
$chatgptUse = Join-Path $BinDir 'chatgpt-use.exe'

Write-Step 'registering the chrome-use native messaging host'
& $chromeUse extension install
if ($LASTEXITCODE -ne 0) {
    Write-Host "    'chrome-use extension install' exited $LASTEXITCODE - check the Chrome Web Store" -ForegroundColor Yellow
    Write-Host "    extension step below, or run the command yourself for its output."
}

Write-Step 'generating the MCP auth token'
& $chatgptUse init

# Deliberately not a here-string: Windows PowerShell 5.1's tokenizer requires a
# here-string's terminator to be preceded by CRLF, so a file saved with LF-only
# line endings fails to parse with a confusing "missing terminator" error.
Write-Host ''
Write-Host '=== Setup complete ===' -ForegroundColor Green
Write-Host ''
Write-Host 'Next steps'
Write-Host '  1. Install the chrome-use browser extension:'
Write-Host '       https://chromewebstore.google.com/detail/knfcmbamhjmaonkfnjhldjedeobeafmk'
Write-Host '     then sign in to https://chatgpt.com in that Chrome profile.'
Write-Host '  2. Quick test (sidekick mode):'
Write-Host '       chatgpt-use ask "Say hello in one sentence"'
Write-Host '  3. MCP / closed loop:'
Write-Host '       .\start-work-mode.ps1 -Project "C:\path\to\your\app"'
Write-Host '     or by hand:'
Write-Host '       chatgpt-use mcp --port 8788 --cwd .'
Write-Host '     and expose it with cloudflared, then add it as a ChatGPT connector.'
Write-Host ''
Write-Host "Binaries: $BinDir"
