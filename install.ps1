# chatgpt-use installer for Windows - downloads a prebuilt binary from GitHub
# Releases. No cargo, no token, no npm.
#
#   powershell -ExecutionPolicy Bypass -File .\install.ps1
#   iwr -useb https://raw.githubusercontent.com/RudraNarayanP/agenticcoding-with-chatgpt/main/install.ps1 | iex
#
# To build from source instead of installing a release, use setup-windows.ps1.

[CmdletBinding()]
param(
    [string]$Repo = 'RudraNarayanP/agenticcoding-with-chatgpt',
    # Empty resolves the latest release from the GitHub API.
    [string]$Version = '',
    [string]$InstallDir = (Join-Path $env:USERPROFILE '.local\bin'),
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

function Write-Step([string]$m) { Write-Host "==> $m" -ForegroundColor Cyan }

$asset = 'chatgpt-use-x86_64-pc-windows-msvc.zip'

if (-not $Version) {
    Write-Step 'resolving the latest release'
    try {
        $json = (Invoke-WebRequest -UseBasicParsing "https://api.github.com/repos/$Repo/releases/latest").Content
        if ($json -match '"tag_name":\s*"([^"]+)"') { $Version = $Matches[1] }
    } catch {
        Write-Host "    API call failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    if (-not $Version) {
        Write-Host "error: no release found on $Repo, and none was pinned with -Version." -ForegroundColor Red
        Write-Host 'This fork publishes its own releases; until the first tag exists,'
        Write-Host 'build from source instead:  .\setup-windows.ps1'
        exit 1
    }
}
Write-Host "==> chatgpt-use $Version (x86_64-pc-windows-msvc)"

$base = "https://github.com/$Repo/releases/download/$Version"
$tmp = Join-Path $env:TEMP "chatgpt-use-install-$Version"
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

try {
    Invoke-WebRequest -UseBasicParsing -Uri "$base/$asset" -OutFile (Join-Path $tmp $asset)
} catch {
    Write-Host "error: release $Version does not ship $asset" -ForegroundColor Red
    Write-Host "       $($_.Exception.Message)"
    exit 1
}

# The sidecar is optional (a manual release may omit it) but never skipped when
# present: this is an executable that ends up on PATH.
try {
    Invoke-WebRequest -UseBasicParsing -Uri "$base/$asset.sha256" -OutFile (Join-Path $tmp "$asset.sha256")
    $sumText = (Get-Content -Raw (Join-Path $tmp "$asset.sha256")).Trim()
    $expected = ((-split $sumText)[0]).ToLower()
    $actual = (Get-FileHash -Algorithm SHA256 (Join-Path $tmp $asset)).Hash.ToLower()
    if ($expected -ne $actual) {
        throw "checksum mismatch`n  expected $expected`n  actual   $actual"
    }
    Write-Host "    sha256 verified: $actual" -ForegroundColor DarkGray
} catch {
    if ($_.Exception.Message -match 'checksum mismatch') {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
        throw
    }
    Write-Host '    no .sha256 published with this release - installing unverified' -ForegroundColor Yellow
}

Write-Step 'installing'
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
# Built-in bsdtar (Windows 10 1803+) unzips as well as untars, and unlike a
# PATH-resolved tar it is not shadowed by a Git for Windows build that mishandles
# Windows paths.
$tar = Join-Path $env:SystemRoot 'System32\tar.exe'
if (-not (Test-Path $tar)) { $tar = 'tar' }
& $tar -xf (Join-Path $tmp $asset) -C $tmp
if ($LASTEXITCODE -ne 0) { throw "failed to unpack $asset (exit $LASTEXITCODE)" }

$found = Get-ChildItem -Path $tmp -Filter 'chatgpt-use.exe' -Recurse | Select-Object -First 1
if (-not $found) { throw "chatgpt-use.exe not found inside $asset" }

$dest = Join-Path $InstallDir 'chatgpt-use.exe'
if ((Test-Path $dest) -and -not $Force) {
    $existing = (Get-Item $dest).VersionInfo.FileVersion
    Write-Host "    chatgpt-use is already installed at $dest (version $existing); pass -Force to replace it" -ForegroundColor Yellow
} else {
    Copy-Item -Force $found.FullName $dest
    Write-Host "    installed $dest"
}

# Same reason as setup-windows.ps1: chatgpt-use looks for chrome-use on PATH, and
# without it every command fails at connect.
if (-not ((Get-Command chrome-use -ErrorAction SilentlyContinue) -or
          (Test-Path (Join-Path $InstallDir 'chrome-use.exe')))) {
    Write-Host ''
    Write-Host 'chatgpt-use needs chrome-use (it drives the browser). Install it:' -ForegroundColor Yellow
    Write-Host '  powershell -ExecutionPolicy Bypass -File .\setup-windows.ps1 -SkipBuild'
    Write-Host '  (or: iwr -useb https://raw.githubusercontent.com/leeguooooo/chrome-use/main/install.ps1 | iex)'
}

$parts = $env:Path -split ';'
if ($parts -notcontains $InstallDir) {
    Write-Host ''
    Write-Host "note: add $InstallDir to your PATH to use chatgpt-use from a new terminal:" -ForegroundColor Yellow
    Write-Host "  [Environment]::SetEnvironmentVariable('Path', '$InstallDir;' + [Environment]::GetEnvironmentVariable('Path','User'), 'User')"
}

Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
Write-Host ''
Write-Host "==> done. Try:  chatgpt-use --help" -ForegroundColor Green
