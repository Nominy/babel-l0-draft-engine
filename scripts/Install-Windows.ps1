[CmdletBinding()]
param(
    [string]$PythonExecutable = "",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [switch]$Recreate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvRoot = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

function Resolve-PythonLauncher {
    if ($PythonExecutable) {
        $configured = Get-Command $PythonExecutable -ErrorAction SilentlyContinue
        if (-not $configured) {
            throw "Configured Python executable was not found: $PythonExecutable"
        }
        return @{ FilePath = $configured.Source; Arguments = @() }
    }

    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        return @{ FilePath = $py.Source; Arguments = @() }
    }

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        return @{ FilePath = $python.Source; Arguments = @() }
    }

    throw "Python was not found. Install 64-bit Python 3.11 from python.org, including the launcher or PATH option, then rerun this script."
}

$launcher = Resolve-PythonLauncher
$versionCheck = "import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), 'Python 3.10-3.12 is required'; print(sys.version.split()[0])"
Invoke-Checked -FilePath $launcher.FilePath -Arguments ($launcher.Arguments + @("-c", $versionCheck))

if ($Recreate -and (Test-Path $VenvRoot)) {
    Remove-Item -Recurse -Force $VenvRoot
}
if (-not (Test-Path $VenvPython)) {
    Invoke-Checked -FilePath $launcher.FilePath -Arguments ($launcher.Arguments + @("-m", "venv", $VenvRoot))
}

Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools<82", "wheel")
Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--index-url", $TorchIndexUrl, "torch", "torchaudio")
Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--editable", $RepoRoot)

$envFile = Join-Path $RepoRoot ".env"
if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $RepoRoot ".env.example") $envFile
}

$cudaCheck = @"
import torch
if not torch.cuda.is_available():
    raise SystemExit('PyTorch installed, but CUDA is unavailable. Update the NVIDIA driver and rerun Install-Windows.ps1 -Recreate.')
print(f'CUDA ready: {torch.cuda.get_device_name(0)}; torch {torch.__version__}; CUDA {torch.version.cuda}')
"@
Invoke-Checked -FilePath $VenvPython -Arguments @("-c", $cudaCheck)

if (-not (Get-Command ffmpeg.exe -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg is not on PATH. Raw WAV inference works without it; install ffmpeg before enabling afftdn preprocessing."
}

Write-Host "Windows environment ready. Start the endpoint with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\Start-Windows.ps1"
