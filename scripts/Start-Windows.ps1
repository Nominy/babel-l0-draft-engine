[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$Port = 8767,
    [switch]$PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Windows environment is missing. Run .\scripts\Install-Windows.ps1 first."
}

function Import-DotEnv {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path $Path)) {
        return
    }
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        $separator = $trimmed.IndexOf("=")
        if ($separator -le 0) {
            throw "Invalid .env line: $line"
        }
        $name = $trimmed.Substring(0, $separator).Trim()
        $value = $trimmed.Substring($separator + 1).Trim()
        if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

Import-DotEnv (Join-Path $RepoRoot ".env")
if (-not $env:HF_HOME) {
    $env:HF_HOME = Join-Path $RepoRoot ".cache\huggingface"
}
if (-not $env:TMPDIR) {
    $env:TMPDIR = Join-Path $RepoRoot ".cache\tmp"
}
New-Item -ItemType Directory -Force $env:HF_HOME, $env:TMPDIR | Out-Null

$preflight = @"
import torch
import fastapi
import gigaam
import transformers
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable. Run Install-Windows.ps1 -Recreate after updating the NVIDIA driver.')
print(f'Ready: {torch.cuda.get_device_name(0)}; torch {torch.__version__}; CUDA {torch.version.cuda}')
"@
& $VenvPython -c $preflight
if ($LASTEXITCODE -ne 0) {
    throw "L0 Draft Engine preflight failed."
}
if ($PreflightOnly) {
    return
}

Write-Host "Starting L0 Draft Engine at http://127.0.0.1:$Port"
Write-Host "The first request downloads the configured public models. Press Ctrl+C to stop."
& $VenvPython -m uvicorn l0_draft_engine.app:app --host 127.0.0.1 --port $Port
if ($LASTEXITCODE -ne 0) {
    throw "L0 Draft Engine exited with code $LASTEXITCODE."
}
